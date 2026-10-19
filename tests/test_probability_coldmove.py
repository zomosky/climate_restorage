"""Cold storage must retain raw data whenever the processed product is incomplete.

Failure cases: stale/failed manifests, missing or unreadable chunks (including
all chunks of a variable), wrong windows, and a damaged cold copy. Tests use
temporary raw files and tiny, real Zarr stores; they never touch production.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from climate_restore.processor import write_dataset
from climate_restore.config import ZarrOptions
from climate_restore.sources.probabilities import ProbabilityAdapter


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "probability_coldmove_check", Path(__file__).parents[1] / "deploy/probability_coldmove_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def product(tmp_path):
    source, date = "ifs-ens-probabilities", "20260901"
    raw, zarr, cold = (tmp_path / name for name in ("raw", "zarr", "cold"))
    cycle = raw / source / date / "00z"
    cycle.mkdir(parents=True)
    entries = []
    for bundle in (240, 360):
        p = cycle / f"f{bundle}.subset.grib2"
        p.write_bytes(b"GRIB" + str(bundle).encode() + b"7777")
        entries.append(dict(step_hours=bundle, path=f"/climate_data/{source}/{date}/00z/{p.name}",
                            size_bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    manifest = cycle / f"{date}_00z_{source}.manifest.json"
    manifest.write_text(json.dumps(dict(schema_version=1, source={"name":source}, date=date,
        cycle=0, init_time="2026-09-01T00:00:00+00:00", completed_at="2026-09-01T09:00:00Z",
        failures=[], files=entries)))
    time = np.datetime64("2026-09-01T00:00:00", "ns")
    step = np.arange(24, 361, 12).astype("timedelta64[h]").astype("timedelta64[ns]")
    ds = xr.Dataset(coords=dict(time=time, step=step, latitude=[30.,31.], longitude=[100.,101.,102.]))
    for v in ("tpg20", "tpg50", "tpg100", "10fgg15", "10fgg25"):
        start, end = f"probability_window_start__{v}", f"probability_window_end__{v}"
        ds[v] = xr.DataArray(np.full((29,2,3), 25., dtype="float32"),
            dims=("step","latitude","longitude"), attrs=dict(units="%", zrread_kind="probability",
            probability_window_hours=24, probability_window_start_coord=start, probability_window_end_coord=end))
        ds = ds.assign_coords({start:("step", time+step-np.timedelta64(24,"h")), end:("step", time+step)})
    ds = ds.assign_coords(valid_time=("step", time+step))
    ProbabilityAdapter(source).validate_product(ds)
    store = zarr / source / f"{date}_00z_{source}.zarr"
    write_dataset(ds, store, fmt="zarr", zarr_options=ZarrOptions(chunks={"step":-1,"latitude":1,"longitude":2}))
    return dict(raw_root=raw, zarr_root=zarr, source=source, date=date), manifest, store, cold


def test_complete_product_and_matching_copy_pass(product):
    args, _, store, cold = product
    checker = load_checker()
    assert checker.verify_day(**args)["cycles"] == [0]
    shutil.copytree(args["raw_root"] / args["source"] / args["date"], cold / args["source"] / args["date"])
    assert checker.verify_day(**args, cold_root=cold)["copy_verified"] is True
    assert store.is_dir()


def test_history_day_waits_for_all_queued_cycles_before_moving(product):
    args,manifest,store,cold=product
    queue=args["raw_root"]/'_probability_backfill/queue.json'
    queue.parent.mkdir()
    queue.write_text(json.dumps(dict(inits=[dict(model='ifs',date=args['date'],cycle=c) for c in (0,12)])))
    with pytest.raises(ValueError,match='still has queued cycles'):
        load_checker().verify_day(**args)
    assert manifest.is_file() and store.is_dir()


@pytest.mark.parametrize("failure", ["one_chunk", "all_chunks", "unreadable_chunk", "window", "range", "stale", "flag", "failed_manifest", "raw_hash", "missing_cycle"])
def test_incomplete_product_blocks_move(product, failure):
    args, manifest, store, _ = product
    if failure in {"one_chunk", "all_chunks", "unreadable_chunk"}:
        chunks = sorted(p for p in (store/"tpg50").iterdir() if not p.name.startswith("."))
        if failure == "unreadable_chunk":
            chunks[0].write_bytes(b"invalid compressed chunk")
        else:
            for p in chunks if failure == "all_chunks" else chunks[:1]: p.unlink()
    elif failure in {"window", "range"}:
        import zarr
        root = zarr.open_group(store, mode="a")
        if failure == "window": root["probability_window_start__tpg50"][0] += 1
        else: root["tpg50"][0,0,0] = 101
    elif failure == "stale":
        os.utime(manifest, (store.stat().st_mtime+10, store.stat().st_mtime+10))
    elif failure == "flag":
        p = store/".zmetadata"
        meta = json.loads(p.read_text());meta["metadata"][".zattrs"]["probability_complete"] = 0
        p.write_text(json.dumps(meta))
    elif failure == "failed_manifest":
        meta = json.loads(manifest.read_text());meta["failures"] = [{"step":360}]
        manifest.write_text(json.dumps(meta))
    elif failure == "raw_hash":
        (manifest.parent/"f240.subset.grib2").write_bytes(b"GRIBbad7777")
    else:
        (manifest.parent.parent/"12z").mkdir()
    with pytest.raises((ValueError, OSError, RuntimeError)):
        load_checker().verify_day(**args)
    assert manifest.exists(), "failed verification must never remove raw data"


def test_cold_copy_with_same_size_but_changed_bytes_is_rejected(product):
    args, manifest, _, cold = product
    dest = cold/args["source"]/args["date"]
    shutil.copytree(manifest.parent.parent, dest)
    p = dest/"00z/f240.subset.grib2"
    p.write_bytes(p.read_bytes().replace(b"240", b"241"))
    with pytest.raises(ValueError, match="copy"):
        load_checker().verify_day(**args, cold_root=cold)
    assert manifest.exists()


def test_default_shell_policy_includes_both_sources():
    text = (Path(__file__).parents[1]/"deploy/forecast-coldmove.sh").read_text()
    assert "ifs-ens-probabilities aifs-ens-probabilities" in text
    assert 'KEEP_DAYS="${KEEP_DAYS:-30}"' in text


def test_verified_partial_history_can_archive_but_undeclared_raw_is_rejected(product):
    args,manifest,store,_=product
    from climate_restore.sources.probabilities import expected_samples
    with xr.open_zarr(store,consolidated=True,chunks=None) as opened:
        ds=opened[["tpg50"]].isel(step=slice(0,2)).load()
    full=expected_samples(args["source"])
    actual={"tpg50":[24,36]}
    contract=dict(schema_version=1,policy="historical_available",status="partial",
        expected_samples={v:sorted(s) for v,s in full.items()},available_samples=actual,
        missing_variables=sorted(set(full)-set(actual)),
        missing_samples={v:sorted(s-set(actual.get(v,[]))) for v,s in full.items()},bundles=[])
    for bundle,status,events in ((240,"available",[["tpg50","0-24"],["tpg50","12-36"]]),(360,"no_probability_fields",[])):
        index=manifest.parent/f"f{bundle}.upstream.index"
        index.write_text("historical upstream index\n")
        contract["bundles"].append(dict(bundle=bundle,status=status,available_events=events,
            selected_records=len(events),index_sha256=hashlib.sha256(index.read_bytes()).hexdigest()))
    raw=json.loads(manifest.read_text())
    raw["files"]=raw["files"][:1]
    (manifest.parent/"f360.subset.grib2").unlink()
    raw["probability_availability"]=contract
    manifest.write_text(json.dumps(raw))
    adapter=ProbabilityAdapter(args["source"]);adapter.historical_availability=contract
    adapter.validate_product(ds)
    write_dataset(ds,store,fmt="zarr",zarr_options=ZarrOptions())
    assert load_checker().verify_day(**args)["product_verified"]
    (manifest.parent/"unverified.grib2").write_bytes(b"GRIB7777")
    with pytest.raises(ValueError,match="unverified raw"):
        load_checker().verify_day(**args)


@pytest.mark.skipif(sys.platform != "linux", reason="production script requires GNU date and flock")
@pytest.mark.parametrize("case", ["move", "dry_run", "bad_chunk", "bad_copy", "busy_lock", "too_recent"])
def test_shell_moves_only_verified_old_raw_data(product, tmp_path, case):
    """Exercise the real shell/tar/gate/deletion flow in isolated directories."""
    import fcntl
    args, manifest, store, cold = product
    project = Path(__file__).parents[1]
    binary = tmp_path/"bin"
    binary.mkdir()
    docker = binary/"docker"
    # This adapter replaces container transport only. The real checker, GNU tar,
    # locks, age filter, and shell branch all execute, with deletion confined to
    # this test's temporary raw directory.
    docker.write_text(f"#!{sys.executable}\n" + '''import os,sys,subprocess,shutil
from pathlib import Path
args=sys.argv[1:]
if args[:2]==["exec","-w"]:
    if "--cold-root" in args and os.environ.get("TEST_BAD_COPY")=="1":
        root=Path(args[args.index("--cold-root")+1])
        p=next(root.rglob("f240.subset.grib2"))
        p.write_bytes(p.read_bytes().replace(b"240",b"241"))
    pos=args.index("python")
    raise SystemExit(subprocess.call([sys.executable,*args[pos+1:]]))
if args[:4]==["exec","zhangmy-dev","rm","-rf"]:
    p=Path(args[4]).resolve()
    assert p.is_relative_to(Path(os.environ["TEST_RAW_ROOT"]).resolve())
    shutil.rmtree(p)
    raise SystemExit(0)
raise SystemExit(99)
''')
    docker.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", SOURCES=args["source"],
        HOT=str(args["raw_root"]), CHOT=str(args["raw_root"]), ZARR=str(args["zarr_root"]),
        CZARR=str(args["zarr_root"]), COLD=str(cold), CCOLD=str(cold), RESTORE_DIR=str(project),
        LOCK=str(tmp_path/"cold.lock"), PROBABILITY_LOCK=str(tmp_path/"prob.lock"),
        LOG=str(tmp_path/"move.log"), TEST_RAW_ROOT=str(args["raw_root"]),
        DRY_RUN="1" if case=="dry_run" else "0", KEEP_DAYS="3650" if case=="too_recent" else "30",
        TEST_BAD_COPY="1" if case=="bad_copy" else "0")
    if case=="bad_chunk":
        next(p for p in (store/"tpg50").iterdir() if not p.name.startswith(".")).unlink()
    before = load_checker().tree_hashes(store)
    with (tmp_path/"prob.lock").open("w") as lock:
        if case=="busy_lock": fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        result = subprocess.run(["bash", str(project/"deploy/forecast-coldmove.sh")],
            cwd=project, env=env, capture_output=True, text=True, timeout=60)
    log = (tmp_path/"move.log").read_text() if (tmp_path/"move.log").exists() else result.stdout
    assert result.returncode == (1 if case in {"bad_chunk","bad_copy"} else 0), log
    assert load_checker().tree_hashes(store) == before, "Zarr must stay unchanged on the hot disk"
    if case=="move":
        assert not manifest.parent.parent.exists()
        assert (cold/args["source"]/args["date"]/"00z"/manifest.name).exists()
    else:
        assert manifest.exists(), log
        if case!="bad_copy": assert not cold.exists()
