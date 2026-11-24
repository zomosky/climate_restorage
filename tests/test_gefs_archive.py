"""Full native GEFS dates may leave hot storage only after both products are verified."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import xarray as xr

from climate_restore.gefs_control import export_gefs_control
from climate_restore.processor import write_dataset
from climate_restore.config import ZarrOptions

DEPLOY = Path(__file__).parents[1] / "deploy"
sys.path.insert(0, str(DEPLOY))
DATE = "20260621"
STEPS = [*range(0, 241, 3), *range(246, 841, 6)]
INSTANT = ("d2m", "prmsl", "r2", "sp", "t2m", "u10", "u100", "v10", "v100")
INTERVAL = ("sdlwrf_avg", "sdswrf_avg", "tcc_avg", "tmax2m_max", "tmin2m_min", "tp_accum")
RECORD = {"status": "complete", "member_count": 5, "completed_at": "2026-11-24T05:52:05+00:00"}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def files_in(root):
    return {str(x.relative_to(root)): digest(x) for x in root.rglob("*") if x.is_file()}


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    base = tmp_path_factory.mktemp("gefs-full-archive-fixture")
    init = base / "raw" / "gefs-ens" / DATE / "00z"
    init.mkdir(parents=True)
    files = []
    for step in STEPS:
        p = init / f"f{step:03d}.subset.grib2"
        p.write_bytes(f"GRIB synthetic step {step:03d}7777".encode())
        files.append(dict(step_hours=step, path=f"/climate_data/gefs-ens/{DATE}/00z/{p.name}",
                          size_bytes=p.stat().st_size, sha256=digest(p),
                          records_selected=45 if step == 0 else 75))
    cache = init / ".members" / "c00"
    cache.mkdir(parents=True)
    (cache / "f000.grib2").write_bytes(b"preserve cached member")
    manifest = init / f"{DATE}_00z_gefs-ens.manifest.json"
    manifest.write_text(json.dumps(dict(schema_version=1, source={"name": "gefs-ens", "members": list(range(5))},
        date=DATE, cycle=0, init_time="2026-06-21T00:00:00+00:00",
        completed_at="2026-11-24T05:33:00+00:00", failures=[],
        variables=[{"name": n} for n in INSTANT + INTERVAL], files=files)))
    initial = np.datetime64("2026-06-21T00:00:00", "ns")
    step_axis = np.array(STEPS, dtype="timedelta64[h]").astype("timedelta64[ns]")
    times = initial + step_axis
    dims = ("step", "member", "latitude", "longitude")
    ds = xr.Dataset(coords=dict(time=initial, step=step_axis, valid_time=("step", times),
        member=np.arange(5), latitude=[40., 39.5], longitude=[117., 117.5]),
        attrs=dict(s2s_source="gefs-ens", gefs_complete_horizon=1, gefs_actual_member_count=5,
                   gefs_selected_members="[0, 1, 2, 3, 4]"))
    shape = (181, 5)
    end = np.broadcast_to(times[:, None], shape).copy()
    for i, name in enumerate(INSTANT + INTERVAL):
        present = np.ones(shape, dtype=bool)
        start = end.copy()
        finish = end.copy()
        data = np.broadcast_to(
            (np.arange(181)[:, None] + np.arange(5)[None, :] * 10 + i)[:, :, None, None],
            (181, 5, 2, 2)).astype("float32").copy()
        instant = name in INSTANT
        if not instant:
            present[0] = False
            start[1:] -= np.timedelta64(3, "h")
            start[0] = finish[0] = np.datetime64("NaT", "ns")
            data[0] = np.nan
        p, s, e = [f"{prefix}__{name}" for prefix in ("present", "start", "end")]
        ds[name] = xr.DataArray(data, dims=dims, attrs=dict(
            GRIB_stepType="instant" if instant else "avg",
            s2s_sample_present_coord=p, s2s_window_start_coord=s, s2s_window_end_coord=e))
        ds = ds.assign_coords({p: (("step","member"), present),
                              s: (("step","member"), start), e: (("step","member"), finish)})
    ens = base / "zarr" / "gefs-ens" / f"{DATE}_00z_gefs-ens.zarr"
    write_dataset(ds, ens, fmt="zarr", zarr_options=ZarrOptions(chunks={"step": -1, "member": 1}))
    (ens / ".gefs-manifest-sha256").write_text(digest(manifest))
    export_gefs_control(ens, base / "zarr")
    return base


@pytest.fixture
def case(tmp_path, sample):
    shutil.copytree(sample / "raw", tmp_path / "raw")
    shutil.copytree(sample / "zarr", tmp_path / "zarr")
    (tmp_path / "cold").mkdir()
    roots = dict(hot_root=tmp_path/"raw", cold_root=tmp_path/"cold", zarr_root=tmp_path/"zarr")
    args = dict(**roots, date=DATE, member_count=5, record=dict(RECORD), cutoff="20261025", guard=lambda: None)
    hot = roots["hot_root"]/"gefs-ens"/DATE
    cold = roots["cold_root"]/"gefs-ens"/DATE
    ens = roots["zarr_root"]/"gefs-ens"/f"{DATE}_00z_gefs-ens.zarr"
    ctrl = roots["zarr_root"]/"gefs-ctrl"/f"{DATE}_00z_gefs-ctrl.zarr"
    return args, hot, cold, ens, ctrl


def module():
    # Deferred import lets the red run report the missing feature as a test failure.
    import importlib.util
    assert importlib.util.find_spec("gefs_archive") is not None, "GEFS archive gate has not been implemented"
    import gefs_archive
    return gefs_archive


def archive_day(**args):
    """Simulate the production phases with local transport and temporary roots."""
    from gefs_archive_io import copy_to_cold
    archive = module()
    ready = archive.check_archive_day(**args)
    if ready["raw_storage"] == "hot":
        hot = args["hot_root"]/"gefs-ens"/args["date"]
        copy_to_cold(args["hot_root"], args["cold_root"], args["date"], ready["tree_sha256"],
                     args["guard"], populate=lambda stage: shutil.copytree(hot, stage, dirs_exist_ok=True))
    return archive.retire_day(**args, tree_sha256=ready["tree_sha256"])


def test_migration_preserves_all_raw_and_both_zarr_and_audit_can_read_cold(case):
    args, hot, cold, ens, ctrl = case
    original = files_in(hot)
    products = files_in(args["zarr_root"])
    assert archive_day(**args)["status"] == "archived"
    assert not hot.exists()
    assert files_in(cold) == original
    assert files_in(args["zarr_root"]) == products
    assert not list(cold.parent.glob(".gefs-stage-*"))
    result = module().audit_range(hot_root=args["hot_root"], cold_root=args["cold_root"],
        zarr_root=args["zarr_root"], start="20260620", end=DATE, member_count=5)
    assert result["complete"] == [DATE] and result["missing"] == ["20260620"]
    assert result["evidence"][0]["raw_storage"] == "cold"
    assert archive_day(**args)["status"] == "already_archived"


@pytest.mark.parametrize("problem", [
    "pending", "age_boundary", "no_control", "member_count", "short_manifest",
    "bad_raw", "no_chunk", "bad_control", "wrong_presence", "stale_manifest", "unfinished_cycle",
])
def test_incomplete_or_conflicting_data_keeps_hot_original(case, problem):
    args, hot, cold, ens, ctrl = case
    manifest = hot/"00z"/f"{DATE}_00z_gefs-ens.manifest.json"
    if problem == "pending":
        args["record"] = {"status": "download", "download_complete": True, "member_count": 5}
    elif problem == "age_boundary":
        args["cutoff"] = DATE
    elif problem == "no_control":
        shutil.rmtree(ctrl)
    elif problem in {"member_count", "short_manifest", "stale_manifest"}:
        obj = json.loads(manifest.read_text())
        if problem == "member_count":
            obj["source"]["members"] = [0, 1, 2, 3]
        elif problem == "short_manifest":
            obj["files"].pop()
        else:
            obj["completed_at"] = "2026-11-25T12:00:00+00:00"
        manifest.write_text(json.dumps(obj))
    elif problem == "bad_raw":
        p = manifest.parent / "f003.subset.grib2"
        p.write_bytes(p.read_bytes().replace(b"003", b"004"))
    elif problem == "no_chunk":
        for p in (ens/"t2m").iterdir():
            if not p.name.startswith("."):
                p.unlink()
    elif problem == "bad_control":
        import zarr
        group = zarr.open_group(ctrl, mode="a", use_consolidated=False)
        group["t2m"][1,0,0,0] = 999
    elif problem == "wrong_presence":
        # Keep the aggregate sample count and c00 unchanged, but swap f000
        # and f003 presence for p01. A missing required lead must be rejected.
        import zarr
        group = zarr.open_group(ens, mode="a", use_consolidated=False)
        group["present__tcc_avg"][0,1] = True
        group["present__tcc_avg"][1,1] = False
        group["tcc_avg"][0,1,:,:] = 100
        group["tcc_avg"][1,1,:,:] = np.nan
    elif problem == "unfinished_cycle":
        (hot/"12z").mkdir()
    original = files_in(hot)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        archive_day(**args)
    assert files_in(hot) == original
    assert not cold.exists()


def test_conflicting_cold_bytes_keep_hot_and_existing_cold(case):
    args, hot, cold, _, _ = case
    shutil.copytree(hot, cold)
    p = cold/"00z"/"f003.subset.grib2"
    p.write_bytes(p.read_bytes().replace(b"003", b"004"))
    original, conflicting = files_in(hot), files_in(cold)
    with pytest.raises(ValueError, match="copy|archive|cold"):
        archive_day(**args)
    assert files_in(hot) == original and files_in(cold) == conflicting


def test_corrupted_staged_copy_never_removes_hot(case, monkeypatch):
    args, hot, cold, _, _ = case
    original = files_in(hot)
    actual = shutil.copytree
    def corrupt(src, dst, *a, **kw):
        result = actual(src, dst, *a, **kw)
        if Path(src) == hot:
            (Path(dst)/"00z"/"f003.subset.grib2").write_bytes(b"wrong")
        return result
    monkeypatch.setattr(shutil, "copytree", corrupt)
    with pytest.raises(ValueError, match="copy"):
        archive_day(**args)
    assert files_in(hot) == original and not cold.exists()


def test_complete_hidden_stage_is_resumed_without_copying_raw_again(case, monkeypatch):
    args, hot, cold, _, _ = case
    stage = cold.parent/f".gefs-stage-{DATE}"
    shutil.copytree(hot, stage)
    def forbidden(*a, **kw):
        pytest.fail("previously verified staged copy should be reused")
    monkeypatch.setattr(shutil, "copytree", forbidden)
    assert archive_day(**args)["status"]=="archived"
    assert not hot.exists() and cold.exists() and not stage.exists()


def test_container_tar_stream_keeps_hidden_files_and_manifest_mtime(case, tmp_path, monkeypatch):
    from gefs_archive_io import stream_tar
    _, hot, _, _, _ = case
    binary=tmp_path/"bin"
    binary.mkdir()
    # Replace only Docker transport. The producer tar, binary pipe and consumer
    # tar all execute; data and archive timestamps must survive the real pipe.
    docker=binary/"docker"
    docker.write_text(f"#!{sys.executable}\nimport os, sys\n"
                      "i=sys.argv.index('tar'); os.execvp('tar', sys.argv[i:])\n")
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary}{os.pathsep}{os.environ['PATH']}")
    target=tmp_path/"tar-copy"
    target.mkdir()
    stream_tar("test-container", hot, target, 30)
    assert files_in(target)==files_in(hot)
    name=f"00z/{DATE}_00z_gefs-ens.manifest.json"
    assert int((hot/name).stat().st_mtime)==int((target/name).stat().st_mtime)


def test_mounted_storage_is_rechecked_before_hot_removal(case):
    args, hot, cold, _, _ = case
    original = files_in(hot)
    calls = 0
    def lost_mount():
        nonlocal calls
        calls += 1
        if calls >= 5:
            raise OSError("mount changed")
    args["guard"] = lost_mount
    with pytest.raises(OSError, match="mount changed"):
        archive_day(**args)
    assert files_in(hot) == original


def test_resume_after_cold_publication_and_partial_hot_cleanup(case):
    args, hot, cold, _, _ = case
    shutil.copytree(hot, cold)
    original = files_in(cold)
    retired = hot.parent/f".gefs-retired-{DATE}"
    hot.rename(retired)
    (retired/"00z"/"f003.subset.grib2").unlink()
    assert archive_day(**args)["status"] == "archived"
    assert not hot.exists() and not retired.exists()
    assert files_in(cold) == original


def test_audit_rechecks_cold_path_when_hot_day_is_retired_mid_read(case, monkeypatch):
    args, hot, cold, _, _ = case
    archive = module()
    original = archive.verify_day
    def concurrent_move(day, *a, **kw):
        if day == hot:
            shutil.copytree(hot, cold)
            shutil.rmtree(hot)
            raise FileNotFoundError("hot path was retired")
        return original(day, *a, **kw)
    monkeypatch.setattr(archive, "verify_day", concurrent_move)
    result = archive.audit_range(hot_root=args["hot_root"], cold_root=args["cold_root"],
        zarr_root=args["zarr_root"], start=DATE,end=DATE,member_count=5)
    assert result["complete"]==[DATE] and not result["problems"]
    assert result["evidence"][0]["raw_storage"]=="cold"


def test_scheduler_skips_busy_shared_lock_and_never_runs_outside_window(tmp_path):
    import fcntl
    assert module()  # red before the worker exists
    import gefs_coldmove
    lock = tmp_path/"gefs.lock"
    cfg = dict(lock_file=str(lock))
    history = {}
    def forbidden(*args, **kwargs):
        pytest.fail("Docker worker must not start")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert gefs_coldmove.tick(cfg, history, datetime(2026,11,24,6,5,tzinfo=timezone.utc), run=forbidden)["status"] == "busy"
    assert gefs_coldmove.tick(cfg, history, datetime(2026,11,24,6,10,tzinfo=timezone.utc), run=forbidden)["status"] == "yield"


@pytest.mark.parametrize("copy_fails", [False, True])
def test_scheduler_uses_container_to_read_and_retire_but_host_to_copy(case, tmp_path, monkeypatch, copy_fails):
    """The production cold bind is read-only; the host cannot read the raw GRIBs."""
    from types import SimpleNamespace
    import gefs_coldmove
    args, hot, cold, _, _ = case
    nested_zarr=args["hot_root"]/"climate_data_storage"/"zarr"
    nested_zarr.parent.mkdir()
    args["zarr_root"].rename(nested_zarr)
    args["zarr_root"]=nested_zarr
    live_state = tmp_path/"live-state.json";live_state.write_text('{"dates": {}}')
    history_state = tmp_path/"history-state.json";history_state.write_text(json.dumps({"dates": {DATE: RECORD}}))
    live = dict(lock_file=str(tmp_path/"gefs.lock"), host_data_root=str(args["hot_root"]),
        host_cold_root=str(args["cold_root"]), host_zarr_root=str(args["zarr_root"]),
        data_root="/climate_data", cold_data_root="/save_data/climate_data", zarr_root="/climate_data/zarr",
        release_root="/workspace/gefs-release", start_date="20261105", state_file=str(live_state), member_count=5)
    history = dict(history_start="20260601", history_end="20261104", state_file=str(history_state),
        live_config="/config", member_count=5)
    monkeypatch.setattr(gefs_coldmove, "mount_identity", lambda path: [2 if path == args["cold_root"] else 1, 1])
    phases = []
    def transport(cmd, **kw):
        if cmd[0] == "docker":
            phase = next((x for x in ("check", "retire") if x in cmd), None)
            assert phase is not None, "read-only container must not copy to cold"
            phases.append(phase)
            if phase == "check":
                result = module().check_archive_day(**args)
                return SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr="")
            result = module().retire_day(**args, tree_sha256=cmd[cmd.index("--tree-sha256")+1])
            return SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr="")
        assert "gefs_archive_io.py" in " ".join(cmd) and "copy" in cmd and "--container" in cmd
        phases.append("copy")
        if copy_fails:
            return SimpleNamespace(returncode=1)
        from gefs_archive_io import copy_to_cold
        copy_to_cold(args["hot_root"], args["cold_root"], DATE,
                     cmd[cmd.index("--tree-sha256")+1], args["guard"],
                     populate=lambda stage: shutil.copytree(hot, stage, dirs_exist_ok=True))
        return SimpleNamespace(returncode=0)
    result = gefs_coldmove.tick(live, history, datetime(2026,11,24,6,5,tzinfo=timezone.utc),
                                run=transport, monotonic=lambda: 0)
    if copy_fails:
        assert phases == ["check", "copy"]
        assert result["status"] == "deferred" and hot.exists() and not cold.exists()
    else:
        assert phases == ["check", "copy", "retire"]
        assert result["status"] == "archived" and cold.exists() and not hot.exists()
