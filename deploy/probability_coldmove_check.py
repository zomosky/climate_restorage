#!/usr/bin/env python3
"""Read-only cold-move gate; run inside the climate_restorage uv environment.

Verify every downloaded cycle of one day. A completion flag alone is never
enough: require fresh, readable data, exact native windows, and physical chunks.
With --cold-root, also hash every original/copied file before deleting originals.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import xarray as xr

from climate_restore.sources.probabilities import ProbabilityAdapter, expected_samples

SOURCES = ("ifs-ens-probabilities", "aifs-ens-probabilities")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"copy directory missing or symlink: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink is not permitted in raw data: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = sha256(path)
        elif not path.is_dir():
            raise ValueError(f"unexpected raw file type: {path}")
    return result


def verify_chunks(store: Path, metadata: dict, variables: dict[str, int]) -> None:
    # Coordinate chunks may be legitimately omitted when their encoded values
    # equal the Zarr fill value (e.g. the scalar reference time encoded as zero).
    # Probability fields use NaN fill and every spatial chunk has real samples.
    for variable in variables:
        key = f"{variable}/.zarray"
        if key not in metadata:
            raise ValueError(f"missing probability array: {key}")
        array = metadata[key]
        shape, chunks = array["shape"], array["chunks"]
        if len(shape) != len(chunks) or any(n <= 0 for n in shape + chunks):
            raise ValueError(f"invalid array shape/chunks: {key}")
        ranges = [range(math.ceil(n / c)) for n, c in zip(shape, chunks)]
        separator = array.get("dimension_separator", ".")
        directory = store / key.removesuffix("/.zarray")
        for index in itertools.product(*ranges):
            chunk = directory / (separator.join(map(str, index)) if index else "0")
            if not chunk.is_file():
                raise ValueError(f"missing physical chunk: {chunk}")


def verify_day(*, raw_root: Path, zarr_root: Path, source: str, date: str,
               cold_root: Path | None = None) -> dict:
    if source not in SOURCES or not re.fullmatch(r"\d{8}", date):
        raise ValueError("invalid probability source/date")
    datetime.strptime(date, "%Y%m%d")
    day = raw_root / source / date
    if not day.is_dir() or day.is_symlink():
        raise ValueError(f"raw day missing or symlink: {day}")
    cycles = sorted(p for p in day.iterdir() if p.is_dir() and p.name.endswith("z"))
    if not cycles:
        raise ValueError(f"no raw cycles: {day}")
    queue_path = raw_root / "_probability_backfill" / "queue.json"
    if queue_path.is_file():
        queue = json.loads(queue_path.read_text())
        model = "ifs" if source == "ifs-ens-probabilities" else "aifs-ens"
        planned = {f'{item["cycle"]:02}z' for item in queue["inits"] if item["model"] == model and item["date"] == date}
        if not planned <= {directory.name for directory in cycles}:
            raise ValueError(f"historical day still has queued cycles: {source}/{date}")
    checked = []
    for directory in cycles:
        allowed_cycles = {"00z", "12z", "06z", "18z"} if source == "aifs-ens-probabilities" else {"00z", "12z"}
        if directory.name not in allowed_cycles or directory.is_symlink():
            raise ValueError(f"unexpected probability cycle: {directory}")
        cycle = int(directory.name[:2])
        manifest = directory / f"{date}_{cycle:02}z_{source}.manifest.json"
        raw = json.loads(manifest.read_text())
        init = datetime.strptime(f"{date}{cycle:02}", "%Y%m%d%H").replace(tzinfo=timezone.utc)
        if (raw.get("schema_version") != 1 or raw.get("source", {}).get("name") != source
            or raw.get("date") != date or raw.get("cycle") != cycle
            or not raw.get("completed_at") or raw.get("failures") != []
            or datetime.fromisoformat(raw["init_time"]) != init):
            raise ValueError(f"manifest not ready or wrong init: {manifest}")
        files = raw.get("files", [])
        contract = raw.get("probability_availability")
        wanted = {v: len(times) for v, times in expected_samples(source).items()}
        wanted_bundles = {240, 360}
        if contract is not None:
            if contract.get("policy") != "historical_available" or contract.get("schema_version") != 1:
                raise ValueError(f"invalid history receipt: {manifest}")
            bundles = contract.get("bundles", [])
            if len(bundles) != 2 or {b["bundle"] for b in bundles} != {240,360}:
                raise ValueError(f"history bundle discovery incomplete: {manifest}")
            for bundle in bundles:
                if bundle.get("status") != "missing_bundle":
                    index = directory / f'f{bundle["bundle"]}.upstream.index'
                    if sha256(index) != bundle.get("index_sha256"):
                        raise ValueError(f"historical upstream index changed: {index}")
            wanted_bundles = {b["bundle"] for b in bundles if b.get("selected_records", 0)}
            wanted = {v: len(times) for v,times in contract.get("available_samples", {}).items()}
        if len(files) != len(wanted_bundles) or {f["step_hours"] for f in files} != wanted_bundles:
            raise ValueError(f"probability bundles incomplete: {manifest}")
        declared_files = {Path(f["path"]).name for f in files}
        actual_files = {p.name for p in directory.glob("*.grib2")}
        if actual_files != declared_files or list(directory.glob("*.part")):
            raise ValueError(f"unverified raw GRIB or unfinished download: {directory}")
        for entry in files:
            path = directory / Path(entry["path"]).name
            if (path.is_symlink() or path.stat().st_size != entry["size_bytes"]
                or sha256(path) != entry["sha256"]):
                raise ValueError(f"raw checksum mismatch: {path}")
        if not files:
            # No original probability GRIB exists for this discovered init.
            # Preserve its proven unavailability and index files in the archive.
            if contract is None or contract.get("status") != "unavailable" or wanted:
                raise ValueError(f"empty historical probability product: {manifest}")
            checked.append(cycle)
            continue
        store = zarr_root / source / f"{date}_{cycle:02}z_{source}.zarr"
        marker = store / ".zmetadata"
        if marker.stat().st_mtime_ns < manifest.stat().st_mtime_ns:
            raise ValueError(f"processed product older than manifest: {store}")
        metadata = json.loads(marker.read_text())["metadata"]
        attrs = metadata[".zattrs"]
        if (attrs.get("probability_complete") != 1 or attrs.get("probability_source") != source
            or attrs.get("probability_bundles") != [240, 360]
            or json.loads(attrs.get("probability_expected_samples", "null")) != wanted
            or (contract is not None and json.loads(attrs.get("probability_availability", "null")) != contract)):
            raise ValueError(f"probability receipt incomplete: {store}")
        verify_chunks(store, metadata, wanted)
        with xr.open_zarr(store, consolidated=True, chunks=None) as ds:
            # Eager reads expose compressed-chunk corruption; missing chunks were
            # checked above because Zarr would otherwise silently supply fill values.
            ds.load()
            if np.datetime64(ds.time.values, "ns") != np.datetime64(init.replace(tzinfo=None), "ns"):
                raise ValueError(f"wrong processed init: {store}")
            if any(ds[v].attrs.get("units") != "%" or ds[v].attrs.get("zrread_kind") != "probability" for v in ds.data_vars):
                raise ValueError(f"non-probability variable: {store}")
            for coordinate in ("latitude", "longitude"):
                values = ds[coordinate].values
                delta = np.diff(values)
                if (values.size < 2 or not np.isfinite(values).all()
                    or not (np.all(delta > 0) or np.all(delta < 0))):
                    raise ValueError(f"invalid spatial coordinate: {coordinate}")
            adapter = ProbabilityAdapter(source)
            adapter.historical_availability = contract
            adapter.validate_product(ds)
        checked.append(cycle)
    if cold_root is not None:
        original = tree_hashes(day)
        copied = tree_hashes(cold_root / source / date)
        if not original or original != copied:
            raise ValueError(f"cold copy file list/checksum mismatch: {source}/{date}")
    return dict(source=source, date=date, cycles=checked, product_verified=True,
                copy_verified=cold_root is not None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--zarr-root", type=Path, required=True)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--cold-root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_day(**vars(args))
    except Exception as exc:
        print(json.dumps(dict(source=args.source, date=args.date, error=f"{type(exc).__name__}: {exc}")), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
