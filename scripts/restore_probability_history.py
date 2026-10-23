#!/usr/bin/env python3
"""Verify a historical manifest, stage a readable product, then publish atomically."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import xarray as xr

from climate_restore.coverage_index import update_store
from climate_restore.config import ZarrOptions
from climate_restore.manifest import load_manifest, verify
from climate_restore.processor import process_init
from climate_restore.sources.probabilities import ProbabilityAdapter


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_dataset(ds, manifest, adapter):
    init=datetime.fromisoformat(manifest.init_time).astimezone(timezone.utc).replace(tzinfo=None)
    if np.datetime64(ds.time.values, 'ns') != np.datetime64(init, 'ns'):
        raise ValueError('historical probability GRIB init differs from manifest')
    adapter.validate_product(ds)


def restore(path: Path, output: Path) -> dict:
    manifest = load_manifest(path)
    adapter = ProbabilityAdapter(manifest.source_name)
    adapter.configure_manifest(manifest)
    files = verify(manifest, path.parent)
    if not files:
        if manifest.probability_availability and manifest.probability_availability.get("status") == "unavailable":
            return dict(status="unavailable", source=manifest.source_name, date=manifest.date, cycle=manifest.cycle)
        raise ValueError("manifest has no probability data")
    for entry, raw in zip(manifest.files, files):
        if raw.stat().st_size != entry.size_bytes or digest(raw) != entry.sha256:
            raise ValueError(f"raw checksum mismatch: {raw}")
    name = f"{manifest.date}_{manifest.cycle:02}z_{manifest.source_name}"
    store = output / manifest.source_name / f"{name}.zarr"
    receipt = dict(source=manifest.source_name, date=manifest.date, cycle=manifest.cycle,
                   manifest_sha256=digest(path), verified_at=datetime.now(timezone.utc).isoformat())
    if (store / ".zmetadata").is_file() and (store / ".zmetadata").stat().st_mtime_ns >= path.stat().st_mtime_ns:
        with xr.open_zarr(store, consolidated=True, chunks=None) as ds:
            ds.load()
            validate_dataset(ds,manifest,adapter)
        update_store(store)
        receipt.update(status="done", resumed=True, variables=len(ds.data_vars))
        return receipt
    staging = store.with_name(f".{name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    process_init(files, adapter=adapter, bbox=(70, 140, 15, 55), out_path=staging,
                 workers=1, output_format="zarr", zarr_options=ZarrOptions())
    with xr.open_zarr(staging, consolidated=True, chunks=None) as ds:
        ds.load()
        validate_dataset(ds,manifest,adapter)
        receipt.update(status="done", resumed=False, variables=len(ds.data_vars),
                       samples=sum(int(ds[v].attrs["probability_sample_count"]) for v in ds.data_vars),
                       probability_status=(manifest.probability_availability or {}).get("status", "complete"))
    (staging / "history-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    # The scheduler prevents concurrent writers. New products publish with one
    # rename; replacing a prior product retains a rollback directory until done.
    if store.exists():
        backup = store.with_name(f".{name}.previous")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(store, backup)
        try:
            os.replace(staging, store)
        except BaseException:
            os.replace(backup, store)
            raise
        shutil.rmtree(backup)
    else:
        os.replace(staging, store)
    update_store(store)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/climate_data/climate_data_storage/zarr"))
    args = parser.parse_args()
    print(json.dumps(restore(args.manifest, args.output)))
