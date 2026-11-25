#!/usr/bin/env python3
"""Verify GEFS products, archive raw files, and audit both storage tiers.

The host scheduler holds the applicable GEFS lock during production moves.
The cold date is published atomically before the hot date is retired.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import xarray as xr

from climate_restore.manifest import load_manifest
from climate_restore.processor import _verify_zarr_chunks
from climate_restore.sources.gefs_s2s import ALL_FIELDS, INSTANT_FIELDS, STEPS, GefsEnsAdapter
from gefs_publish import validate_product
from gefs_archive_io import (SOURCE, check_mounts, fsync_path, parse_day, require_directory,
                             sha256, tree_fingerprint, tree_hashes)
CONTROL = "gefs-ctrl"


def manifest_path(day, date):
    return day / "00z" / f"{date}_00z_{SOURCE}.manifest.json"


def product_paths(zarr_root, date):
    return [zarr_root / source / f"{date}_00z_{source}.zarr" for source in (SOURCE, CONTROL)]


def verify_day(day, zarr_root, date, member_count, *, raw_hashes=None, eager=False):
    """Validate same-directory raw paths and both products' actual provenance."""
    initial = np.datetime64(parse_day(date), "ns")
    if type(member_count) is not int or not 1 <= member_count <= 31:
        raise ValueError("invalid GEFS member count")
    members = list(range(member_count))
    require_directory(day)
    require_directory(day / "00z")
    cycles = {p.name for p in day.iterdir() if p.is_dir() and p.name.endswith("z")}
    if cycles != {"00z"}:
        raise ValueError(f"unexpected or unfinished GEFS cycle: {day}")
    path = manifest_path(day, date)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"GEFS manifest not ready: {path}")
    manifest = load_manifest(path)
    if (manifest.date != date or manifest.cycle != 0 or manifest.source_name != SOURCE
            or not manifest.completed_at or manifest.failures
            or manifest.source_metadata.get("members") != members
            or {item["name"] for item in manifest.variables} != ALL_FIELDS
            or [entry.step_hours for entry in manifest.files] != list(STEPS)):
        raise ValueError(f"incomplete GEFS manifest: {path}")
    GefsEnsAdapter().configure_manifest(manifest)
    expected_files = set()
    for entry in manifest.files:
        step = entry.step_hours
        name = f"f{step:03d}.subset.grib2"
        raw = path.parent / name
        expected_records = (len(INSTANT_FIELDS) if step == 0 else len(ALL_FIELDS)) * member_count
        if (entry.path.name != name or not raw.is_file() or raw.is_symlink()
                or entry.size_bytes <= 0 or raw.stat().st_size != entry.size_bytes
                or entry.records_selected != expected_records
                or not re.fullmatch(r"[0-9a-f]{64}", entry.sha256)):
            raise ValueError(f"incomplete GEFS raw file: {raw}")
        if raw_hashes is not None and raw_hashes.get(str(raw.relative_to(day))) != entry.sha256:
            raise ValueError(f"GEFS raw checksum mismatch: {raw}")
        expected_files.add(name)
    if {p.name for p in path.parent.glob("*.grib2")} != expected_files or list(path.parent.glob("*.part")):
        raise ValueError(f"unexpected or unfinished GEFS raw files: {path.parent}")
    fingerprint = sha256(path)
    if raw_hashes is not None and raw_hashes.get(str(path.relative_to(day))) != fingerprint:
        raise ValueError("GEFS manifest changed during verification")
    ens, ctrl = product_paths(zarr_root, date)
    for store in (ens, ctrl):
        require_directory(store)
        if not (store / ".zmetadata").is_file() or _verify_zarr_chunks(store):
            raise ValueError(f"incomplete Zarr product: {store}")
    if (ens / ".gefs-manifest-sha256").read_text().strip() != fingerprint:
        raise ValueError("GEFS ensemble provenance differs from manifest")
    native_steps = np.asarray(STEPS, dtype="timedelta64[h]")
    for source, store, expected_members in ((SOURCE, ens, members), (CONTROL, ctrl, [0])):
        with xr.open_zarr(store, consolidated=True, chunks=None) as dataset:
            if (dataset.attrs.get("s2s_source") != source
                    or dataset.attrs.get("gefs_complete_horizon") != 1
                    or dataset.attrs.get("gefs_actual_member_count") != len(expected_members)
                    or dataset.member.values.tolist() != expected_members
                    or not np.array_equal(dataset.step.values, native_steps)
                    or np.datetime64(dataset.time.values, "ns") != initial
                    or not np.array_equal(dataset.valid_time.values, initial + native_steps)
                    or set(dataset.data_vars) != ALL_FIELDS):
                raise ValueError(f"GEFS product coverage mismatch: {store}")
            if source == CONTROL and dataset.attrs.get("gefs_parent_metadata_sha256") != sha256(ens / ".zmetadata"):
                raise ValueError("GEFS control provenance differs from ensemble")
            if eager:
                shape = (len(STEPS), len(expected_members))
                valid = np.broadcast_to((initial + native_steps)[:, None], shape)
                for name in sorted(ALL_FIELDS):
                    attrs = dataset[name].attrs
                    present = dataset[attrs["s2s_sample_present_coord"]].transpose("step", "member").values
                    start = dataset[attrs["s2s_window_start_coord"]].transpose("step", "member").values
                    end = dataset[attrs["s2s_window_end_coord"]].transpose("step", "member").values
                    expected = np.ones(shape, dtype=bool)
                    instant = name in INSTANT_FIELDS
                    if not instant:
                        expected[0] = False
                    if (present.dtype != bool or not np.array_equal(present, expected)
                            or not np.array_equal(end[present], valid[present])
                            or not np.isnat(start[~present]).all() or not np.isnat(end[~present]).all()
                            or not (np.all(start[present] == end[present]) if instant
                                    else np.all(start[present] < end[present]))):
                        raise ValueError(f"GEFS native presence/window mismatch: {source}/{name}")
        if eager:
            validate_product(store, expected_members)
    if eager:
        with (xr.open_zarr(ens, consolidated=True, chunks=None) as full,
              xr.open_zarr(ctrl, consolidated=True, chunks=None) as control):
            for name in sorted(ALL_FIELDS):
                try:
                    xr.testing.assert_equal(full[name].sel(member=[0]).load(), control[name].load())
                except AssertionError as exc:
                    raise ValueError(f"GEFS control differs from c00: {name}") from exc
    return dict(date=date, manifest_sha256=fingerprint, steps=len(STEPS), members=member_count,
                control_members=[0], raw_size_and_record_checks=True, chunks_present=True,
                raw_hashes_checked=raw_hashes is not None, full_products_read=eager)


def eligible_paths(hot_root, cold_root, zarr_root, date, member_count, record, cutoff, guard):
    parse_day(date)
    parse_day(cutoff)
    if (date >= cutoff or record.get("status") != "complete"
            or record.get("member_count") != member_count or not record.get("completed_at")):
        raise ValueError(f"GEFS date not eligible for cold storage: {date}")
    guard()
    for root in (hot_root, cold_root, zarr_root):
        require_directory(root)
    if hot_root == cold_root or zarr_root.is_relative_to(cold_root):
        raise ValueError("GEFS cold destination must be separate from hot data and products")
    hot_parent = hot_root / SOURCE
    cold_parent = cold_root / SOURCE
    require_directory(hot_parent)
    if hot_parent.stat().st_dev != hot_root.stat().st_dev:
        raise ValueError("GEFS hot source directory is on another device")
    if cold_parent.exists():
        require_directory(cold_parent)
        if cold_parent.stat().st_dev != cold_root.stat().st_dev:
            raise ValueError("GEFS cold archive directory is on another device")
    hot = hot_parent / date
    retired = hot_parent / f".gefs-retired-{date}"
    cold = cold_parent / date
    if hot.is_symlink() or retired.is_symlink() or cold.is_symlink():
        raise ValueError("symlink in GEFS archive paths")
    if hot.exists() and retired.exists():
        raise ValueError("both active and retired GEFS dates exist")
    return hot, cold, retired


def check_archive_day(*, hot_root, cold_root, zarr_root, date, member_count, record, cutoff, guard):
    """Read-only pre-copy gate, executed inside the container under the shared host lock."""
    hot, cold, _ = eligible_paths(hot_root, cold_root, zarr_root, date, member_count, record, cutoff, guard)
    # A verified final cold date can exist after a previous time window ended
    # before retirement. Skip another copy and recheck both trees at retirement.
    day = cold if cold.exists() else hot
    hashes = tree_hashes(day)
    verified = verify_day(day, zarr_root, date, member_count, raw_hashes=hashes, eager=True)
    return dict(date=date, status="ready", raw_storage="hot" if day == hot else "cold",
                tree_sha256=tree_fingerprint(hashes), files=len(hashes),
                manifest_sha256=verified["manifest_sha256"], file_hashes=hashes)


def recheck_archive_day(*, hot_root, cold_root, zarr_root, date, member_count, record, cutoff, guard, manifest_sha256):
    """Revalidate slicing/provenance before a later copy window.

    All copied bytes must still match the previously verified raw tree.
    Retirement independently rehashes both trees before removing the hot one.
    """
    hot, cold, _ = eligible_paths(hot_root, cold_root, zarr_root, date, member_count, record, cutoff, guard)
    day = cold if cold.exists() else hot
    verified = verify_day(day, zarr_root, date, member_count, eager=True)
    if verified["manifest_sha256"] != manifest_sha256:
        raise ValueError("GEFS manifest changed after preparation")
    return dict(date=date, status="ready", raw_storage="hot" if day == hot else "cold",
                manifest_sha256=manifest_sha256)


def retire_day(*, hot_root, cold_root, zarr_root, date, member_count, record, cutoff, guard, tree_sha256):
    """Validate the published cold copy on the read-only bind, then retire only hot raw data."""
    hot, cold, retired = eligible_paths(hot_root, cold_root, zarr_root, date, member_count, record, cutoff, guard)
    hashes = tree_hashes(cold)
    if tree_fingerprint(hashes) != tree_sha256:
        raise ValueError("GEFS cold copy checksum or file list mismatch")
    # Revalidate the actual archive and products before any removal, including
    # when a previous run stopped between publication and retirement.
    verify_day(cold, zarr_root, date, member_count, raw_hashes=hashes, eager=True)
    if hot.exists():
        if tree_hashes(hot) != hashes:
            raise ValueError("GEFS hot original changed during copy")
        guard()
        hot.rename(retired)
        fsync_path(hot.parent)
    elif not retired.exists():
        return dict(date=date, status="already_archived")
    else:
        remaining = tree_hashes(retired)
        if any(hashes.get(name) != digest for name, digest in remaining.items()):
            raise ValueError("retired GEFS files differ from the cold archive")
    guard()
    shutil.rmtree(retired)
    fsync_path(hot.parent)
    return dict(date=date, status="archived", files=len(hashes))


def audit_range(*, hot_root, cold_root, zarr_root, start, end, member_count):
    """Read-only daily audit; never count a partial cache or an unverified copy."""
    first, last = parse_day(start), parse_day(end)
    if first > last:
        raise ValueError("GEFS history start exceeds end")
    days = [(first + timedelta(days=i)).strftime("%Y%m%d") for i in range((last-first).days+1)]
    complete, missing, problems, evidence, current = [], [], [], [], []
    for date in days:
        hot = hot_root / SOURCE / date
        cold = cold_root / SOURCE / date
        if not all((p / ".zmetadata").is_file() for p in product_paths(zarr_root, date)):
            missing.append(date)
        else:
            errors = []
            # Do not snapshot both paths in advance: the hot date can be
            # retired while auditing, but its cold copy was published first.
            for storage, directory in (("hot", hot), ("cold", cold)):
                if not manifest_path(directory, date).is_file():
                    continue
                try:
                    result = verify_day(directory, zarr_root, date, member_count)
                except Exception as exc:
                    errors.append(f"{storage}: {type(exc).__name__}: {exc}")
                else:
                    complete.append(date)
                    evidence.append(dict(**result, raw_storage=storage))
                    break
            else:
                if errors:
                    problems.append(dict(date=date, error="; ".join(errors)))
                else:
                    missing.append(date)
        if date not in complete and (hot / "00z").is_dir():
            init = hot / "00z"
            try:
                initial_mtime = init.stat().st_mtime
            except FileNotFoundError:
                continue
            cached = list((init / ".members").glob("*/*.grib2"))
            # A running download may replace cache/status files. Sample each
            # existing file once and ignore those that disappear during the read.
            sizes, mtimes = [], []
            for file in cached:
                try:
                    info = file.stat()
                except FileNotFoundError:
                    continue
                sizes.append(info.st_size)
                mtimes.append(info.st_mtime)
            try:
                status = json.loads((init / ".download-status.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                status = {}
            failures = status.pop("failures", [])
            current.append(dict(date=date, cached_files=len(sizes), expected=member_count*len(STEPS),
                                bytes=sum(sizes), status=status, failed_tasks_this_pass=len(failures),
                                failure_examples=failures[:2],
                                last_write_utc=datetime.fromtimestamp(max(mtimes, default=initial_mtime), timezone.utc).isoformat()))
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), start=start, end=end, total=len(days),
                complete=complete, missing=missing, problems=problems, evidence=evidence,
                current=list(reversed(current)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("audit", "check", "recheck", "retire"):
        command = sub.add_parser(mode)
        for root in ("hot", "cold", "zarr"):
            command.add_argument(f"--{root}-root", required=True, type=Path)
        command.add_argument("--member-count", required=True, type=int, choices=range(1, 32))
        if mode == "audit":
            command.add_argument("--start", required=True)
            command.add_argument("--end", required=True)
        else:
            command.add_argument("--date", required=True)
            command.add_argument("--cutoff", required=True)
            command.add_argument("--record", required=True, type=json.loads)
            command.add_argument("--mounts", required=True, type=json.loads)
            if mode == "retire":
                command.add_argument("--tree-sha256", required=True)
            elif mode == "recheck":
                command.add_argument("--manifest-sha256", required=True)
    args = vars(parser.parse_args(argv))
    mode = args.pop("mode")
    try:
        if mode == "audit":
            result = audit_range(**args)
        else:
            mounts = args.pop("mounts")
            operation = {"check": check_archive_day, "recheck": recheck_archive_day, "retire": retire_day}[mode]
            result = operation(**args, guard=lambda: check_mounts(
                args["hot_root"], args["cold_root"], args["zarr_root"], mounts))
    except Exception as exc:
        print(json.dumps(dict(status="blocked", error=f"{type(exc).__name__}: {exc}")), file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
