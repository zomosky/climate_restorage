#!/usr/bin/env python3
"""Standard-library GEFS archive filesystem operations.

The host writes cold storage. Raw files are streamed from the container since
the host cannot read root-owned raw data and the cold bind is read-only inside
the container. This module never removes hot files.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys

SOURCE = "gefs-ens"


def parse_day(date):
    parsed = datetime.strptime(date, "%Y%m%d")
    if parsed.strftime("%Y%m%d") != date:
        raise ValueError(f"invalid GEFS date: {date}")
    return parsed


def require_directory(path):
    if not path.is_dir() or path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError(f"missing or unsafe directory: {path}")


def directory_identity(path):
    require_directory(path)
    info = path.stat()
    return [info.st_dev, info.st_ino]


def check_mounts(hot_root, cold_root, zarr_root, expected):
    actual = {"hot": directory_identity(hot_root), "cold": directory_identity(cold_root),
              "zarr": directory_identity(zarr_root)}
    if actual != expected or actual["hot"][0] == actual["cold"][0]:
        raise ValueError("GEFS hot/cold device or bind mount changed")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_files(root):
    require_directory(root)
    device = root.stat().st_dev
    files = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if info.st_dev != device or stat.S_ISLNK(info.st_mode):
            raise ValueError(f"symlink or another device inside GEFS tree: {path}")
        if stat.S_ISREG(info.st_mode):
            files[path.relative_to(root).as_posix()] = path
        elif not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"unexpected file type inside GEFS tree: {path}")
    return files


def tree_hashes(root):
    return {name: sha256(path) for name, path in tree_files(root).items()}


def tree_fingerprint(hashes):
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_file_hashes(hashes, expected):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("empty GEFS prepared raw tree")
    for name, digest in hashes.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or path.as_posix() != name
                or name == "." or "\x00" in name or not re.fullmatch("[0-9a-f]{64}", digest)):
            raise ValueError("invalid GEFS prepared file record")
    if tree_fingerprint(hashes) != expected:
        raise ValueError("GEFS prepared tree checksum differs")


def revision(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def resume_stage(stage, expected, populate):
    """Retry only files missing from the hidden stage or failing their SHA."""
    stage.mkdir(exist_ok=True)
    files = tree_files(stage)
    if set(files) - set(expected):
        raise ValueError("unexpected files in GEFS cold staging directory")
    matched = {}
    for name, path in files.items():
        before = revision(path)
        digest = sha256(path)
        after = revision(path)
        if before == after and digest == expected[name]:
            matched[name] = after
    missing = sorted(set(expected) - set(matched))
    if missing:
        populate(stage, missing)
    files = tree_files(stage)
    if set(files) != set(expected):
        raise ValueError("GEFS cold copy file list mismatch")
    for name, path in files.items():
        if matched.get(name) == revision(path):
            continue
        if sha256(path) != expected[name]:
            raise ValueError(f"GEFS cold copy checksum mismatch: {name}")


def fsync_path(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_tree(root):
    for folder, _, files in os.walk(root, topdown=False):
        path = Path(folder)
        for name in files:
            fsync_path(path / name)
        fsync_path(path)


def copy_to_cold(hot_root, cold_root, date, expected, guard, *, populate, expected_files=None):
    parse_day(date)
    if expected_files is not None:
        validate_file_hashes(expected_files, expected)
    guard()
    hot = hot_root / SOURCE / date
    require_directory(hot)
    require_directory(cold_root)
    parent = cold_root / SOURCE
    parent.mkdir(exist_ok=True)
    require_directory(parent)
    if parent.stat().st_dev != cold_root.stat().st_dev:
        raise ValueError("GEFS cold archive directory is on another device")
    fsync_path(cold_root)
    cold = parent / date
    stage = parent / f".gefs-stage-{date}"
    if cold.is_symlink() or stage.is_symlink():
        raise ValueError("symlink in GEFS cold archive paths")
    if cold.exists():
        if tree_fingerprint(tree_hashes(cold)) != expected:
            raise ValueError("existing cold GEFS copy conflicts with the verified hot source")
        sync_tree(cold)
        guard()
        return
    # Interrupted staging is not a visible archive and may be retried while
    # the verified hot original is still present.
    if expected_files is not None:
        resume_stage(stage, expected_files, populate)
    else:
        if stage.exists():
            require_directory(stage)
            if tree_fingerprint(tree_hashes(stage)) != expected:
                shutil.rmtree(stage)
        if not stage.exists():
            stage.mkdir()
            populate(stage)
            if tree_fingerprint(tree_hashes(stage)) != expected:
                raise ValueError("GEFS cold copy checksum or file list mismatch")
    sync_tree(stage)
    guard()
    if cold.exists():
        raise ValueError("GEFS cold destination appeared during copy")
    stage.rename(cold)
    fsync_path(parent)


def stream_tar(container, container_day, stage, seconds, paths=None):
    inputs = ["."] if paths is None else ["--", *paths]
    command = ["docker", "exec", container, "timeout", "--signal=TERM", "--kill-after=5s",
               f"{seconds}s", "tar", "-C", str(container_day), "-cf", "-", *inputs]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as producer:
        try:
            copied = subprocess.run(["tar", "--no-same-owner", "--no-same-permissions",
                                     "-xf", "-", "-C", str(stage)], stdin=producer.stdout, check=False)
        finally:
            producer.stdout.close()
        if producer.wait() or copied.returncode:
            raise OSError("GEFS archive tar stream failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["copy"])
    for name in ("hot", "cold", "zarr"):
        parser.add_argument(f"--{name}-root", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--tree-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mounts", type=json.loads, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--container-data-root", type=Path, required=True)
    parser.add_argument("--tar-timeout", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        if args.tar_timeout <= 0 or not args.container_data_root.is_absolute():
            raise ValueError("invalid GEFS tar configuration")
        guard = lambda: check_mounts(args.hot_root, args.cold_root, args.zarr_root, args.mounts)
        expected_files = None
        if args.checkpoint is not None:
            checkpoint = json.loads(args.checkpoint.read_text())
            if (checkpoint.get("date") != args.date or checkpoint.get("phase") != "prepared"
                    or checkpoint.get("tree_sha256") != args.tree_sha256):
                raise ValueError("GEFS copy checkpoint differs")
            expected_files = checkpoint["file_hashes"]
        def populate(stage, paths=None):
            stream_tar(args.container, args.container_data_root / SOURCE / args.date,
                       stage, args.tar_timeout, paths)
        copy_to_cold(args.hot_root, args.cold_root, args.date, args.tree_sha256, guard,
                     populate=populate, expected_files=expected_files)
    except Exception as exc:
        print(json.dumps(dict(status="blocked", error=f"{type(exc).__name__}: {exc}")), file=sys.stderr, flush=True)
        return 1
    print(json.dumps(dict(status="copied", date=args.date)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
