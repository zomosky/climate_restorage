#!/usr/bin/env python3
"""Host-side GEFS cold storage scheduler, using the existing GEFS pipeline lock."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import re

from gefs_archive_io import directory_identity as mount_identity, fsync_path, validate_file_hashes


def window_seconds(now):
    # Realtime :00/:30, historical backfill :10/:40. Leave one minute before
    # history starts, in addition to the worker's ten-second SIGKILL grace.
    offset = (now.minute % 30) * 60 + now.second + now.microsecond / 1e6
    return max(0, int(540 - offset)) if offset >= 60 else 0


def read_config(path):
    with Path(path).open() as handle:
        return json.load(handle)


def save_checkpoint(path, value):
    if value is None:
        if path.exists():
            path.unlink()
            fsync_path(path.parent)
        return
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        fsync_path(path.parent)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def verification_receipt(result, date, *, full):
    ready = json.loads(result.stdout.strip().splitlines()[-1])
    if (ready["status"] != "ready" or ready["date"] != date
            or ready["raw_storage"] not in ("hot", "cold")
            or not re.fullmatch("[0-9a-f]{64}", ready["manifest_sha256"])):
        raise ValueError("invalid GEFS verification receipt")
    if full:
        validate_file_hashes(ready["file_hashes"], ready["tree_sha256"])
    return ready


def candidate_dates(live, history, hot_root, cutoff):
    """Only successful scheduler receipts are eligible; age alone is insufficient."""
    if history["history_end"] >= live["start_date"]:
        raise ValueError("GEFS history and realtime overlap")
    if history["live_config"] != live.get("_config_path", history["live_config"]):
        raise ValueError("GEFS history uses a different realtime config")
    candidates = []
    groups = [(history, history["history_start"], history["history_end"]),
              (live, live["start_date"], cutoff)]
    for cfg, start, end in groups:
        state_path = Path(cfg["state_file"])
        if not state_path.is_file():
            continue
        members = cfg.get("member_count", live.get("member_count", 5))
        for date, record in read_config(state_path).get("dates", {}).items():
            if (len(date) != 8 or not date.isdigit() or not start <= date <= end or date >= cutoff
                    or record.get("status") != "complete" or record.get("member_count") != members
                    or not record.get("completed_at")):
                continue
            raw = hot_root / "gefs-ens"
            if (raw / date).is_dir() or (raw / f".gefs-retired-{date}").is_dir():
                candidates.append((date, members, record))
    return sorted(candidates, key=lambda item: item[0])


def tick(live, history, now, *, run=subprocess.run, monotonic=time.monotonic, dry_run=False):
    budget = window_seconds(now)
    if budget < 60:
        return {"status": "yield"}
    deadline = monotonic() + budget
    with Path(live["lock_file"]).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "busy"}
        zarr_host = Path(live["host_zarr_root"])
        hot_host = Path(live.get("host_data_root", zarr_host.parent.parent))
        cold_host = Path(live["host_cold_root"])
        mounts = {"hot": mount_identity(hot_host), "cold": mount_identity(cold_host),
                  "zarr": mount_identity(zarr_host)}
        if mounts["hot"][0] == mounts["cold"][0] or not zarr_host.is_relative_to(hot_host):
            raise ValueError("GEFS archive requires a separate cold device and hot Zarr")
        cutoff = (now.astimezone(timezone.utc).date() - timedelta(days=30)).strftime("%Y%m%d")
        candidates = candidate_dates(live, history, hot_host, cutoff)
        checkpoint_path = Path(live.get("coldmove_checkpoint", Path(live["state_file"]).parent / "gefs-coldmove-checkpoint.json"))
        checkpoint = read_config(checkpoint_path) if checkpoint_path.is_file() else None
        if checkpoint is not None:
            match = next((item for item in candidates if item[0] == checkpoint["date"]
                         and item[1] == checkpoint["member_count"]
                         and item[2]["completed_at"] == checkpoint["completed_at"]), None)
            if match is not None:
                if checkpoint["phase"] not in ("prepared", "copied"):
                    raise ValueError("invalid GEFS checkpoint phase")
                validate_file_hashes(checkpoint["file_hashes"], checkpoint["tree_sha256"])
                candidates = [match, *(item for item in candidates if item != match)]
            else:
                checkpoint = None
                if not dry_run:
                    save_checkpoint(checkpoint_path, None)
        if dry_run:
            return {"status": "dry_run", "cutoff": cutoff, "candidates": [item[0] for item in candidates],
                    "checkpoint_phase": checkpoint["phase"] if checkpoint else None}
        if not candidates:
            return {"status": "idle", "cutoff": cutoff}
        release = live["release_root"]
        restorage = f"{release}/climate_restorage"
        python = f"{restorage}/.venv/bin/python"
        errors = []
        container = live.get("container", "zhangmy-dev")
        for date, members, record in candidates:
            remaining = int(deadline - monotonic())
            if remaining < 60:
                break
            common = ["--hot-root", live["data_root"], "--cold-root", live["cold_data_root"],
                      "--zarr-root", live["zarr_root"], "--date", date, "--member-count", str(members),
                      "--cutoff", cutoff, "--record", json.dumps(record), "--mounts", json.dumps(mounts)]
            def worker(phase, seconds, extra=()):
                return ["docker", "exec", "-w", restorage, container,
                        "timeout", "--signal=TERM", "--kill-after=10s", f"{seconds}s",
                        python, f"{release}/download/deploy/gefs_run.py", "--bounded-exec", "--",
                        python, f"{restorage}/deploy/gefs_archive.py", phase, *common, *extra]
            if checkpoint is None:
                print(json.dumps({"status": "checking", "date": date, "budget_seconds": remaining}), flush=True)
                result = run(worker("check", remaining), check=False, capture_output=True, text=True)
                if result.returncode:
                    if result.stderr:
                        print(result.stderr, file=sys.stderr, flush=True)
                    errors.append({"date": date, "phase": "check", "exit_code": result.returncode})
                    if result.returncode in (124, 137):
                        break
                    continue
                try:
                    ready = verification_receipt(result, date, full=True)
                except (ValueError, IndexError, KeyError, TypeError) as exc:
                    errors.append({"date": date, "phase": "check", "error": str(exc)})
                    continue
                checkpoint = {**ready, "member_count": members, "completed_at": record["completed_at"],
                              "phase": "prepared" if ready["raw_storage"] == "hot" else "copied"}
                save_checkpoint(checkpoint_path, checkpoint)
                # A successful full read is durable even if it takes the
                # entire idle window. Do not start another expensive phase.
                return {"status": checkpoint["phase"], "date": date, "blocked": errors}

            digest = checkpoint["tree_sha256"]
            if checkpoint["phase"] == "prepared":
                print(json.dumps({"status": "copy_window", "date": date, "budget_seconds": remaining}), flush=True)
                result = run(worker("recheck", remaining, ["--manifest-sha256", checkpoint["manifest_sha256"]]),
                             check=False, capture_output=True, text=True)
                if result.returncode:
                    if result.stderr:
                        print(result.stderr, file=sys.stderr, flush=True)
                    if result.returncode not in (124, 137):
                        save_checkpoint(checkpoint_path, None)
                    errors.append({"date": date, "phase": "recheck", "exit_code": result.returncode})
                    break
                ready = verification_receipt(result, date, full=False)
                if ready["manifest_sha256"] != checkpoint["manifest_sha256"]:
                    save_checkpoint(checkpoint_path, None)
                    raise ValueError("GEFS prepared manifest changed")
                if ready["raw_storage"] == "hot":
                    remaining = int(deadline - monotonic())
                    if remaining < 40:
                        break
                    copy = ["timeout", "--signal=TERM", "--kill-after=5s", f"{remaining}s",
                            sys.executable, str(Path(__file__).with_name("gefs_archive_io.py")), "copy",
                            "--hot-root", str(hot_host), "--cold-root", str(cold_host),
                            "--zarr-root", str(zarr_host), "--date", date, "--tree-sha256", digest,
                            "--mounts", json.dumps(mounts), "--container", container,
                            "--container-data-root", live["data_root"], "--tar-timeout", str(remaining-10),
                            "--checkpoint", str(checkpoint_path)]
                    result = run(copy, check=False)
                    if result.returncode:
                        errors.append({"date": date, "phase": "copy", "exit_code": result.returncode})
                        break
                checkpoint["phase"] = "copied"
                save_checkpoint(checkpoint_path, checkpoint)
                return {"status": "copied", "date": date, "blocked": errors}

            print(json.dumps({"status": "retire_window", "date": date, "budget_seconds": remaining}), flush=True)
            result = run(worker("retire", remaining, ["--tree-sha256", digest]), check=False)
            if result.returncode == 0:
                save_checkpoint(checkpoint_path, None)
                return {"status": "archived", "date": date, "candidates": len(candidates), "blocked": errors}
            errors.append({"date": date, "phase": "retire", "exit_code": result.returncode})
            break
        return {"status": "deferred", "blocked": errors, "candidates": len(candidates)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-config", type=Path, required=True)
    parser.add_argument("--history-config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        live = read_config(args.live_config)
        history = read_config(args.history_config)
        live["_config_path"] = str(args.live_config)
        result = tick(live, history, datetime.now(timezone.utc), dry_run=args.dry_run)
    except Exception as exc:
        print(json.dumps({"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result), flush=True)
    return 1 if result["status"] == "deferred" and result["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
