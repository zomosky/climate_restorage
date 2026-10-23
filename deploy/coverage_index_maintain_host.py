#!/usr/bin/env python3
"""Host cron -> existing scheduler locks -> bounded in-container metadata worker."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess


def run(args):
    locks = [args.operation_dir / 'lock' / name for name in (
        'coverage_index_maintenance.lock', 'restore_scan.lock', 'restore_scan_aifs.lock',
        'restore_scan_ifs.lock', 'restore_scan_icon.lock', 'restore_scan_ifs_0p1.lock',
        'ens_probabilities.lock')]
    locks += [Path('/tmp/restore_cams.lock'), *args.extra_lock]
    with ExitStack() as stack:
        for path in sorted(set(locks)):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open('a'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return dict(status='deferred', reason='production job busy', lock=str(path))
        probe = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', args.container],
                               capture_output=True, text=True, timeout=15, check=True)
        if probe.stdout.strip() != 'true':
            return dict(status='deferred', reason='producer container not running')
        command = ['docker', 'exec', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=1',
                   '-w', args.restore_dir, args.container, args.restore_dir + '/.venv/bin/python',
                   '-u', args.worker, '--root', args.root, '--timeout-seconds', '120']
        if args.apply:
            command.append('--apply')
        worker = subprocess.run(command, capture_output=True, text=True, timeout=150)
        result = json.loads(worker.stdout)
        if worker.returncode or result.get('status') != 'ok':
            return dict(status='error', reason='worker failed', worker=result,
                        stderr=worker.stderr[-2000:], returncode=worker.returncode)
        return dict(status='ok', worker=result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--container', default='zhangmy-dev')
    parser.add_argument('--restore-dir', default='/workspace/climate_restorage')
    parser.add_argument('--worker', default='/workspace/climate_restorage/deploy/coverage_index_maintain.py')
    parser.add_argument('--root', default='/climate_data/climate_data_storage/zarr')
    parser.add_argument('--extra-lock', action='append', type=Path, default=[])
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        result = dict(status='error', reason=f'{type(exc).__name__}: {exc}')
    result['at'] = datetime.now(timezone.utc).isoformat()
    state = args.operation_dir / 'logs' / 'coverage-index-maintenance-status.json'
    state.parent.mkdir(parents=True, exist_ok=True)
    temporary = state.with_name(state.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, state)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return int(result['status'] == 'error')


if __name__ == '__main__':
    raise SystemExit(main())
