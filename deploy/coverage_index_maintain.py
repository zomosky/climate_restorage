#!/usr/bin/env python3
"""Reconcile forecast indexes off the HTTP path; data stores are always read-only.

Run while holding the production restore scheduler locks (see the host wrapper).
Only directory entries and .zmetadata stats are inspected for unchanged products.
Changed products commit in one SQLite transaction; rebuilt sources are published
by atomic replacement. Failures retain the old index. This is metadata maintenance,
not Zarr chunk-integrity validation.
"""
from __future__ import annotations

import argparse
from contextlib import closing, nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import sqlite3
import tempfile
import time

from climate_restore import coverage_index as index

FINGERPRINT_INDEX = 'maintenance_product_signatures'


def identity(path):
    try:
        return index._identity(path)
    except ValueError:
        return None


def discover(root):
    """Include indexed empty sources, but never traverse arbitrary nested trees."""
    sources = set()
    for folder in root.iterdir():
        if folder.name.startswith(('.', '_')) or folder.is_symlink() or not folder.is_dir():
            continue
        for store in folder.iterdir():
            item = identity(store)
            if item and folder.name in {item[0], item[0] + '-sfc', item[0] + '-pres'} and store.is_dir() and not store.is_symlink():
                sources.add(item[0])
    directory = root / '.catalog'
    if directory.is_dir():
        sources.update(p.stem for p in directory.glob('*.sqlite3')
                       if index._SOURCE.fullmatch(p.stem) and p.stem not in {'.', '..'})
    return sorted(sources)


def inventory(root, source):
    grouped = {}
    folders = [root / (source + suffix) for suffix in ('', '-sfc', '-pres')]
    if not any(p.is_dir() and not p.is_symlink() for p in folders):
        raise OSError('source directory unavailable; retaining index instead of assuming all data deleted')
    for folder in folders:
        if not folder.is_dir() or folder.is_symlink():
            continue
        for store in folder.iterdir():
            item = identity(store)
            if item and item[0] == source and store.is_dir() and not store.is_symlink():
                grouped.setdefault(item[1], []).append(store)
    return {init: (sorted(parts), index._signature(sorted(parts)))
            for init, parts in sorted(grouped.items())}


def recent(groups, min_age_seconds):
    cutoff = time.time_ns() - min_age_seconds * 1_000_000_000
    for parts, signature in groups.values():
        if signature is not None:
            if any(row[1] > cutoff for row in json.loads(signature)):
                return True
        elif any(p.stat().st_mtime_ns > cutoff for p in parts):
            return True
    return False


def load_index(path, source):
    """Reject unusable and inconsistent indexes without creating/recovering one."""
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=2)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        if db.execute('PRAGMA journal_mode').fetchone() != ('delete',):
            raise ValueError('unsupported journal mode')
        state = db.execute('SELECT version,source,ready,total,first_init,last_init FROM state').fetchall()
        if len(state) != 1 or state[0][:3] != (index.SCHEMA_VERSION, source, 1):
            raise ValueError('index missing initialization or incompatible schema')
        # A table scan would traverse the large definitions column before reaching
        # signature on every row. Keep routine checks in this small covering index.
        products = dict(db.execute(f'SELECT init,signature FROM products INDEXED BY {FINGERPRINT_INDEX}'))
        first, last, total = min(products, default=None), max(products, default=None), len(products)
        if state[0][3:] != (total, first, last):
            raise ValueError('index summary inconsistent')
        coverage = db.execute('SELECT name,first_init,last_init,count FROM coverage ORDER BY name').fetchall()
        if any(not isinstance(count, int) or not 1 <= count <= total or start > end
               or start < first or end > last for _, start, end, count in coverage):
            raise ValueError('variable coverage inconsistent')
        latest = db.execute('SELECT definitions FROM products ORDER BY init DESC LIMIT 1').fetchone()
        if latest:
            definitions = json.loads(latest[0])
            names = {row[0] for row in coverage}
            if not isinstance(definitions, list) or any(
                not isinstance(v, dict) or not {'name', 'base', 'level', 'attrs'} <= v.keys()
                or not isinstance(v['attrs'], dict) or v['name'] not in names for v in definitions
            ):
                raise ValueError('variable definitions inconsistent')
        return products


def reconcile(root, source, *, apply=False, min_age_seconds=120, force_rebuild=False):
    root = Path(root)
    if not root.is_dir():
        raise OSError('Zarr root unavailable')
    if not index._SOURCE.fullmatch(source) or source in {'.', '..'}:
        raise ValueError('invalid source')
    directory = root / '.catalog'
    if apply:
        directory.mkdir(exist_ok=True)
    with index._locked(directory, source) if apply else nullcontext():
        groups = inventory(root, source)
        if recent(groups, min_age_seconds):
            return dict(source=source, action='deferred', reason='recent metadata; retry next tick')
        current = {init: signature for init, (_, signature) in groups.items() if signature is not None}
        path = directory / f'{source}.sqlite3'
        try:
            previous = load_index(path, source)
            reason = 'forced metadata rebuild' if force_rebuild else None
        except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
            previous, reason = {}, f'{type(exc).__name__}: {exc}'
        rebuilding = reason is not None
        if rebuilding and not current and groups:
            return dict(source=source, action='deferred', reason='no complete product yet; retaining index')
        changed = sorted(init for init, signature in current.items()
                         if rebuilding or previous.get(init) != signature)
        removed = sorted(set(previous) - current.keys())
        if not rebuilding and len(changed) > 256:
            rebuilding, reason, changed = True, 'bulk metadata changes', sorted(current)
        report = dict(source=source, action='rebuilt' if rebuilding else 'repaired' if changed or removed else 'unchanged',
                      added=len(current.keys() - previous.keys()), changed=len(changed),
                      removed=len(removed), total=len(current), reason=reason, applied=False)
        if report['action'] == 'unchanged' or not apply:
            return report
        # Parse and check all changed products before opening a write transaction.
        if not rebuilding:
            definitions = {}
            for init in changed:
                parts, signature = groups[init]
                definitions[init] = index._definitions(parts)
                if index._signature(parts) != signature:
                    raise RuntimeError('metadata changed during maintenance; retaining previous index')
            if inventory(root, source) != groups:
                raise RuntimeError('product inventory changed during maintenance; retaining previous index')
            db = index._connect(path, source)
            try:
                with db:
                    for init in removed:
                        index._replace(db, init, None)
                    for init in changed:
                        index._replace(db, init, definitions[init], groups[init][1])
                    if inventory(root, source) != groups:
                        raise RuntimeError('product inventory changed during commit; retaining previous index')
            finally:
                db.close()
            report['applied'] = True
            return report
        fd, name = tempfile.mkstemp(prefix=f'.{source}-maintenance-', suffix='.sqlite3', dir=directory)
        os.close(fd)
        temporary = Path(name)
        db = None
        try:
            db = index._connect(temporary, source)
            db.execute(f'CREATE INDEX {FINGERPRINT_INDEX} ON products(init,signature)')
            with db:
                for init in changed:
                    parts, signature = groups[init]
                    definitions = index._definitions(parts)
                    if index._signature(parts) != signature:
                        raise RuntimeError('metadata changed during rebuild; retaining previous index')
                    index._replace(db, init, definitions, signature)
                db.execute('UPDATE state SET ready=1')
            # Full physical checking is cheap on the just-built, hot temporary DB.
            # Never do this full-file read on the periodic path of an existing DB.
            if db.execute('PRAGMA quick_check(1)').fetchone() != ('ok',):
                raise ValueError('rebuilt SQLite quick_check failed')
            if inventory(root, source) != groups:
                raise RuntimeError('product inventory changed during maintenance; retaining previous index')
            db.close()
            db = None
            load_index(temporary, source)
            os.chmod(temporary, 0o644)
            os.replace(temporary, path)
            report['applied'] = True
            return report
        finally:
            if db is not None:
                db.close()
            temporary.unlink(missing_ok=True)
            Path(str(temporary) + '-journal').unlink(missing_ok=True)


class DeadlineExceeded(BaseException):
    pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', action='append')
    parser.add_argument('--apply', action='store_true', help='otherwise report only')
    parser.add_argument('--force-rebuild', action='store_true')
    parser.add_argument('--min-age-seconds', type=int, default=120)
    parser.add_argument('--timeout-seconds', type=int, default=120)
    parser.add_argument('--memory-mib', type=int, default=1024)
    args = parser.parse_args(argv)
    if args.min_age_seconds < 0 or args.timeout_seconds < 1 or args.memory_mib < 512:
        parser.error('invalid age, timeout or memory limit')
    # Applied only to this process; no producer/container/global settings change.
    if hasattr(os, 'sched_setaffinity'):
        os.nice(10)
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_mib * 1024**2,) * 2)
    def expired(*_):
        raise DeadlineExceeded('maintenance deadline exceeded')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.timeout_seconds)
    started = time.monotonic()
    rows = []
    try:
        import zarr
        zarr.config.set({'threading.max_workers': 1, 'async.concurrency': 1})
        if not args.root.is_dir():
            raise OSError('Zarr root unavailable')
        for source in args.source or discover(args.root):
            try:
                rows.append(reconcile(args.root, source, apply=args.apply,
                    min_age_seconds=args.min_age_seconds, force_rebuild=args.force_rebuild))
            except Exception as exc:
                rows.append(dict(source=source, action='error', reason=f'{type(exc).__name__}: {exc}'))
    except (Exception, DeadlineExceeded) as exc:
        rows.append(dict(action='error', reason=f'{type(exc).__name__}: {exc}'))
    finally:
        signal.alarm(0)
    failed = any(row['action'] == 'error' for row in rows)
    print(json.dumps(dict(status='error' if failed else 'ok', apply=args.apply,
        at=datetime.now(timezone.utc).isoformat(), seconds=round(time.monotonic() - started, 3),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sources=rows)), flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
