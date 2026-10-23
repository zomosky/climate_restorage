"""Producer-owned forecast metadata index, filesystem protocol v1.

Only the explicit rebuild command scans history. Incremental publication reads the
(one to three) stores of one init. Readers require ready=1 and never write here.
SQLite DELETE journals allow consumers to mount the data tree read-only.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile

SCHEMA_VERSION = 1
_PATTERN = re.compile(r'^(\d{8})_(\d{2})z_(.+?)(?:-(sfc|pres))?\.zarr$')
_SOURCE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
_COORDS = {'latitude', 'longitude', 'step', 'time', 'valid_time', 'pressure_level',
           'isobaricInhPa', 'number', 'surface'}


def _identity(store: Path):
    match = _PATTERN.fullmatch(store.name)
    if not match or store.name.startswith('.'):
        return None
    date, cycle, source, _ = match.groups()
    init = datetime.strptime(date + cycle, '%Y%m%d%H').replace(tzinfo=timezone.utc).isoformat()
    return source, init, f'{date}_{cycle}z_{source}'


def _index_dir(root, source, index_dir=None):
    if not _SOURCE.fullmatch(source) or source in {'.', '..'}:
        raise ValueError('invalid coverage index source')
    target = Path(index_dir) if index_dir is not None else Path(root) / '.catalog'
    target.mkdir(parents=True, exist_ok=True)
    return target


@contextmanager
def _locked(directory, source):
    with (directory / f'{source}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _connect(path, source):
    db = sqlite3.connect(path, timeout=30)
    try:
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA synchronous=FULL')
        db.executescript('''
            CREATE TABLE IF NOT EXISTS state (
              version INTEGER NOT NULL, source TEXT NOT NULL, ready INTEGER NOT NULL,
              total INTEGER NOT NULL, first_init TEXT, last_init TEXT);
            CREATE TABLE IF NOT EXISTS products (init TEXT PRIMARY KEY, definitions TEXT NOT NULL, signature TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS product_variables (
              init TEXT REFERENCES products(init) ON DELETE CASCADE,
              name TEXT NOT NULL, PRIMARY KEY(init,name));
            CREATE INDEX IF NOT EXISTS membership ON product_variables(name,init);
            CREATE TABLE IF NOT EXISTS coverage (
              name TEXT PRIMARY KEY, first_init TEXT NOT NULL, last_init TEXT NOT NULL, count INTEGER NOT NULL);
        ''')
        row = db.execute('SELECT version, source FROM state').fetchone()
        if row is None:
            db.execute('INSERT INTO state VALUES (?,?,0,0,NULL,NULL)', (SCHEMA_VERSION, source))
            db.commit()
        elif row != (SCHEMA_VERSION, source):
            raise ValueError('incompatible coverage index; rebuild required')
        return db
    except BaseException:
        db.close()
        raise


def _parts(store, prefix):
    source = _identity(store)[0]
    root = store.parent.parent
    folders = {store.parent, *(root / f'{source}{suffix}' for suffix in ('', '-sfc', '-pres'))}
    return sorted({p for folder in folders for suffix in ('', '-sfc', '-pres')
                   if (p := folder / f'{prefix}{suffix}.zarr').is_dir()})


def _signature(parts):
    if not parts:
        return None
    result = []
    for part in parts:
        try:
            stat = (part / '.zmetadata').stat()
        except FileNotFoundError:
            return None
        result.append((part.name, stat.st_mtime_ns, stat.st_size))
    return json.dumps(sorted(result))


def _definitions(parts):
    """Read consolidated JSON and only the small pressure coordinate, not data arrays."""
    result = {}
    for part in parts:
        meta = json.loads((part / '.zmetadata').read_text())['metadata']
        levels = None
        coords = set(_COORDS)
        for key, attrs in meta.items():
            if key.endswith('.zattrs') and isinstance(attrs, dict):
                coords.update(str(attrs.get('coordinates', '')).split())
        for key, array in sorted(meta.items()):
            if not key.endswith('/.zarray'):
                continue
            base = key.split('/')[0]
            attrs = meta.get(f'{base}/.zattrs', {})
            dims = attrs.get('_ARRAY_DIMENSIONS', [])
            if base in coords or dims == [base] or attrs.get('zrread_probability_boundary'):
                continue
            if attrs.get('probability_sample_count') == 0:
                continue
            attrs = {k: v for k, v in attrs.items() if k != '_ARRAY_DIMENSIONS'}
            if 'pressure_level' in dims:
                if levels is None:
                    import zarr
                    levels = list(zarr.open_array(str(part / 'pressure_level'), mode='r')[:])
                for value in levels:
                    level = format(float(value), 'g')
                    name = f'{base}_{level}'
                    result[name] = dict(name=name, base=base, level=level, attrs=attrs)
            else:
                result[base] = dict(name=base, base=base, level=None, attrs=attrs)
    return [result[k] for k in sorted(result)]


def _replace(db, init, definitions, signature=""):
    old = {r[0] for r in db.execute('SELECT name FROM product_variables WHERE init=?', (init,))}
    names = {v[k] for v in definitions or [] for k in ('name', 'base')}
    db.execute('DELETE FROM products WHERE init=?', (init,))
    if definitions is not None:
        db.execute('INSERT INTO products VALUES (?,?,?)', (init, json.dumps(definitions, ensure_ascii=False), signature))
        db.executemany('INSERT INTO product_variables VALUES (?,?)', [(init, n) for n in sorted(names)])
    # Recalculate only changed fields from indexed membership, never Zarr history.
    for name in old | names:
        first, last, count = db.execute(
            'SELECT MIN(init),MAX(init),COUNT(*) FROM product_variables WHERE name=?', (name,)).fetchone()
        db.execute('DELETE FROM coverage WHERE name=?', (name,))
        if count:
            db.execute('INSERT INTO coverage VALUES (?,?,?,?)', (name, first, last, count))
    first, last, total = db.execute('SELECT MIN(init),MAX(init),COUNT(*) FROM products').fetchone()
    db.execute('UPDATE state SET total=?,first_init=?,last_init=?', (total, first, last))


def update_store(store: Path) -> bool:
    """Publish/repair one final init. A newly created index stays unready until rebuild."""
    store = Path(store)
    identity = _identity(store)
    if identity is None:
        return False  # hidden staging output and noncanonical test/scratch stores
    source, init, prefix = identity
    directory = _index_dir(store.parent.parent, source)
    with _locked(directory, source):
        parts = _parts(store, prefix)
        signature = _signature(parts)
        db = _connect(directory / f'{source}.sqlite3', source)
        try:
            previous = db.execute('SELECT signature FROM products WHERE init=?', (init,)).fetchone()
            if signature is not None and previous == (signature,):
                return False
            definitions = _definitions(parts) if signature is not None else None
            with db:
                _replace(db, init, definitions, signature)
        finally:
            db.close()
    return True


def remove_store(store: Path) -> bool:
    """Withdraw an init before destructive rewrite/removal (caller owns the data operation)."""
    identity = _identity(Path(store))
    if identity is None:
        return False
    source, init, _ = identity
    directory = _index_dir(Path(store).parent.parent, source)
    path = directory / f'{source}.sqlite3'
    with _locked(directory, source):
        if not path.exists():
            return False
        db = _connect(path, source)
        try:
            with db:
                _replace(db, init, None)
        finally:
            db.close()
    return True


def rebuild(root: Path, source: str, index_dir: Path | None = None) -> dict:
    """Explicit one-off history scan, atomic replacement; failed builds keep old DB."""
    root = Path(root)
    directory = _index_dir(root, source, index_dir)
    with _locked(directory, source):
        fd, name = tempfile.mkstemp(prefix=f'.{source}-', suffix='.sqlite3', dir=directory)
        os.close(fd)
        temporary = Path(name)
        db = None
        try:
            db = _connect(temporary, source)
            grouped = {}
            for suffix in ('', '-sfc', '-pres'):
                source_dir = root / f'{source}{suffix}'
                if not source_dir.is_dir():
                    continue
                for store in source_dir.iterdir():
                    identity = _identity(store)
                    if identity and identity[0] == source and store.is_dir():
                        grouped.setdefault(identity[1], []).append(store)
            with db:
                for init, parts in sorted(grouped.items()):
                    if all((p / '.zmetadata').is_file() for p in parts):
                        _replace(db, init, _definitions(parts), _signature(parts))
                db.execute('UPDATE state SET ready=1')
            total = db.execute('SELECT total FROM state').fetchone()[0]
            db.close()
            db = None
            os.chmod(temporary, 0o644)
            os.replace(temporary, directory / f'{source}.sqlite3')
            return {'source': source, 'total': total, 'index': str(directory / f'{source}.sqlite3')}
        finally:
            if db is not None:
                db.close()
            temporary.unlink(missing_ok=True)
            Path(str(temporary) + '-journal').unlink(missing_ok=True)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['rebuild', 'update', 'remove'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--source')
    parser.add_argument('--all', action='store_true', help='bootstrap every forecast source under --root')
    parser.add_argument('--store', type=Path)
    parser.add_argument('--index-dir', type=Path, help='bootstrap output override, e.g. for read-only audits')
    args = parser.parse_args(argv)
    if args.action == 'rebuild':
        if args.root is None or (bool(args.source) == args.all):
            parser.error('rebuild requires --root and either --source or --all')
        sources = [args.source] if args.source else sorted({
            identity[0] for folder in args.root.iterdir() if folder.is_dir() and not folder.name.startswith(('.', '_'))
            for path in folder.iterdir() if (identity := _identity(path))})
        for source in sources:
            print(json.dumps(rebuild(args.root, source, args.index_dir)), flush=True)
    else:
        if args.store is None or args.index_dir is not None:
            parser.error('update/remove require --store, and use its adjacent .catalog')
        print(json.dumps({'updated': (update_store if args.action == 'update' else remove_store)(args.store)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
