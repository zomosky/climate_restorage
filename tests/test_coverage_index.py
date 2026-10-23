import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from climate_restore.coverage_index import rebuild, update_store, remove_store


def product(root, date, names=('t2m',), part=''):
    path = root / 'demo' / f'{date}_00z_demo{part}.zarr'
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({n: ('step', [1.]) for n in names}).to_zarr(path, mode='w', consolidated=True, zarr_format=2)
    return path


def snapshot(root):
    with sqlite3.connect(root / '.catalog/demo.sqlite3') as db:
        state = db.execute('select ready,total,first_init,last_init from state').fetchone()
        coverage = {name: (first, last, count) for name, first, last, count in db.execute('select * from coverage')}
        latest = db.execute('select definitions from products order by init desc limit 1').fetchone()
    return state, coverage, json.loads(latest[0]) if latest else []


def test_append_backfill_rewrite_and_remove(tmp_path):
    root = tmp_path
    newest = product(root, '20261023', ('t2m', 'u10'))
    rebuild(root, 'demo')
    old = product(root, '20250101', ('t2m',))
    update_store(old)
    update_store(old)
    state, coverage, _ = snapshot(root)
    assert state[0:2] == (1, 2)
    assert coverage['t2m'] == ('2025-01-01T00:00:00+00:00', '2026-10-23T00:00:00+00:00', 2)
    assert coverage['u10'][2] == 1
    product(root, '20261023', ('v10',))
    update_store(newest)
    _, coverage, defs = snapshot(root)
    assert 'u10' not in coverage
    assert coverage['t2m'][1].startswith('2025')
    assert [v['name'] for v in defs] == ['v10']
    remove_store(old)
    assert snapshot(root)[0][1] == 1


def test_staging_incomplete_and_uninitialized(tmp_path):
    p = product(tmp_path, '20261023')
    update_store(p)
    assert snapshot(tmp_path)[0][0] == 0  # one update cannot certify legacy history
    (p / '.zmetadata').unlink()
    update_store(p)
    assert snapshot(tmp_path)[0][1] == 0
    assert update_store(tmp_path / 'demo/.20261024_00z_demo.staging') is False


def test_levels_multipart_probability_metadata_and_atomic_rebuild(tmp_path):
    p = product(tmp_path, '20261023', ('t2m',), '-sfc')
    pres = p.with_name('20261023_00z_demo-pres.zarr')
    ds = xr.Dataset({'t': ('pressure_level', [280., 270.]),
                     'prob': ('step', [20.]), 'boundary': ('step', [1.])},
                    coords={'pressure_level': [850, 500]})
    ds['prob'].attrs.update(units='%', probability_window_hours=24, probability_threshold=50)
    ds['boundary'].attrs['zrread_probability_boundary'] = 1
    ds.to_zarr(pres, consolidated=True, zarr_format=2)
    rebuild(tmp_path, 'demo')
    _, coverage, defs = snapshot(tmp_path)
    assert {'t', 't_850', 't_500', 't2m', 'prob'} == set(coverage)
    assert next(v for v in defs if v['name'] == 'prob')['attrs']['probability_threshold'] == 50
    # Bad metadata makes bootstrap fail without replacing the last good DB.
    (pres / '.zmetadata').write_text('broken')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        rebuild(tmp_path, 'demo')
    assert snapshot(tmp_path)[0][1] == 1


def test_concurrent_updates_are_serialized(tmp_path):
    rebuild(tmp_path, 'demo')
    paths = [product(tmp_path, f'202610{day:02d}') for day in range(1, 9)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(update_store, paths * 2))
    assert snapshot(tmp_path)[0][1] == 8
    assert snapshot(tmp_path)[1]['t2m'][2] == 8


def test_write_dataset_withdraws_failed_rewrite_and_can_repair_index(tmp_path, monkeypatch):
    from climate_restore import processor, coverage_index
    from climate_restore.config import ZarrOptions
    store = tmp_path / 'demo/20261023_00z_demo.zarr'
    ds = xr.Dataset({'t2m': ('step', [280.])})
    rebuild(tmp_path, 'demo')
    processor.write_dataset(ds, store, fmt='zarr', zarr_options=ZarrOptions())
    assert snapshot(tmp_path)[0][1] == 1
    monkeypatch.setattr(processor, '_verify_zarr_chunks', lambda _: ['missing chunk'])
    with pytest.raises(processor.ChunkIntegrityError):
        processor.write_dataset(ds, store, fmt='zarr', zarr_options=ZarrOptions())
    assert snapshot(tmp_path)[0][1] == 0
    assert not store.exists()
    monkeypatch.setattr(processor, '_verify_zarr_chunks', lambda _: [])
    publish = coverage_index.update_store
    def fail(_):
        raise OSError('index disk temporarily unavailable')
    monkeypatch.setattr(coverage_index, 'update_store', fail)
    with pytest.raises(OSError, match='index disk'):
        processor.write_dataset(ds, store, fmt='zarr', zarr_options=ZarrOptions())
    assert (store / '.zmetadata').exists()
    monkeypatch.setattr(coverage_index, 'update_store', publish)
    publish(store)  # scan skip / history resume retries publication without decoding GRIB
    assert snapshot(tmp_path)[0][1] == 1
    monkeypatch.setattr(coverage_index, '_definitions', lambda _: pytest.fail('unchanged retry parsed metadata'))
    assert publish(store) is False


def test_rebuild_and_update_cannot_lose_a_completed_new_init(tmp_path, monkeypatch):
    import threading
    from climate_restore import coverage_index
    old = product(tmp_path, '20261001')
    rebuild(tmp_path, 'demo')
    entered, release = threading.Event(), threading.Event()
    original = coverage_index._definitions
    def hold(parts):
        entered.set()
        assert release.wait(timeout=5)
        return original(parts)
    monkeypatch.setattr(coverage_index, '_definitions', hold)
    with ThreadPoolExecutor(max_workers=2) as pool:
        boot = pool.submit(rebuild, tmp_path, 'demo')
        assert entered.wait(timeout=5)
        latest = product(tmp_path, '20261023')
        update = pool.submit(update_store, latest)
        release.set()
        boot.result(timeout=5)
        update.result(timeout=5)
    assert snapshot(tmp_path)[0][1] == 2


def test_legacy_parts_in_separate_source_directories(tmp_path):
    sfc = product(tmp_path, '20261023', ('t2m',), '-sfc')
    pres = product(tmp_path, '20261023', ('u10',), '-pres')
    (tmp_path/'demo-sfc').mkdir()
    (tmp_path/'demo-pres').mkdir()
    sfc = sfc.rename(tmp_path/'demo-sfc'/sfc.name)
    pres.rename(tmp_path/'demo-pres'/pres.name)
    rebuild(tmp_path, 'demo')
    assert set(snapshot(tmp_path)[1]) == {'t2m','u10'}
    assert update_store(sfc) is False


def test_readers_only_observe_complete_commits(tmp_path):
    import threading
    from contextlib import closing
    rebuild(tmp_path, 'demo')
    paths = [product(tmp_path, f'202610{day:02d}') for day in range(1, 13)]
    finished = threading.Event()
    def read():
        reads = 0
        while not finished.is_set() or reads < 1:
            with closing(sqlite3.connect(tmp_path/'.catalog/demo.sqlite3')) as db:
                db.execute('BEGIN')
                total = db.execute('SELECT total FROM state').fetchone()[0]
                row = db.execute("SELECT count FROM coverage WHERE name='t2m'").fetchone()
                assert (row[0] if row else 0) == total
                reads += 1
        return reads
    with ThreadPoolExecutor(max_workers=2) as pool:
        reading = pool.submit(read)
        try:
            for path in paths:
                update_store(path)
        finally:
            finished.set()
        assert reading.result(timeout=5) > 0


def test_history_publishes_after_rename_and_resume_repairs_failed_index(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    from types import SimpleNamespace
    from climate_restore.processor import write_dataset
    path = Path(__file__).parents[1] / 'scripts/restore_probability_history.py'
    spec = importlib.util.spec_from_file_location('history_index_test', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    raw = tmp_path / 'raw.grib2'
    raw.write_bytes(b'GRIBfixture7777')
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text('{}')
    manifest = SimpleNamespace(source_name='demo', date='20261023', cycle=0,
        probability_availability=None, files=[SimpleNamespace(size_bytes=raw.stat().st_size,
        sha256=hashlib.sha256(raw.read_bytes()).hexdigest())])
    monkeypatch.setattr(helper, 'load_manifest', lambda _: manifest)
    monkeypatch.setattr(helper, 'verify', lambda *a: [raw])
    monkeypatch.setattr(helper, 'ProbabilityAdapter', lambda _: SimpleNamespace(configure_manifest=lambda _: None))
    monkeypatch.setattr(helper, 'validate_dataset', lambda *a: None)  # event validation has its own tests
    output = tmp_path / 'data'
    rebuild(output, 'demo')
    decoded = []
    def process(*a, out_path, **kw):
        decoded.append(out_path)
        ds = xr.Dataset({'tpg50': ('step',[25.])})
        ds['tpg50'].attrs['probability_sample_count'] = 1
        write_dataset(ds, out_path, fmt='zarr')
        assert snapshot(output)[0][1] == 0  # hidden staging store cannot publish
    monkeypatch.setattr(helper, 'process_init', process)
    def unavailable(store):
        assert (store / '.zmetadata').is_file()  # publish only after final rename
        raise OSError('index unavailable')
    monkeypatch.setattr(helper, 'update_store', unavailable)
    with pytest.raises(OSError, match='index unavailable'):
        helper.restore(manifest_path, output)
    monkeypatch.setattr(helper, 'update_store', update_store)
    receipt = helper.restore(manifest_path, output)
    assert receipt['resumed'] is True
    assert len(decoded) == 1
    assert snapshot(output)[0][1] == 1
