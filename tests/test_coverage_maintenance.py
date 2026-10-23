import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace

import pytest
import xarray as xr

from climate_restore import coverage_index as index


def module(name):
    path = Path(__file__).parents[1] / 'deploy' / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


@pytest.fixture
def helper():
    return module('coverage_index_maintain.py')


def product(root, date='20261023', names=('t2m',), source='demo', part=''):
    path = root / source / f'{date}_00z_{source}{part}.zarr'
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset({n: ('step', [25.]) for n in names})
    for name in names:
        ds[name].attrs.update(units='%', probability_window_hours=24)
    ds.to_zarr(path, mode='w', consolidated=True, zarr_format=2)
    return path


def apply(helper, root, **kw):
    return helper.reconcile(root, 'demo', apply=True, min_age_seconds=0, **kw)


def snapshot(root):
    with sqlite3.connect(root / '.catalog/demo.sqlite3') as db:
        state = db.execute('SELECT ready,total,first_init,last_init FROM state').fetchone()
        fields = db.execute('SELECT name,count FROM coverage ORDER BY name').fetchall()
        return state, fields


def test_missing_unready_and_corrupt_indexes_bootstrap(helper, tmp_path):
    store = product(tmp_path)
    assert helper.discover(tmp_path) == ['demo']
    assert apply(helper, tmp_path)['action'] == 'rebuilt'
    assert snapshot(tmp_path)[0][:2] == (1, 1)
    with sqlite3.connect(tmp_path / '.catalog/demo.sqlite3') as db:
        db.execute('UPDATE state SET ready=0')
    assert apply(helper, tmp_path)['action'] == 'rebuilt'
    (tmp_path / '.catalog/demo.sqlite3').write_bytes(b'not sqlite')
    assert apply(helper, tmp_path)['action'] == 'rebuilt'
    assert snapshot(tmp_path)[1] == [('t2m', 1)]
    assert (store / '.zmetadata').exists()


def test_unchanged_pass_reads_no_zarr_definitions_or_rewrites(helper, tmp_path, monkeypatch):
    product(tmp_path)
    apply(helper, tmp_path)
    db = tmp_path / '.catalog/demo.sqlite3'
    before = (db.stat().st_mtime_ns, db.read_bytes())
    monkeypatch.setattr(index, '_definitions', lambda _: pytest.fail('unchanged metadata parsed'))
    assert apply(helper, tmp_path)['action'] == 'unchanged'
    assert (db.stat().st_mtime_ns, db.read_bytes()) == before


def test_existing_producer_keeps_small_fingerprint_index_updated(helper, tmp_path):
    product(tmp_path)
    apply(helper, tmp_path)
    store = product(tmp_path, '20250101')
    index.update_store(store)
    assert apply(helper, tmp_path)['action'] == 'unchanged'
    with sqlite3.connect(tmp_path / '.catalog/demo.sqlite3') as db:
        plan = db.execute(f'EXPLAIN QUERY PLAN SELECT init,signature FROM products INDEXED BY {helper.FINGERPRINT_INDEX}').fetchall()
    assert 'COVERING INDEX' in str(plan)


def test_external_backfill_rewrite_and_deletion(helper, tmp_path, monkeypatch):
    newest = product(tmp_path)
    apply(helper, tmp_path)
    old = product(tmp_path, '20250101', ('t2m', 'u10'))
    assert apply(helper, tmp_path)['added'] == 1
    assert snapshot(tmp_path)[0][2].startswith('2025')
    product(tmp_path, names=('v10',))
    original = index._definitions
    reads = []
    def record(parts):
        reads.extend(parts)
        return original(parts)
    monkeypatch.setattr(index, '_definitions', record)
    assert apply(helper, tmp_path)['changed'] == 1
    assert reads == [newest]
    shutil.rmtree(old)
    assert apply(helper, tmp_path)['removed'] == 1
    assert snapshot(tmp_path)[1] == [('v10', 1)]
    shutil.rmtree(newest)
    assert 'demo' in helper.discover(tmp_path)  # indexed empty source stays discoverable
    assert apply(helper, tmp_path)['removed'] == 1
    assert snapshot(tmp_path)[0][:2] == (1, 0)


def test_bad_metadata_and_concurrent_change_keep_last_good_index(helper, tmp_path, monkeypatch):
    product(tmp_path)
    apply(helper, tmp_path)
    path = tmp_path / '.catalog/demo.sqlite3'
    before = path.read_bytes()
    old = product(tmp_path, '20250101')
    (old / '.zmetadata').write_text('bad metadata')
    with pytest.raises(ValueError):
        apply(helper, tmp_path)
    assert path.read_bytes() == before
    product(tmp_path, '20250101')
    original = index._definitions
    def racing(parts):
        definitions = original(parts)
        product(tmp_path, '20261024')
        return definitions
    monkeypatch.setattr(index, '_definitions', racing)
    with pytest.raises(RuntimeError, match='inventory changed'):
        apply(helper, tmp_path)
    assert path.read_bytes() == before
    assert not list(path.parent.glob('.*maintenance-*.sqlite3*'))


def test_recent_incomplete_and_unavailable_sources_are_not_assumed_empty(helper, tmp_path):
    store = product(tmp_path)
    result = helper.reconcile(tmp_path, 'demo', apply=True)
    assert result['action'] == 'deferred'
    assert not (tmp_path / '.catalog/demo.sqlite3').exists()
    apply(helper, tmp_path)
    before = (tmp_path / '.catalog/demo.sqlite3').read_bytes()
    shutil.rmtree(store.parent)
    with pytest.raises(OSError, match='directory unavailable'):
        apply(helper, tmp_path)
    assert (tmp_path / '.catalog/demo.sqlite3').read_bytes() == before
    store = product(tmp_path)
    (store / '.zmetadata').unlink()
    (tmp_path / '.catalog/demo.sqlite3').unlink()
    assert apply(helper, tmp_path)['action'] == 'deferred'
    assert not (tmp_path / '.catalog/demo.sqlite3').exists()


def test_index_semantic_damage_is_rebuilt(helper, tmp_path):
    product(tmp_path)
    apply(helper, tmp_path)
    with sqlite3.connect(tmp_path / '.catalog/demo.sqlite3') as db:
        db.execute('UPDATE coverage SET count=2')
    assert apply(helper, tmp_path)['action'] == 'rebuilt'
    assert snapshot(tmp_path)[1] == [('t2m', 1)]


def test_incremental_transaction_rolls_back_all_changes(helper, tmp_path, monkeypatch):
    product(tmp_path)
    apply(helper, tmp_path)
    product(tmp_path, '20250101')
    product(tmp_path, '20250102')
    previous = snapshot(tmp_path)
    original = index._replace
    calls = []
    def fail_second(*args, **kwargs):
        calls.append(args[1])
        if len(calls) == 2:
            raise sqlite3.OperationalError('simulated write failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(index, '_replace', fail_second)
    with pytest.raises(sqlite3.OperationalError, match='simulated write'):
        apply(helper, tmp_path)
    assert snapshot(tmp_path) == previous


def test_dry_run_never_creates_catalog_and_ignores_noncanonical_staging(helper, tmp_path):
    product(tmp_path)
    (tmp_path / 'demo/.20261024_00z_demo.zarr').mkdir()
    (tmp_path / 'demo/20269999_00z_demo.zarr').mkdir()
    result = helper.reconcile(tmp_path, 'demo', min_age_seconds=0)
    assert result['action'] == 'rebuilt' and not result['applied']
    assert not (tmp_path / '.catalog').exists()
    assert apply(helper, tmp_path)['total'] == 1


def test_legacy_parts_and_probability_attributes(helper, tmp_path):
    surface = product(tmp_path, part='-sfc')
    pressure = product(tmp_path, names=('prob',), part='-pres')
    (tmp_path / 'demo-sfc').mkdir()
    surface.rename(tmp_path / 'demo-sfc' / surface.name)
    (tmp_path / 'demo-pres').mkdir()
    pressure.rename(tmp_path / 'demo-pres' / pressure.name)
    assert helper.discover(tmp_path) == ['demo']
    assert apply(helper, tmp_path)['total'] == 1
    with sqlite3.connect(tmp_path / '.catalog/demo.sqlite3') as db:
        definitions = json.loads(db.execute('SELECT definitions FROM products').fetchone()[0])
    assert {d['name'] for d in definitions} == {'t2m', 'prob'}
    assert all(d['attrs']['units'] == '%' and d['attrs']['probability_window_hours'] == 24 for d in definitions)


def test_host_respects_production_lock_and_single_flight(tmp_path, monkeypatch):
    import fcntl
    host = module('coverage_index_maintain_host.py')
    lock = tmp_path / 'lock/restore_scan.lock'
    lock.parent.mkdir()
    with lock.open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        monkeypatch.setattr(host.subprocess, 'run', lambda *a, **k: pytest.fail('started while producer busy'))
        assert host.run(SimpleNamespace(operation_dir=tmp_path, extra_lock=[]))['status'] == 'deferred'


def test_host_command_and_failed_worker_record(tmp_path, monkeypatch):
    host = module('coverage_index_maintain_host.py')
    commands = []
    def run(command, **kw):
        commands.append(command)
        if command[1] == 'inspect':
            return SimpleNamespace(stdout='true\n')
        return SimpleNamespace(stdout=json.dumps({'status': 'error', 'sources': []}), stderr='failure', returncode=1)
    monkeypatch.setattr(host.subprocess, 'run', run)
    args = SimpleNamespace(operation_dir=tmp_path, extra_lock=[], container='fixture',
        restore_dir='/repo', worker='/repo/deploy/coverage_index_maintain.py', root='/data', apply=True)
    assert host.run(args)['status'] == 'error'
    assert '--apply' in commands[-1]
    assert 'OPENBLAS_NUM_THREADS=1' in commands[-1]
    assert host.main(['--operation-dir', str(tmp_path), '--apply']) == 1
    assert json.loads((tmp_path / 'logs/coverage-index-maintenance-status.json').read_text())['status'] == 'error'
