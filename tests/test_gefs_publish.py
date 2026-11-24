import importlib.util
import hashlib
import json
from pathlib import Path
import pytest
from climate_restore.manifest import load_manifest
p=Path(__file__).parents[1]/'deploy/gefs_publish.py'
spec=importlib.util.spec_from_file_location('gefs_publish',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def manifest(tmp_path):
 f=tmp_path/'sample.grib2';f.write_bytes(b'GRIB7777');digest=hashlib.sha256(f.read_bytes()).hexdigest()
 data=dict(schema_version=1,source=dict(name='gefs-ens',members=list(range(5))),date='20261105',cycle=0,init_time='2026-11-05T00:00:00+00:00',completed_at='yes',failures=[],variables=[],files=[dict(step_hours=s,path=str(f),size_bytes=8,sha256=digest) for s in m.STEPS])
 p=tmp_path/'manifest.json';p.write_text(json.dumps(data));return p,data

@pytest.mark.parametrize('broken',['step','member','sha','failure'])
def test_full_manifest_gate(tmp_path,broken):
 p,data=manifest(tmp_path)
 if broken=='step':data['files'].pop()
 if broken=='member':data['source']['members']=[0]
 if broken=='sha':data['files'][0]['sha256']='bad'
 if broken=='failure':data['failures']=['incomplete']
 p.write_text(json.dumps(data))
 with pytest.raises(Exception):m.check_manifest(load_manifest(p),5)

def test_failed_validation_never_publishes(tmp_path,monkeypatch):
 p,data=manifest(tmp_path);out=tmp_path/'zarr'
 monkeypatch.setattr(m.GefsEnsAdapter,'configure_manifest',lambda *a:None)
 def process(*a,**kw):kw['out_path'].mkdir();(kw['out_path']/'.zmetadata').write_text('{}')
 monkeypatch.setattr(m,'process_init',process)
 def fail(*a):raise ValueError('bad field')
 monkeypatch.setattr(m,'validate_product',fail)
 with pytest.raises(ValueError,match='bad field'):m.publish(p,out,5)
 assert not list(out.rglob('.zmetadata'))
 assert not list(tmp_path.glob('.gefs-stage-*'))


def test_archive_manifest_reads_its_own_files_when_original_absolute_path_exists(tmp_path):
 import shutil
 p,data=manifest(tmp_path)
 cold=tmp_path/'cold';cold.mkdir()
 archived=cold/'manifest.json';shutil.copy2(p,archived)
 shutil.copy2(tmp_path/'sample.grib2',cold/'sample.grib2')
 (tmp_path/'sample.grib2').write_bytes(b'GRIBlost')
 resolved=m.check_manifest(load_manifest(archived),5)
 assert set(resolved)=={cold/'sample.grib2'}


def test_missing_archived_file_does_not_silently_read_hot_original(tmp_path):
 import shutil
 p,data=manifest(tmp_path)
 cold=tmp_path/'cold';cold.mkdir()
 archived=cold/'manifest.json';shutil.copy2(p,archived)
 with pytest.raises((OSError,ValueError)):
  m.check_manifest(load_manifest(archived),5)
