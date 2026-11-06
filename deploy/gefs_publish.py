#!/usr/bin/env python3
"""Publish only a complete, eagerly checked GEFS init; private staging is never catalogued."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import xarray as xr
from climate_restore.coverage_index import update_store
from climate_restore.gefs_control import export_gefs_control
from climate_restore.manifest import load_manifest, verify
from climate_restore.processor import process_init, _verify_zarr_chunks
from climate_restore.config import ZarrOptions
from climate_restore.sources.gefs_s2s import GefsEnsAdapter, STEPS, ALL_FIELDS, INSTANT_FIELDS


def check_manifest(manifest, member_count):
    if manifest.source_name != 'gefs-ens' or manifest.cycle != 0: raise ValueError('wrong GEFS init/source')
    if manifest.source_metadata.get('members') != list(range(member_count)): raise ValueError('wrong member selection')
    if [f.step_hours for f in manifest.files] != list(STEPS): raise ValueError('35-day native steps are incomplete')
    paths = verify(manifest, manifest.root_dir)
    for path, entry in zip(paths, manifest.files):
        if path.stat().st_size != entry.size_bytes or hashlib.sha256(path.read_bytes()).hexdigest() != entry.sha256:
            raise ValueError(f'raw checksum mismatch: {path.name}')
    return paths


def validate_product(path, members):
    if _verify_zarr_chunks(path): raise ValueError('missing data chunks')
    with xr.open_zarr(path, consolidated=True) as ds:
        if ds.member.values.tolist() != members or (ds.step.values/np.timedelta64(1,'h')).astype(int).tolist() != list(STEPS):
            raise ValueError('member/time coverage mismatch')
        if set(ds.data_vars) != ALL_FIELDS: raise ValueError('variable coverage mismatch')
        # Eager full-field reads verify all compressed blocks; one variable at a time bounds memory.
        for name in ds.data_vars:
            da = ds[name].load()
            present = ds[da.attrs['s2s_sample_present_coord']].load()
            if int(present.sum()) != len(members)*(181 if name in INSTANT_FIELDS else 180):
                raise ValueError(f'{name}: sample count mismatch')
            if bool((da.isnull() & present).any()) or bool((da.notnull() & ~present).any()):
                raise ValueError(f'{name}: invalid missingness')
        return dict(dimensions=dict(ds.sizes),variables=sorted(ds.data_vars))


def publish(manifest_path, output_dir, member_count):
    manifest = load_manifest(manifest_path)
    paths = check_manifest(manifest, member_count)
    root = Path(output_dir).resolve(); target = root/'gefs-ens'/f'{manifest.date}_00z_gefs-ens.zarr'
    root.parent.mkdir(parents=True, exist_ok=True); target.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    with (root.parent/'.gefs-publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not target.exists():
            stage_root = Path(tempfile.mkdtemp(prefix='.gefs-stage-',dir=root.parent))
            try:
                stage = stage_root/target.name
                adapter = GefsEnsAdapter();adapter.configure_manifest(manifest)
                process_init(paths,adapter=adapter,bbox=(70,140,15,55),out_path=stage,workers=2,
                    output_format='zarr',zarr_options=ZarrOptions(chunks={'step':-1,'member':1,'latitude':64,'longitude':64}))
                validate_product(stage,list(range(member_count)))
                (stage/'.gefs-manifest-sha256').write_text(fingerprint)
                stage.rename(target)
            finally:
                shutil.rmtree(stage_root,ignore_errors=True)
        else:
            if not (target/'.gefs-manifest-sha256').is_file() or (target/'.gefs-manifest-sha256').read_text()!=fingerprint:
                raise ValueError('existing product has different provenance; refusing overwrite')
        result=validate_product(target,list(range(member_count)))
        update_store(target)
        control=export_gefs_control(target,root)
        validate_product(control,[0])
        print(json.dumps(dict(status='complete',member_store=str(target),control_store=str(control),**result)),flush=True)
        return target,control


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--member-count',type=int,default=5,choices=range(1,32))
    a=p.parse_args();publish(a.manifest,a.output_dir,a.member_count)
