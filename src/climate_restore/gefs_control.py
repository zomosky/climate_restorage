"""Publish an atomic physical GEFS control product from a consolidated member store."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import xarray as xr

from .processor import write_dataset


def export_gefs_control(input_path: Path, output_dir: Path) -> Path:
    """Extract member zero; validate repeat calls and never overwrite conflicting data.

    The member dimension remains in storage so statistical metadata keep their
    original shape. The reader removes it only at the query output boundary.
    """
    input_path = Path(input_path).resolve()
    metadata = input_path / '.zmetadata'
    if not metadata.is_file():
        raise ValueError('GEFS input must be a consolidated complete Zarr product')
    with xr.open_zarr(input_path, consolidated=True) as original:
        if original.attrs.get('s2s_source') not in {'gefs-s2s', 'gefs-ens'}:
            raise ValueError('input must be a gefs-s2s member product')
        if ('member' not in original.dims or not np.issubdtype(original.member.dtype, np.integer)
                or len(np.unique(original.member)) != original.sizes['member']
                or 0 not in original.member.values):
            raise ValueError('GEFS input must contain a unique control member 0')
        ds = original.sel(member=[0]).load()
    coverage = {}
    for name in ds.data_vars:
        da = ds[name]
        keys = [f's2s_{key}_coord' for key in ('sample_present', 'window_start', 'window_end')]
        if any(da.attrs.get(key) not in ds.coords for key in keys):
            raise ValueError(f'{name}: missing statistical window metadata')
        present, start, end = [ds[da.attrs[key]] for key in keys]
        if present.dtype != bool or 'member' not in present.dims:
            raise ValueError(f'{name}: invalid presence metadata')
        if (bool((present & (start.isnull() | end.isnull() | (start > end))).any())
                or bool((~present & (start.notnull() | end.notnull())).any())
                or bool((da.notnull() & ~present).any())):
            raise ValueError(f'{name}: inconsistent source data/window metadata')
        coverage[name] = {'members': [0] if bool(present.any()) else [],
                          'sample_count': int(present.sum())}
        # This is a count of stored messages, not the parent ensemble size.
        # GRIB_numberOfForecastsInEnsemble/totalNumber and
        # gefs_perturbed_members_metadata describe the original 30 perturbations
        # and intentionally remain unchanged as source-header provenance.
        da.attrs.update(s2s_sample_count=int(present.sum()),
                        GRIB_dataType='cf', GRIB_typeOfEnsembleForecast=1)
    ds.attrs.update(s2s_source='gefs-ctrl',
                    s2s_message_coverage=json.dumps(coverage, sort_keys=True),
                    gefs_member_types=json.dumps({'0': 'cf'}), gefs_control_only=1,
                    gefs_actual_member_count=1, gefs_selected_members='[0]', gefs_full_ensemble=0,
                    gefs_parent_source=original.attrs.get('s2s_source', 'gefs-ens'), gefs_parent_store=str(input_path),
                    gefs_parent_metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
                    GRIB_dataType='cf', GRIB_typeOfEnsembleForecast=1)
    stamp = np.datetime_as_string(ds.time.values, unit='h').replace('-', '').replace('T', '_')
    target = Path(output_dir).resolve() / 'gefs-ctrl' / f'{stamp}z_gefs-ctrl.zarr'
    if target == input_path:
        raise ValueError('input and output must differ')
    target.parent.mkdir(parents=True, exist_ok=True)
    # A process lock serializes repeat callers without publishing a half-written store.
    import fcntl
    with (target.parent / f'.{target.name}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            with xr.open_zarr(target, consolidated=True) as existing:
                try:
                    xr.testing.assert_identical(existing.load(), ds)
                except AssertionError as exc:
                    raise ValueError(f'existing control product differs: {target}') from exc
            from .coverage_index import update_store
            update_store(target)
            return target
        stage = Path(tempfile.mkdtemp(prefix='.gefs-control-', suffix='.zarr', dir=target.parent))
        try:
            write_dataset(ds, stage, fmt='zarr')
            with xr.open_zarr(stage, consolidated=True) as written:
                xr.testing.assert_identical(written.load(), ds)
            stage.rename(target)
            from .coverage_index import update_store
            update_store(target)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return target
