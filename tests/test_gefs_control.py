"""Physical export failures: missing control, stale coverage, partial publication,
conflicting repeats, mutation of the parent member product."""
import json
import numpy as np
import pytest
import xarray as xr
from climate_restore.gefs_control import export_gefs_control


def source(tmp_path, members=(0, 1)):
    times = np.array(['2026-10-29T00','2026-10-29T03'], dtype='datetime64[ns]')
    shape = (2, len(members))
    ds = xr.Dataset({'t2m': (('step','member','latitude','longitude'),
        np.arange(2*len(members), dtype=float).reshape(2,len(members),1,1))},
        coords={'time':times[0], 'step':times-times[0], 'valid_time':('step',times),
            'member':list(members), 'latitude':[39.], 'longitude':[117.]},
        attrs={'s2s_source':'gefs-s2s', 'gefs_actual_member_count':len(members)})
    for key, name, values in [('sample_present','present',np.ones(shape,dtype=bool)),
            ('window_start','start',np.broadcast_to(times[:,None],shape)),
            ('window_end','end',np.broadcast_to(times[:,None],shape))]:
        ds = ds.assign_coords({name:(('step','member'),values)})
        ds.t2m.attrs[f's2s_{key}_coord']=name
    ds.t2m.attrs.update(s2s_sample_count=int(ds.present.sum()),
        GRIB_numberOfForecastsInEnsemble=30, GRIB_totalNumber=30,
        gefs_perturbed_members_metadata=30)
    path=tmp_path/'input.zarr';ds.to_zarr(path,consolidated=True,zarr_format=2)
    return path,ds


def test_export_exact_atomic_and_repeatable(tmp_path):
    path, original=source(tmp_path)
    target=export_gefs_control(path,tmp_path/'output')
    assert target.name=='20261029_00z_gefs-ctrl.zarr'
    assert (target/'.zmetadata').is_file()
    with xr.open_zarr(target) as ds:
        np.testing.assert_array_equal(ds.t2m,original.t2m.sel(member=[0]))
        assert ds.member.values.tolist()==[0]
        assert ds.attrs['gefs_actual_member_count']==1
        assert ds.attrs['gefs_selected_members']=='[0]'
        assert ds.t2m.attrs['s2s_sample_count']==2
        for key in ['GRIB_numberOfForecastsInEnsemble','GRIB_totalNumber','gefs_perturbed_members_metadata']:
            assert ds.t2m.attrs[key]==30
        assert json.loads(ds.attrs['s2s_message_coverage'])['t2m']=={'members':[0],'sample_count':2}
        for name in ['present','start','end']:
            xr.testing.assert_equal(ds[name],original[name].sel(member=[0]))
    assert export_gefs_control(path,tmp_path/'output')==target
    with xr.open_zarr(path) as ds: xr.testing.assert_identical(ds,original)
    assert not list(target.parent.glob('.gefs-control-*'))


def test_missing_control_rejected(tmp_path):
    path,_=source(tmp_path,members=(1,2))
    with pytest.raises(ValueError,match='control member 0'):
        export_gefs_control(path,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_failed_write_never_publishes(tmp_path,monkeypatch):
    path,_=source(tmp_path)
    def fail(*args,**kwargs): raise RuntimeError('interrupted')
    monkeypatch.setattr('climate_restore.gefs_control.write_dataset',fail)
    with pytest.raises(RuntimeError,match='interrupted'): export_gefs_control(path,tmp_path/'output')
    assert not list((tmp_path/'output').rglob('*.zarr'))
    assert not list((tmp_path/'output').rglob('.gefs-control-*'))


def test_changed_parent_refuses_to_replace_existing(tmp_path):
    path,original=source(tmp_path)
    target=export_gefs_control(path,tmp_path/'output')
    original.t2m.values[0,0]=999
    original.to_zarr(path,mode='w',consolidated=True,zarr_format=2)
    with pytest.raises(ValueError,match='differs'): export_gefs_control(path,tmp_path/'output')
    with xr.open_zarr(target) as ds: assert ds.t2m.values[0,0,0,0]==0


def test_control_sample_count_uses_presence_including_missing_messages(tmp_path):
    path, ds = source(tmp_path)
    # One absent control message; all perturbed messages remain present.
    ds.present.values[0,0] = False
    ds.start.values[0,0] = np.datetime64('NaT', 'ns')
    ds.end.values[0,0] = np.datetime64('NaT', 'ns')
    ds.t2m.values[0,0] = np.nan
    ds.t2m.attrs['s2s_sample_count'] = 3
    ds.to_zarr(path, mode='w', consolidated=True, zarr_format=2)
    target = export_gefs_control(path, tmp_path/'output')
    with xr.open_zarr(target) as control:
        assert control.t2m.attrs['s2s_sample_count'] == int(control.present.sum()) == 1
        assert json.loads(control.attrs['s2s_message_coverage'])['t2m']['sample_count'] == 1
    with xr.open_zarr(path) as parent:
        assert parent.t2m.attrs['s2s_sample_count'] == 3
