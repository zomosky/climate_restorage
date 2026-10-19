"""Real, tiny GRIB probability records exercise the cfgrib boundary."""
from pathlib import Path

import eccodes as ec
import numpy as np
import pytest


def write_messages(path: Path, specs):
    with path.open("wb") as handle:
        for param, start, end, statistic, value in specs:
            g = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            try:
                for key, val in {"Ni":3,"Nj":2,"latitudeOfFirstGridPointInDegrees":31.0,
                                 "latitudeOfLastGridPointInDegrees":30.0,
                                 "longitudeOfFirstGridPointInDegrees":100.0,
                                 "longitudeOfLastGridPointInDegrees":102.0,
                                 "iDirectionIncrementInDegrees":1.0,"jDirectionIncrementInDegrees":1.0,
                                 "dataDate":20261015,"dataTime":0}.items():
                    ec.codes_set(g,key,val)
                ec.codes_set(g,"dataType","ep")
                ec.codes_set(g,"stepType",statistic)
                ec.codes_set(g,"paramId",param)
                ec.codes_set(g,"startStep",start)
                ec.codes_set(g,"endStep",end)
                ec.codes_set_values(g,np.full(6,value))
                ec.codes_write(g,handle)
            finally:
                ec.codes_release(g)


def test_aifs_same_threshold_different_statistics_do_not_collide(tmp_path):
    from climate_restore.sources.probabilities import ProbabilityAdapter
    path=tmp_path/"events.grib2"
    write_messages(path,[(131073,240,240,"instant",30),(131073,120,168,"max",40),
                         (131073,168,240,"max",50),(131073,120,240,"max",60)])
    adapter=ProbabilityAdapter("aifs-ens-probabilities")
    renamed=[out for ds in adapter.open_hypercubes(path) for out in adapter.rename_hypercube(ds)]
    names={v for ds in renamed for v in ds.data_vars}
    assert names == {"2tl273","2tl273_max48h","2tl273_max72h","2tl273_max120h"}
    for ds in renamed:
        v=next(iter(ds.data_vars))
        da=ds[v]
        assert da.attrs["units"] == "%"
        assert da.attrs["zrread_kind"] == "probability"
        start=ds[da.attrs["probability_window_start_coord"]].values
        end=ds[da.attrs["probability_window_end_coord"]].values
        assert np.all((end-start)/np.timedelta64(1,"h") == da.attrs["probability_window_hours"])


def test_rain_probability_has_threshold_and_original_24h_window(tmp_path):
    from climate_restore.sources.probabilities import ProbabilityAdapter
    path=tmp_path/"rain.grib2"
    write_messages(path,[(131099,0,24,"accum",70),(131099,12,36,"accum",20)])
    adapter=ProbabilityAdapter("ifs-ens-probabilities")
    ds=adapter.rename_hypercube(adapter.open_hypercubes(path)[0])[0].load()
    assert set(ds.data_vars)=={"tpg50"}
    assert ds.tpg50.attrs["probability_threshold"]==50
    assert ds.tpg50.attrs["probability_threshold_units"]=="mm"
    assert ds.tpg50.attrs["probability_statistic"]=="accum"
    assert ds.tpg50.attrs["probability_type"]==3
    assert ds.tpg50.attrs["probability_comparison"]=="gt"
    assert ds.tpg50.values[:,0,0].tolist()==[70,20]
    assert ds[ds.tpg50.attrs["probability_window_start_coord"]].values.astype(str).tolist()==[
        "2026-10-15T00:00:00.000000000","2026-10-15T12:00:00.000000000"]
    with pytest.raises(ValueError,match="incomplete"):
        adapter.validate_product(ds)


def test_historical_missing_receipt_allows_only_declared_available_windows(tmp_path):
    from types import SimpleNamespace
    from climate_restore.sources.probabilities import ProbabilityAdapter, expected_samples
    path=tmp_path/"partial.grib2"
    write_messages(path,[(131099,0,24,"accum",70),(131099,12,36,"accum",20)])
    adapter=ProbabilityAdapter("ifs-ens-probabilities")
    ds=adapter.rename_hypercube(adapter.open_hypercubes(path)[0])[0].load()
    full=expected_samples(adapter.name)
    available={"tpg50":[24,36]}
    contract=dict(schema_version=1,policy="historical_available",status="partial",
        expected_samples={v:sorted(s) for v,s in full.items()},available_samples=available,
        missing_samples={v:sorted(s-set(available.get(v,[]))) for v,s in full.items()},
        missing_variables=sorted(set(full)-set(available)))
    adapter.configure_manifest(SimpleNamespace(probability_availability=contract))
    adapter.validate_product(ds)
    assert ds.attrs["probability_complete"]==1
    assert ds.attrs["probability_catalog_complete"]==0
    assert ds.tpg50.attrs["probability_sample_count"]==2
    assert ds.tpg50.values[:,0,0].tolist()==[70,20]
    broken=ds.copy(deep=True)
    broken.tpg50.values[1]=float("nan")
    with pytest.raises(ValueError,match="incomplete probability windows"):
        adapter.validate_product(broken)
    contract["missing_samples"]["tpg50"]=[]
    with pytest.raises(ValueError,match="inconsistent"):
        adapter.validate_product(ds)


def test_history_publication_rejects_a_wrong_grib_reference_time():
    import importlib.util
    from types import SimpleNamespace
    import xarray as xr
    path=Path(__file__).parents[1]/'scripts/restore_probability_history.py'
    spec=importlib.util.spec_from_file_location('history_restore',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    calls=[]
    adapter=SimpleNamespace(validate_product=lambda ds:calls.append(True))
    manifest=SimpleNamespace(init_time='2026-10-15T12:00:00+00:00')
    ds=xr.Dataset(coords=dict(time=np.datetime64('2026-10-15T00:00','ns')))
    with pytest.raises(ValueError,match='init differs'):
        helper.validate_dataset(ds,manifest,adapter)
    assert not calls
    helper.validate_dataset(ds.assign_coords(time=np.datetime64('2026-10-15T12:00','ns')),manifest,adapter)
    assert calls==[True]
