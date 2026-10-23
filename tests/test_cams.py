"""Metadata-only CAMS semantics; raw values, native leads and other adapters stay intact."""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from climate_restore.config import load_job
from climate_restore.sources import get_source
from climate_restore.sources.cams import CamsAdapter


def cube(param, name, values, units, kind="instant", leads=None):
    leads = leads if leads is not None else list(range(len(values)))
    return xr.Dataset({name: (("step", "latitude", "longitude"), np.asarray(values, dtype="float32")[:, None, None],
                             {"GRIB_paramId": param, "GRIB_dataType": "fc", "GRIB_typeOfLevel": "surface",
                              "GRIB_stepType": kind, "units": units})},
                      coords={"step": np.array(leads, dtype="timedelta64[h]"), "latitude": [30.0],
                              "longitude": [100.0], "surface": 0})


@pytest.mark.parametrize("param,name,units", [(169, "ssrd", "J m**-2"), (228, "tp", "kg m**-2")])
def test_cumulative_mapping_changes_only_attrs(param, name, units):
    ds = cube(param, name, [0, 3600, 7200], units)
    result = CamsAdapter().rename_hypercube(ds)[0]
    out = result[f"{name}_accum"]
    np.testing.assert_array_equal(out, ds[name])
    assert out.attrs["GRIB_stepType"] == "accum"
    assert out.attrs["cams_original_GRIB_stepType"] == "instant"
    assert ds[name].attrs["GRIB_stepType"] == "instant"


def test_h0_boundary_and_units_are_not_relaxed():
    with pytest.raises(ValueError, match="Nonzero"):
        CamsAdapter().rename_hypercube(cube(169, "ssrd", [1, 3600], "J m**-2"))
    with pytest.raises(ValueError, match="Unverified"):
        CamsAdapter().rename_hypercube(cube(169, "ssrd", [0, 1], "W m**-2"))


def test_aod_stays_instant_and_gust_partitions_keep_native_times():
    ds = cube(210207, "aod550", [0.1, 0.2], "~")
    out = CamsAdapter().rename_hypercube(ds)[0]
    np.testing.assert_array_equal(out.aod550, ds.aod550)
    assert out.aod550.attrs["GRIB_stepType"] == "instant"
    for steps, kind in [([0], "instant"), ([3, 6, 120], "max")]:
        gust = cube(228028, "10fg3", list(range(len(steps))), "m s**-1", kind, steps)
        mapped = CamsAdapter().rename_hypercube(gust)[0]
        assert len(mapped.data_vars) == 1
        a = next(iter(mapped.data_vars.values()))
        np.testing.assert_array_equal(a.step, gust.step)
        np.testing.assert_array_equal(a, gust["10fg3"])
        assert a.attrs["GRIB_stepType"] == "max" and a.attrs["cams_native_interval_hours"] == 3


def test_registered_source_and_single_worker_job():
    assert get_source("ec_cams") is CamsAdapter
    cfg = load_job(Path(__file__).parents[1] / "config/jobs/cams_zarr_china.yaml")
    assert cfg.workers == 1 and cfg.output_format == "zarr"
