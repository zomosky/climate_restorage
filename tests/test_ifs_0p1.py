from pathlib import Path

import numpy as np
import xarray as xr
import cfgrib

from climate_restore.config import load_job
from climate_restore.sources import get_source
from climate_restore.sources.aifs import AifsAdapter


def test_independent_ifs_0p1_reuses_rename_contract_without_replacing_025():
    adapter = get_source("ifs-hres-0p1")
    assert adapter is not get_source("ifs-hres")
    assert get_source("ifs-hres") is AifsAdapter
    assert adapter.RULES == AifsAdapter.RULES
    assert adapter().name == "ifs-hres-0p1"


def test_server_restore_limits_workers_and_keeps_native_grid():
    cfg = load_job(Path(__file__).parents[1] / "config/jobs/ifs_0p1_zarr_china_ser.yaml")
    assert cfg.workers == 2
    assert cfg.bbox == (70.0, 140.0, 15.0, 55.0)
    assert cfg.output_format == "zarr" and cfg.zarr.consolidated


def test_crop_keeps_native_boundary_cells_despite_cfgrib_coordinate_roundoff():
    latitude = [55.00000000000135, 54.90000000000135, 15.0000000000008]
    longitude = [69.99999999999434, 70.09999999999434, 139.99999999999037]
    values = np.arange(9, dtype=np.float32).reshape(3, 3)
    raw = xr.Dataset({"t2m": (("latitude", "longitude"), values)},
                     coords={"latitude": latitude, "longitude": longitude})
    cropped = get_source("ifs-hres-0p1")().crop_bbox(raw, (70, 140, 15, 55))
    assert dict(cropped.sizes) == {"latitude": 3, "longitude": 3}
    np.testing.assert_array_equal(cropped.latitude, [15, 54.9, 55])
    np.testing.assert_array_equal(cropped.longitude, [70, 70.1, 140])
    np.testing.assert_array_equal(cropped.t2m, values[::-1])
    np.testing.assert_array_equal(raw.latitude, latitude)


def test_analysis_file_preview_cannot_override_canonical_forecast(monkeypatch):
    instant = xr.Dataset({"t2m": (("latitude", "longitude"), [[280.0]])},
                         coords={"latitude": [30], "longitude": [100], "step": np.timedelta64(0, "h")})
    preview = xr.Dataset({"mn2t3": (("latitude", "longitude"), [[0.0]])},
                         coords={"latitude": [30], "longitude": [100], "step": np.timedelta64(3, "h")})
    monkeypatch.setattr(cfgrib, "open_datasets", lambda *args, **kwargs: [instant, preview])
    result = get_source("ifs-hres-0p1")().open_hypercubes(Path("f000.subset.grib2"))
    assert len(result) == 1 and list(result[0].data_vars) == ["t2m"]
    assert float(result[0].t2m.item()) == 280
