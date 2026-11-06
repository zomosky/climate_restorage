"""S2S failure contracts: grouped manifests, missing members, collisions and windows."""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from climate_restore.manifest import load_manifest
from climate_restore.sources.s2s import S2SAdapter, read_headers


def test_group_manifest_keeps_request_and_normalizes_perturbed(tmp_path):
    raw = dict(
        schema_version=1,
        kind="s2s",
        source={"name": "s2s-ecmwf", "forecast_type": "perturbed_forecast"},
        init_time="2026-10-29T00:00:00+00:00",
        date="20261029",
        cycle=0,
        completed_at="done",
        files=[
            dict(
                path="radiation.grib2",
                size_bytes=1,
                sha256="",
                group="radiation",
                request={"forecast_type": "perturbed_forecast"},
            )
        ],
    )
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(raw))
    m = load_manifest(p)
    assert m.source_name == "s2s-ecmwf-perturbed"
    assert m.files[0].step_hours is None
    assert m.files[0].group == "radiation"
    assert m.files[0].request["forecast_type"] == "perturbed_forecast"
    del raw["kind"]
    p.write_text(json.dumps(raw))
    with pytest.raises((KeyError, ValueError)):
        load_manifest(p)


def cube(member, hours=(24, 48), name="ssrd_accum", init="2026-01-01", value=1.0):
    time = np.datetime64(init, "ns")
    steps = np.array(hours, dtype="timedelta64[h]")
    p, s, e = (
        f"s2s_present__{name}",
        f"s2s_window_start__{name}",
        f"s2s_window_end__{name}",
    )
    ds = xr.Dataset(
        {
            name: (
                ("step", "member", "latitude", "longitude"),
                np.full((len(hours), 1, 1, 1), value),
            )
        },
        coords={
            "step": steps,
            "member": [member],
            "latitude": [30.0],
            "longitude": [110.0],
            "time": time,
            p: (("step", "member"), np.ones((len(hours), 1))),
            s: (("step", "member"), np.full((len(hours), 1), time)),
            e: (("step", "member"), (time + steps)[:, None]),
        },
        attrs={"s2s_grid_hash": "grid"},
    )
    ds[name].attrs.update(
        s2s_sample_present_coord=p,
        s2s_window_start_coord=s,
        s2s_window_end_coord=e,
        GRIB_stepType="accum",
    )
    return ds


def test_coordinate_merge_keeps_members_sparse_windows_and_nan_values():
    a = S2SAdapter()
    ds = a.merge_hypercubes(
        [cube(0), cube(1, (168, 336)), cube(0, (6, 12), name="u10", value=np.nan)]
    )
    assert ds.member.values.tolist() == [0, 1]
    assert int(ds.s2s_present__ssrd_accum.sum()) == 4
    assert (
        int(ds.s2s_present__u10.sum()) == 2
    )  # presence comes from messages, not numeric values
    assert not ds.s2s_present__ssrd_accum.sel(step=np.timedelta64(6, "h")).any()
    assert np.isnat(
        ds.s2s_window_start__ssrd_accum.sel(step=np.timedelta64(6, "h"))
    ).all()
    assert ds.ssrd_accum.sel(member=1, step=np.timedelta64(24, "h")).isnull().all()


@pytest.mark.parametrize(
    "change,match", [("duplicate", "duplicate"), ("time", "init"), ("grid", "grid")]
)
def test_merge_rejects_conflicts(change, match):
    first = cube(0)
    second = cube(0 if change == "duplicate" else 1)
    if change == "time":
        second = second.assign_coords(time=np.datetime64("2026-01-02", "ns"))
    if change == "grid":
        second.attrs["s2s_grid_hash"] = "other"
    with pytest.raises(ValueError, match=match):
        S2SAdapter().merge_hypercubes([first, second])


def header(member=1, end=24, **overrides):
    return dict(
        param=169,
        level_type="surface",
        level=0,
        statistic="accum",
        start=0,
        end=end,
        member=member,
        forecast_type="pf",
        ensemble_size=3,
        init="2026-01-01T00:00:00",
        grid="grid",
        **overrides,
    )


def test_missing_member_and_sparse_native_cadence():
    a = S2SAdapter()
    req = {
        "forecast_type": "perturbed_forecast",
        "variable": ["surface_solar_radiation_downwards"],
        "leadtime_hour": ["6", "12", "18", "24"],
    }
    with pytest.raises(ValueError, match="missing|coverage"):
        a.validate_headers([header(1)], req)
    a.validate_headers([header(1), header(2)], req)
    with pytest.raises(ValueError, match="duplicate"):
        a.validate_headers([header(1), header(1), header(2)], req)


def test_real_control_headers_and_cropped_decode():
    from climate_restore.processor import _decode_one_grib

    path = Path("../docs/analysis/s2s-pilot-20261104/raw/control_wind_precip.grib2")
    if not path.exists():
        pytest.skip("local genuine S2S fixture absent")
    headers = read_headers(path)
    assert {h["member"] for h in headers} == {0}
    a = S2SAdapter()
    cubes = _decode_one_grib((path, a, (109.0, 112.0, 29.0, 32.0)))
    ds = a.merge_hypercubes([c[2] for c in cubes])
    a.validate_product(ds)
    assert set(ds.data_vars) == {"u10", "v10", "tp_accum"}
    assert ds.sizes["member"] == 1
    assert ds.sizes["latitude"] <= 3 and ds.sizes["longitude"] <= 3
    assert ds.s2s_present__tp_accum.all()


@pytest.mark.parametrize("fmt,suffix", [("zarr", ".zarr"), ("netcdf", ".nc")])
def test_sparse_window_serialization(tmp_path, fmt, suffix):
    from climate_restore.processor import write_dataset

    a = S2SAdapter()
    ds = a.merge_hypercubes(
        [cube(0), cube(1, (168, 336)), cube(0, (6, 12), name="u10")]
    )
    a.validate_product(ds)
    path = tmp_path / "s2s-ecmwf-ensemble" / f"20260101_00z_s2s-ecmwf-ensemble{suffix}"
    write_dataset(ds, path, fmt=fmt)
    with (
        xr.open_zarr(path, consolidated=True)
        if fmt == "zarr"
        else xr.open_dataset(path)
    ) as actual:
        actual.load()
        for c in ds.coords:
            xr.testing.assert_equal(ds[c], actual[c])


def test_member_chunked_sparse_fields_keep_integrity_checks(tmp_path):
    from climate_restore.config import ZarrOptions
    from climate_restore.processor import write_dataset, _verify_zarr_chunks

    adapter = S2SAdapter()
    ds = adapter.merge_hypercubes(
        [cube(0), cube(1, (168, 336)), cube(0, (6, 12), name="u10")]
    )
    adapter.validate_product(ds)
    path = tmp_path / "20260101_00z_s2s-ecmwf-ensemble.zarr"
    write_dataset(
        ds,
        path,
        fmt="zarr",
        zarr_options=ZarrOptions(
            chunks={"member": 1, "step": 1, "latitude": 1, "longitude": 1}
        ),
    )
    with xr.open_zarr(path, consolidated=True) as actual:
        xr.testing.assert_equal(ds, actual.load())
        assert actual.u10.sel(member=1).isnull().all()
        assert not actual.s2s_present__u10.sel(member=1).any()
    assert not _verify_zarr_chunks(path)
    # Empty member/time blocks are deliberate; losing a real block must still fail.
    (path / "u10" / "0.0.0.0").unlink()
    assert any(error.startswith("u10:") for error in _verify_zarr_chunks(path))


def test_header_init_type_level_and_window_validation():
    a = S2SAdapter()
    with pytest.raises(ValueError, match="init"):
        a.validate_headers([header(0)], {"year": ["2027"]})
    records = [dict(header(0), forecast_type="cf")]
    with pytest.raises(ValueError, match="member"):
        a.validate_headers(records, {"forecast_type": "perturbed_forecast"})
    records = [dict(header(0), forecast_type="cf", start=1)]
    with pytest.raises(ValueError, match="start"):
        a.validate_headers(records, {"forecast_type": "control_forecast"})


def write_grib(path, members, *, param=165, hours=(6, 12)):
    import eccodes as ec

    with path.open("wb") as output:
        for member in members:
            for hour in hours:
                gid = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
                try:
                    for key, value in dict(
                        productDefinitionTemplateNumber=1,
                        Ni=2,
                        Nj=2,
                        latitudeOfFirstGridPointInDegrees=31.0,
                        latitudeOfLastGridPointInDegrees=30.0,
                        longitudeOfFirstGridPointInDegrees=110.0,
                        longitudeOfLastGridPointInDegrees=111.0,
                        iDirectionIncrementInDegrees=1.0,
                        jDirectionIncrementInDegrees=1.0,
                        dataDate=20260101,
                        dataTime=0,
                        paramId=param,
                        step=hour,
                        numberOfForecastsInEnsemble=3,
                        perturbationNumber=member,
                        typeOfEnsembleForecast=0 if member == 0 else 3,
                    ).items():
                        ec.codes_set(gid, key, value)
                    ec.codes_set(gid, "dataType", "cf" if member == 0 else "pf")
                    if param in (121, 122):
                        ec.codes_set(gid, "startStep", hour - 6)
                        ec.codes_set(gid, "endStep", hour)
                    ec.codes_set_values(gid, np.full(4, member * 10 + hour))
                    ec.codes_write(gid, output)
                finally:
                    ec.codes_release(gid)


def test_real_grib_member_blocks_and_formal_processor(tmp_path):
    from types import SimpleNamespace
    from climate_restore.processor import process_init

    control = tmp_path / "control.grib2"
    perturbed = tmp_path / "perturbed.grib2"
    write_grib(control, [0])
    write_grib(perturbed, [1, 2])
    a = S2SAdapter("s2s-ecmwf-ensemble")
    a.configure_manifest(
        SimpleNamespace(
            kind="s2s",
            init_time="2026-01-01T00:00:00+00:00",
            date="20260101",
            cycle=0,
            source_metadata={"forecast_type": "ensemble"},
            files=[
                SimpleNamespace(
                    path=p,
                    request={
                        "forecast_type": kind,
                        "variable": ["10_m_u_component_of_wind"],
                        "leadtime_hour": ["6", "12"],
                    },
                )
                for p, kind in [
                    (control, "control_forecast"),
                    (perturbed, "perturbed_forecast"),
                ]
            ],
        )
    )
    cubes = a.open_hypercubes(perturbed)
    assert len(cubes) == 2
    assert all(ds.sizes.get("number", 1) == 1 for ds in cubes)
    out = tmp_path / "s2s-ecmwf-ensemble" / "20260101_00z_s2s-ecmwf-ensemble.zarr"
    process_init(
        [control, perturbed],
        adapter=a,
        bbox=(109, 112, 29, 32),
        out_path=out,
        workers=1,
        output_format="zarr",
    )
    with xr.open_zarr(out, consolidated=True) as actual:
        assert actual.member.values.tolist() == [0, 1, 2]
        assert actual.u10.sel(
            member=2, step=np.timedelta64(12, "h")
        ).values.tolist() == [[32.0, 32.0], [32.0, 32.0]]
        assert actual.s2s_present__u10.all()
    write_grib(perturbed, [1])
    with pytest.raises(ValueError, match="missing"):
        a.open_hypercubes(perturbed)


def extension_header(param, member, end):
    kind = {121: "max", 122: "min", 134: "instant"}[param]
    return dict(
        header(member, end),
        param=param,
        statistic=kind,
        level_type="heightAboveGround" if param != 134 else "surface",
        level=2 if param != 134 else 0,
        start=end - 6 if param != 134 else end,
        forecast_type="cf" if member == 0 else "pf",
        ensemble_size=101,
    )


@pytest.mark.parametrize(
    "param,variable,hours",
    [
        (121, "maximum_2_m_temperature_in_the_last_6_hours", range(6, 1105, 6)),
        (122, "minimum_2_m_temperature_in_the_last_6_hours", range(6, 1105, 6)),
        (134, "surface_pressure", range(0, 1105, 24)),
    ],
)
def test_full_horizon_extension_header_coverage(param, variable, hours):
    adapter = S2SAdapter("s2s-ecmwf-perturbed")
    records = [
        extension_header(param, member, end)
        for member in range(1, 101)
        for end in hours
    ]
    request = {
        "forecast_type": "perturbed_forecast",
        "variable": [variable],
        "leadtime_hour": [str(h) for h in hours],
    }
    adapter.validate_headers(records, request)
    with pytest.raises(ValueError, match="coverage"):
        adapter.validate_headers(records[:-1], request)


@pytest.mark.parametrize("param", [121, 122])
@pytest.mark.parametrize(
    "change",
    [
        dict(start=1),
        dict(start=12),
        dict(end=0, start=0),
        dict(start=1104, end=1110),
        dict(statistic="avg"),
    ],
)
def test_extreme_headers_reject_wrong_window_or_statistic(param, change):
    record = dict(extension_header(param, 0, 12), **change)
    with pytest.raises(ValueError, match="6.hour|extreme"):
        S2SAdapter().validate_headers([record], {"forecast_type": "control_forecast"})


@pytest.mark.parametrize("param", [121, 122])
def test_extreme_request_rejects_zero_hour(param):
    record = extension_header(param, 0, 6)
    with pytest.raises(ValueError, match="6.hour|extreme"):
        S2SAdapter().validate_headers(
            [record], {"forecast_type": "control_forecast", "leadtime_hour": ["0", "6"]}
        )


def test_surface_pressure_rejects_noninstant_statistic():
    record = dict(extension_header(134, 0, 24), statistic="avg")
    with pytest.raises(ValueError, match="surface pressure"):
        S2SAdapter().validate_headers([record], {"forecast_type": "control_forecast"})


@pytest.mark.parametrize("fmt,suffix", [("zarr", ".zarr"), ("netcdf", ".nc")])
def test_synthetic_extremes_and_surface_pressure_grib_roundtrip(tmp_path, fmt, suffix):
    """Synthetic ecCodes messages test actual decoding; these are not live S2S samples."""
    from types import SimpleNamespace
    from climate_restore.processor import process_init
    from climate_restore.sources.s2s import PARAMETERS

    entries = []
    for param in (134, 121, 122):
        hours = (0, 24, 1104) if param == 134 else (6, 12, 1104)
        path = tmp_path / f"{param}.grib2"
        write_grib(path, [0], param=param, hours=hours)
        headers = read_headers(path)
        assert {r["param"] for r in headers} == {param}
        entries.append(
            SimpleNamespace(
                path=path,
                request={
                    "forecast_type": "control_forecast",
                    "variable": [PARAMETERS[param][1]],
                    "leadtime_hour": [str(h) for h in hours],
                },
            )
        )
    adapter = S2SAdapter()
    adapter.configure_manifest(
        SimpleNamespace(
            kind="s2s",
            init_time="2026-01-01T00:00:00+00:00",
            date="20260101",
            cycle=0,
            source_metadata={"forecast_type": "control_forecast"},
            files=entries,
        )
    )
    out = tmp_path / "s2s-ecmwf" / f"20260101_00z_s2s-ecmwf{suffix}"
    process_init(
        [e.path for e in entries],
        adapter=adapter,
        bbox=(109, 112, 29, 32),
        out_path=out,
        workers=1,
        output_format=fmt,
    )
    assert_single_field_coordinates(out, fmt, ["sp", "mx2t6_max", "mn2t6_min"])
    opener = xr.open_zarr if fmt == "zarr" else xr.open_dataset
    with opener(out) as ds:
        ds.load()
        assert set(ds.data_vars) == {"sp", "mx2t6_max", "mn2t6_min"}
        for name in ("mx2t6_max", "mn2t6_min"):
            da = ds[name]
            present = ds[da.attrs["s2s_sample_present_coord"]]
            start = ds[da.attrs["s2s_window_start_coord"]]
            end = ds[da.attrs["s2s_window_end_coord"]]
            assert not present.sel(step=np.timedelta64(0, "h")).any()
            assert (
                (end - start).where(present).values[present.values]
                == np.timedelta64(6, "h")
            ).all()
            assert da.sel(step=np.timedelta64(1104, "h"), member=0).values.tolist() == [
                [1104.0, 1104.0],
                [1104.0, 1104.0],
            ]
            assert np.isnat(start.sel(step=np.timedelta64(0, "h"))).all()
            assert da.attrs["units"] == "K"
        assert ds.sp.attrs["units"] == "Pa"
        assert ds.sp.attrs["GRIB_stepType"] == "instant"
        assert ds.s2s_present__sp.sel(step=np.timedelta64(0, "h")).all()


@pytest.mark.parametrize(
    "change", [dict(end=6, start=6), dict(end=1128, start=1128), dict(level=10)]
)
def test_surface_pressure_rejects_wrong_native_coordinate(change):
    record = dict(extension_header(134, 0, 24), **change)
    with pytest.raises(ValueError, match="surface pressure"):
        S2SAdapter().validate_headers([record], {"forecast_type": "control_forecast"})


def test_product_extreme_window_must_remain_six_hours():
    adapter = S2SAdapter()
    ds = adapter.merge_hypercubes([cube(0, (6, 12), name="mx2t6_max")])
    ds.mx2t6_max.attrs.update(GRIB_paramId=121, GRIB_stepType="max")
    with pytest.raises(ValueError, match="6-hour extreme"):
        adapter.validate_product(
            ds
        )  # second inherited [0,12] window is not a 6h maximum


def assert_single_field_coordinates(path, fmt, names):
    """Opening one field must not depend on another field's CF coordinates attr."""
    opener = xr.open_zarr if fmt == "zarr" else xr.open_dataset
    options = {"consolidated": True} if fmt == "zarr" else {}
    with opener(path, **options) as full:
        full.load()
        for name in names:
            with opener(
                path, drop_variables=[v for v in names if v != name], **options
            ) as single:
                single.load()
                assert "member" in single.coords
                for key in (
                    "s2s_sample_present_coord",
                    "s2s_window_start_coord",
                    "s2s_window_end_coord",
                ):
                    coord = single[name].attrs[key]
                    assert coord in single.coords, (name, coord, list(single.data_vars))
                    xr.testing.assert_equal(
                        full[coord].variable, single[coord].variable
                    )
                declared = set(single[name].encoding.get("coordinates", "").split())
                required = {
                    single[name].attrs[key]
                    for key in (
                        "s2s_sample_present_coord",
                        "s2s_window_start_coord",
                        "s2s_window_end_coord",
                    )
                }
                assert required <= declared
                assert (
                    not {"number", "surface", "meanSea", "heightAboveGround"} & declared
                )


@pytest.mark.parametrize("fmt,suffix", [("zarr", ".zarr"), ("netcdf", ".nc")])
def test_genuine_cfgrib_single_field_keeps_window_coordinates(tmp_path, fmt, suffix):
    from climate_restore.processor import process_init

    path = Path("../docs/analysis/s2s-pilot-20261104/raw/control_surface_24h.grib2")
    if not path.exists():
        pytest.skip("local genuine S2S surface fixture absent")
    out = tmp_path / "s2s-ecmwf" / f"20261029_00z_s2s-ecmwf{suffix}"
    process_init(
        [path],
        adapter=S2SAdapter(),
        bbox=(109, 112, 29, 32),
        out_path=out,
        workers=1,
        output_format=fmt,
    )
    assert_single_field_coordinates(out, fmt, ["msl", "ssrd_accum", "strd_accum"])


def test_genuine_mixed_members_single_field_keeps_coordinates(tmp_path):
    from types import SimpleNamespace
    from climate_restore.processor import process_init

    paths = [
        Path("../docs/analysis/s2s-pilot-20261104/raw") / name
        for name in ("control_surface_24h.grib2", "perturbed_radiation.grib2")
    ]
    if not all(p.exists() for p in paths):
        pytest.skip("local genuine S2S control/perturbed radiation fixtures absent")
    adapter = S2SAdapter("s2s-ecmwf-ensemble")
    adapter.configure_manifest(
        SimpleNamespace(
            kind="s2s",
            init_time="2026-10-29T00:00:00+00:00",
            date="20261029",
            cycle=0,
            source_metadata={"forecast_type": "ensemble"},
            files=[
                SimpleNamespace(path=path, request={"forecast_type": kind})
                for path, kind in zip(paths, ("control_forecast", "perturbed_forecast"))
            ],
        )
    )
    out = tmp_path / "s2s-ecmwf-ensemble" / "20261029_00z_s2s-ecmwf-ensemble.zarr"
    process_init(
        paths,
        adapter=adapter,
        bbox=(109, 112, 29, 32),
        out_path=out,
        workers=1,
        output_format="zarr",
    )
    assert_single_field_coordinates(out, "zarr", ["msl", "ssrd_accum", "strd_accum"])
    with xr.open_zarr(
        out, consolidated=True, drop_variables=["msl", "ssrd_accum"]
    ) as ds:
        assert ds.sizes["member"] == 101
        assert "s2s_present__strd_accum" in ds.coords
        assert not ds.s2s_present__strd_accum.sel(member=1).any()
