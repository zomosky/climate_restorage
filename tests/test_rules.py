"""Unit tests for the rename primitives in :mod:`climate_restore.rules`.

Each rule is exercised on synthetic xarray datasets built in
``conftest.py`` so the tests do not depend on cfgrib or real GRIB files.
"""

from __future__ import annotations

import numpy as np

from climate_restore.rules import (
    DimRenameRule,
    HeightSuffixRule,
    PassthroughRule,
    PrefixToSuffixRule,
    StepTypeSuffixRule,
)


class TestPassthroughRule:
    def test_drops_scalar_level_coord(self, surface_instant):
        out = PassthroughRule().apply(
            surface_instant, type_of_level="surface", step_type="instant"
        )
        assert len(out) == 1
        ds = out[0]
        assert "surface" not in ds.coords
        assert set(ds.data_vars) == {"t", "sp"}

    def test_keeps_dim_level(self, hag_multi_level):
        # When the level is a real dim (not just a scalar coord), it must stay.
        out = PassthroughRule().apply(
            hag_multi_level, type_of_level="heightAboveGround", step_type="instant"
        )
        ds = out[0]
        assert "heightAboveGround" in ds.dims


class TestHeightSuffixRule:
    def test_already_encoded_names_unchanged(self, hag_scalar_named):
        out = HeightSuffixRule().apply(
            hag_scalar_named, type_of_level="heightAboveGround", step_type="instant"
        )
        ds = out[0]
        assert set(ds.data_vars) == {"u10", "v10"}
        assert "heightAboveGround" not in ds.coords

    def test_generic_names_get_suffix(self, hag_scalar_generic):
        out = HeightSuffixRule().apply(
            hag_scalar_generic, type_of_level="heightAboveGround", step_type="instant"
        )
        ds = out[0]
        assert set(ds.data_vars) == {"pres80m", "q80m"}

    def test_multi_level_splits_per_level(self, hag_multi_level):
        out = HeightSuffixRule().apply(
            hag_multi_level, type_of_level="heightAboveGround", step_type="instant"
        )
        assert len(out) == 2
        names = {tuple(sorted(map(str, ds.data_vars))) for ds in out}
        assert names == {("t80m", "u80m"), ("t100m", "u100m")}
        for ds in out:
            assert "heightAboveGround" not in ds.coords
            assert "heightAboveGround" not in ds.dims

    def test_custom_unit(self):
        import xarray as xr

        ds = xr.Dataset(
            {"foo": (("y",), np.zeros(2))},
            coords={"y": [0, 1], "depthBelowSea": 500.0},
        )
        out = HeightSuffixRule(unit="hPa").apply(
            ds, type_of_level="depthBelowSea", step_type="instant"
        )
        assert "foo500hPa" in out[0].data_vars


class TestDimRenameRule:
    def test_expands_scalar_to_dim_and_renames(self, pressure_scalar):
        out = DimRenameRule(dst_dim="pressure_level").apply(
            pressure_scalar, type_of_level="isobaricInhPa", step_type="instant"
        )
        ds = out[0]
        assert "pressure_level" in ds.dims
        assert ds.sizes["pressure_level"] == 1
        assert "isobaricInhPa" not in ds.dims
        assert set(ds.data_vars) == {"t", "u"}

    def test_renames_existing_dim(self):
        import xarray as xr

        ds = xr.Dataset(
            {"t": (("isobaricInhPa", "y"), np.zeros((3, 2)))},
            coords={"isobaricInhPa": [1000.0, 850.0, 500.0], "y": [0, 1]},
        )
        out = DimRenameRule(dst_dim="pressure_level").apply(
            ds, type_of_level="isobaricInhPa", step_type="instant"
        )
        assert out[0].sizes["pressure_level"] == 3


class TestStepTypeSuffixRule:
    def test_instant_passthrough(self, surface_instant):
        out = StepTypeSuffixRule().apply(
            surface_instant, type_of_level="surface", step_type="instant"
        )
        ds = out[0]
        assert set(ds.data_vars) == {"t", "sp"}
        assert "surface" not in ds.coords

    def test_non_instant_suffixes_all(self, surface_avg):
        out = StepTypeSuffixRule().apply(
            surface_avg, type_of_level="surface", step_type="avg"
        )
        ds = out[0]
        assert set(ds.data_vars) == {"prate_avg", "tp_avg"}

    def test_instant_renames_apply(self, surface_instant):
        out = StepTypeSuffixRule(instant_renames={"t": "t_sfc"}).apply(
            surface_instant, type_of_level="surface", step_type="instant"
        )
        ds = out[0]
        assert set(ds.data_vars) == {"t_sfc", "sp"}


class TestPrefixToSuffixRule:
    def test_avg_prefix_rewritten(self, cloud_layer_mixed):
        out = PrefixToSuffixRule().apply(
            cloud_layer_mixed,
            type_of_level="highCloudLayer",
            step_type="instant",
        )
        ds = out[0]
        assert set(ds.data_vars) == {"hcc", "hcc_avg"}

    def test_no_match_keeps_name(self, surface_instant):
        out = PrefixToSuffixRule().apply(
            surface_instant, type_of_level="surface", step_type="instant"
        )
        assert set(out[0].data_vars) == {"t", "sp"}

    def test_custom_mapping(self):
        import xarray as xr

        ds = xr.Dataset({"max_gust": (("y",), np.zeros(2))}, coords={"y": [0, 1]})
        out = PrefixToSuffixRule(mapping={"max_": "_max"}).apply(
            ds, type_of_level="surface", step_type="max"
        )
        assert "gust_max" in out[0].data_vars
