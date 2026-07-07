"""Shared fixtures for restorage tests.

Builds tiny in-memory xarray datasets that mimic cfgrib hypercube layouts
(``GRIB_typeOfLevel`` / ``GRIB_stepType`` attrs, scalar level coords,
multi-level dims, lat/lon grids) without touching real GRIB files.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


def _attrs(type_of_level: str, step_type: str = "instant") -> dict[str, str]:
    return {"GRIB_typeOfLevel": type_of_level, "GRIB_stepType": step_type}


def _grid(nlat: int = 2, nlon: int = 3) -> dict[str, np.ndarray]:
    return {
        "latitude": np.linspace(15.0, 55.0, nlat),
        "longitude": np.linspace(70.0, 140.0, nlon),
    }


@pytest.fixture
def hag_scalar_named():
    """heightAboveGround at 10 m with names that already encode the level."""
    g = _grid()
    data = np.zeros((2, 3))
    return xr.Dataset(
        {
            "u10": (("latitude", "longitude"), data, _attrs("heightAboveGround")),
            "v10": (("latitude", "longitude"), data, _attrs("heightAboveGround")),
        },
        coords={**g, "heightAboveGround": 10.0},
    )


@pytest.fixture
def hag_scalar_generic():
    """heightAboveGround at 80 m with generic shortNames that need suffixing."""
    g = _grid()
    data = np.zeros((2, 3))
    return xr.Dataset(
        {
            "pres": (("latitude", "longitude"), data, _attrs("heightAboveGround")),
            "q": (("latitude", "longitude"), data, _attrs("heightAboveGround")),
        },
        coords={**g, "heightAboveGround": 80.0},
    )


@pytest.fixture
def hag_multi_level():
    """heightAboveGround as a dim covering [80, 100] for split-then-suffix."""
    g = _grid()
    data = np.zeros((2, 2, 3))
    return xr.Dataset(
        {
            "t": (("heightAboveGround", "latitude", "longitude"), data, _attrs("heightAboveGround")),
            "u": (("heightAboveGround", "latitude", "longitude"), data, _attrs("heightAboveGround")),
        },
        coords={**g, "heightAboveGround": np.array([80.0, 100.0])},
    )


@pytest.fixture
def pressure_scalar():
    """Pressure-level hypercube as a scalar coord (single level)."""
    g = _grid()
    data = np.zeros((2, 3))
    return xr.Dataset(
        {
            "t": (("latitude", "longitude"), data, _attrs("isobaricInhPa")),
            "u": (("latitude", "longitude"), data, _attrs("isobaricInhPa")),
        },
        coords={**g, "isobaricInhPa": 850.0},
    )


@pytest.fixture
def surface_instant():
    g = _grid()
    data = np.zeros((2, 3))
    return xr.Dataset(
        {
            "t": (("latitude", "longitude"), data, _attrs("surface", "instant")),
            "sp": (("latitude", "longitude"), data, _attrs("surface", "instant")),
        },
        coords={**g, "surface": 0.0},
    )


@pytest.fixture
def surface_avg():
    g = _grid()
    data = np.zeros((2, 3))
    return xr.Dataset(
        {
            "prate": (("latitude", "longitude"), data, _attrs("surface", "avg")),
            "tp": (("latitude", "longitude"), data, _attrs("surface", "avg")),
        },
        coords={**g, "surface": 0.0},
    )


@pytest.fixture
def tcc_conv_instant():
    """convectiveCloudLayer ``tcc`` (instant) — same shortName as atmosphere tcc."""
    g = _grid()
    data = np.zeros((2, 3))
    return xr.Dataset(
        {"tcc": (("latitude", "longitude"), data, _attrs("convectiveCloudLayer", "instant"))},
        coords={**g, "convectiveCloudLayer": 0.0},
    )


@pytest.fixture
def tcc_bl_avg():
    """boundaryLayerCloudLayer ``tcc`` (avg) — same shortName as atmosphere tcc."""
    g = _grid()
    data = np.zeros((2, 3))
    return xr.Dataset(
        {"tcc": (("latitude", "longitude"), data, _attrs("boundaryLayerCloudLayer", "avg"))},
        coords={**g, "boundaryLayerCloudLayer": 0.0},
    )


@pytest.fixture
def cloud_layer_mixed():
    """A cloud-layer hypercube that bundles ``hcc`` and ``avg_hcc``."""
    g = _grid()
    data = np.zeros((2, 3))
    return xr.Dataset(
        {
            "hcc": (("latitude", "longitude"), data, _attrs("highCloudLayer", "instant")),
            "avg_hcc": (("latitude", "longitude"), data, _attrs("highCloudLayer", "avg")),
        },
        coords={**g, "highCloudLayer": 0.0},
    )
