"""Golden test for the GFS adapter against a real GRIB fixture.

Opens ``f001.subset.grib2`` from the bundled download output, runs every
cfgrib hypercube through :class:`GfsAdapter.rename_hypercube` and asserts
that the union of produced variable names matches the pinned set below.

Skipped when the fixture is absent so the suite still passes on machines
without the download checkout.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from climate_restore.sources import get_source

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "download"
    / "output"
    / "gfs-0p25"
    / "20260501"
    / "12z"
    / "f001.subset.grib2"
)

EXPECTED_VARIABLES: set[str] = {
    # 2 m / 10 m surface near-surface (shortNames already encode height)
    "d2m", "r2", "sh2", "t2m", "u10", "v10",
    # 80 m / 100 m winds, temps, pressure, humidity (split + suffixed)
    "pres80m", "q80m",
    "t80m", "u80m", "v80m",
    "t100m", "u100m", "v100m",
    # Pressure-level fields
    "gh", "r", "t", "u", "v",
    # Surface single-layer
    "prmsl", "pwat",
    # Surface instant / time-aggregated (stepType suffix applied per-var)
    "prate", "prate_avg",
    "sp", "t_sfc", "tp_accum",
    "sdlwrf_avg", "sdswrf_avg", "sulwrf_avg", "suswrf_avg",
    # Cloud layers + atmospheric total cloud cover (instant + avg)
    "hcc", "hcc_avg",
    "mcc", "mcc_avg",
    "lcc", "lcc_avg",
    "tcc", "tcc_avg",
}


@pytest.fixture(scope="module")
def renamed_var_names() -> set[str]:
    if not FIXTURE.exists():
        pytest.skip(f"GRIB fixture not present: {FIXTURE}")
    cfgrib = pytest.importorskip("cfgrib")
    adapter = get_source("gfs-0p25")()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        datasets = cfgrib.open_datasets(
            str(FIXTURE), backend_kwargs={"indexpath": ""}
        )
    names: set[str] = set()
    for ds in datasets:
        for out in adapter.rename_hypercube(ds):
            names.update(map(str, out.data_vars))
    return names


def test_gfs_golden_variable_set(renamed_var_names):
    missing = EXPECTED_VARIABLES - renamed_var_names
    extra = renamed_var_names - EXPECTED_VARIABLES
    assert not missing, f"missing expected vars: {sorted(missing)}"
    assert not extra, f"unexpected extra vars: {sorted(extra)}"


def test_gfs_golden_pressure_level_carries_dim(renamed_var_names):
    # Sanity: the four pinned PL vars must all be present so downstream
    # code can rely on the ``pressure_level`` dim being created.
    for v in ("gh", "r", "t", "u", "v"):
        assert v in renamed_var_names


def test_gfs_golden_no_name_collisions_after_rename():
    # Re-run the pipeline and assert that within each rule output, no two
    # variables collide -- this is what the t -> t_sfc instant rename and
    # the avg_ -> _avg rewrite are supposed to prevent.
    if not FIXTURE.exists():
        pytest.skip(f"GRIB fixture not present: {FIXTURE}")
    cfgrib = pytest.importorskip("cfgrib")
    adapter = get_source("gfs-0p25")()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        datasets = cfgrib.open_datasets(
            str(FIXTURE), backend_kwargs={"indexpath": ""}
        )
    for ds in datasets:
        for out in adapter.rename_hypercube(ds):
            names = list(map(str, out.data_vars))
            assert len(names) == len(set(names)), f"collision within bucket: {names}"
