"""Crop downloaded GRIB subsets and time-concatenate one (date, cycle) into NetCDF.

Strategy
--------
For each ``f###.subset.grib2`` file in a manifest:

1. ``cfgrib.open_datasets`` splits the heterogeneous GRIB into one
   ``xr.Dataset`` per coherent hypercube.
2. The source adapter (:mod:`climate_restore.sources`) crops each
   hypercube to the configured bbox and applies its rename rules,
   yielding one or more datasets with friendly variable names.
3. Renamed hypercubes are bucketed by their renamed variable signature,
   concatenated across step files, then merged into a single flat
   ``xr.Dataset`` written to one NetCDF4 file (no groups).

The processor itself contains no source-specific logic; everything that
varies between GFS / AIFS / HRRR / ... lives in the adapter.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

import cfgrib
import xarray as xr

from .logging_setup import get_logger
from .sources.base import BaseAdapter

DEFAULT_BBOX: tuple[float, float, float, float] = (70.0, 140.0, 15.0, 55.0)

_log = get_logger(__name__)


def _open_hypercubes(grib_path: Path) -> list[xr.Dataset]:
    return cfgrib.open_datasets(
        str(grib_path),
        backend_kwargs={"indexpath": ""},
    )


def _ensure_step_dim(ds: xr.Dataset) -> xr.Dataset:
    if "step" in ds.dims:
        return ds
    if "step" in ds.coords:
        return ds.expand_dims("step")
    return ds


def process_init(
    grib_paths: Iterable[Path],
    *,
    adapter: BaseAdapter,
    bbox: tuple[float, float, float, float],
    out_path: Path,
) -> list[str]:
    """Process one (date, cycle): crop + time-concat all step files into ``out_path``.

    Returns the sorted list of data-variable names in the written NetCDF.
    """
    grib_paths = list(grib_paths)
    if not grib_paths:
        raise ValueError("no GRIB files supplied")

    # Bucket renamed hypercubes by (sorted data_vars, sorted non-step dims).
    buckets: "OrderedDict[tuple[tuple[str, ...], tuple[str, ...]], list[xr.Dataset]]" = OrderedDict()
    for p in grib_paths:
        try:
            hypercubes = _open_hypercubes(p)
        except Exception as exc:  # pragma: no cover - surfaced to caller
            raise RuntimeError(f"failed to open {p}: {exc}") from exc
        _log.info("opened_grib", path=str(p), hypercubes=len(hypercubes))
        for ds in hypercubes:
            cropped = adapter.crop_bbox(ds, bbox)
            if cropped.sizes.get("latitude", 0) == 0 or cropped.sizes.get("longitude", 0) == 0:
                continue
            cropped = _ensure_step_dim(cropped)
            for renamed in adapter.rename_hypercube(cropped):
                vars_sig = tuple(sorted(map(str, renamed.data_vars)))
                dim_sig = tuple(sorted(d for d in renamed.dims if d != "step"))
                buckets.setdefault((vars_sig, dim_sig), []).append(renamed)

    bucket_dss: list[xr.Dataset] = []
    for (vars_sig, _dim_sig), ds_list in buckets.items():
        merged = xr.concat(
            ds_list, dim="step", coords="minimal", compat="override", join="outer"
        ).sortby("step")
        # Drop per-bucket valid_time; rebuilt on the merged step axis below
        # (concat with coords='minimal' may already have dropped it).
        merged = merged.drop_vars("valid_time", errors="ignore")
        bucket_dss.append(merged)
        _log.info(
            "concat_bucket",
            vars=list(vars_sig),
            steps=int(merged.sizes.get("step", 0)),
            lat=int(merged.sizes.get("latitude", 0)),
            lon=int(merged.sizes.get("longitude", 0)),
        )

    final = xr.merge(bucket_dss, compat="no_conflicts", join="outer")
    if "time" in final.coords and "step" in final.coords:
        final = final.assign_coords(valid_time=final["time"] + final["step"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    final.to_netcdf(out_path, engine="netcdf4")

    variables = sorted(map(str, final.data_vars))
    _log.info(
        "processed_init",
        out_path=str(out_path),
        source=adapter.name,
        variables=len(variables),
        steps=int(final.sizes.get("step", 0)),
    )
    return variables
