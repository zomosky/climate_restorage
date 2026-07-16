"""DWD ICON global adapter — icosahedral → regular lat/lon.

Unlike GFS / IFS (already on a regular lat/lon grid, so restore just *crops*),
ICON global GRIB is on the **unstructured icosahedral grid**: a 1-D ``values``
dimension with no lat/lon in the message. This adapter overrides
:meth:`crop_bbox` to **regrid** the cells onto a regular China grid via a cached
nearest-neighbour remap (:mod:`climate_restore.sources.icon_grid`); the rename
rules then run exactly as for the other sources.

Target grid resolution is ``RES`` degrees (0.125° by default — finer than the
0.25° global models, keeping some of ICON's ~13 km detail; every target point
gets its nearest cell, so there are no gaps).
"""

from __future__ import annotations

from typing import ClassVar

import xarray as xr

from climate_restore.rules import (
    HeightSuffixRule,
    PassthroughRule,
    Rule,
    StepTypeSuffixRule,
)
from climate_restore.sources.base import BaseAdapter, BBox
from climate_restore.sources.icon_grid import build_remap, remap_values
from climate_restore.sources.registry import register

__all__ = ["IconAdapter"]


@register("dwd-icon-operation")
class IconAdapter(BaseAdapter):
    """Restore adapter for DWD ICON global (icosahedral → regridded lat/lon)."""

    name: ClassVar[str] = "dwd-icon-operation"
    description: ClassVar[str | None] = "DWD ICON global (icosahedral, regridded)"

    # Target regular-grid spacing (degrees) for the icosahedral → lat/lon remap.
    RES: ClassVar[float] = 0.125

    RULES: ClassVar[dict[str, Rule]] = {
        "heightAboveGround": HeightSuffixRule(unit="m"),  # u10/v10/fg10, t2m/d2m/r2
        "isobaricLayer": PassthroughRule(),               # CLCH / CLCM
        "meanSea": PassthroughRule(),                     # prmsl
        "surface": StepTypeSuffixRule(),                  # sp/tp/T_G/TQV/CLCT/ASWDIR_S/...
        "generalVerticalLayer": PassthroughRule(),
        "generalVertical": PassthroughRule(),
        "heightAboveSea": PassthroughRule(),
    }
    # HZEROCL / CAPE_ML / CLCL come through with no typeOfLevel → "unknown".
    DEFAULT_RULE: ClassVar[Rule] = PassthroughRule()

    def crop_bbox(self, ds: xr.Dataset, bbox: BBox) -> xr.Dataset:
        """Regrid the unstructured ``values`` dim onto a regular lat/lon grid.

        Non-``values`` dims (``step``) and scalar coords (``time`` / ``step`` /
        ``valid_time`` / the level coord) are preserved; per-variable ``GRIB_*``
        attributes are carried through so :meth:`rename_hypercube` still keys off
        ``typeOfLevel`` / ``stepType`` afterwards.
        """
        if "values" not in ds.dims:
            return super().crop_bbox(ds, bbox)  # already regular — shouldn't happen

        remap = build_remap(tuple(bbox), self.RES)
        data_vars = {}
        for name, da in ds.data_vars.items():
            other = [d for d in da.dims if d != "values"]
            arr = da.transpose(*other, "values").values
            grid = remap_values(arr, remap)  # (*other, nlat, nlon)
            data_vars[str(name)] = (
                tuple(other) + ("latitude", "longitude"), grid, dict(da.attrs)
            )
        coords: dict = {
            "latitude": ("latitude", remap.lats),
            "longitude": ("longitude", remap.lons),
        }
        # Keep only step/time/valid_time, as raw (dims, values) so no sibling
        # coords tag along — importantly the scalar ``level`` coord from the
        # no-typeOfLevel fields (HZEROCL/CAPE_ML/CLCL), which otherwise rides in
        # on ``valid_time`` and conflicts at the cross-bucket merge. The rename
        # rules key off per-variable GRIB attrs (typeOfLevel / stepType), not the
        # level coord, so dropping it is safe.
        for c in ("step", "time", "valid_time"):
            if c in ds.coords and "values" not in ds.coords[c].dims:
                cc = ds.coords[c]
                coords[c] = (cc.dims, cc.values)
        return xr.Dataset(data_vars, coords=coords, attrs=dict(ds.attrs))
