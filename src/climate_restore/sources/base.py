"""Source adapter contract for the restorage pipeline.

Each upstream data source (NOAA GFS, ECMWF AIFS/IFS, NOAA HRRR, ...) lives
in its own module under :mod:`climate_restore.sources` and exposes a class
that satisfies the :class:`SourceAdapter` protocol. The processor in
:mod:`climate_restore.processor` only ever calls protocol methods, so
adding a new source means writing one file plus registering it; no edits
to :mod:`processor` are required.

An adapter mainly declares a ``RULES`` table mapping ``typeOfLevel`` to a
``Rule`` from :mod:`climate_restore.rules`. Sources with a non-standard
spatial layout (regional projections, alternate lon conventions) can also
override :meth:`crop_bbox`.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import xarray as xr

from climate_restore.logging_setup import get_logger
from climate_restore.rules import PassthroughRule, Rule

__all__ = ["BaseAdapter", "SourceAdapter"]

BBox = tuple[float, float, float, float]

_log = get_logger(__name__)


@runtime_checkable
class SourceAdapter(Protocol):
    """Protocol every restore adapter implements."""

    name: str
    description: str | None

    def rules(self) -> dict[str, Rule]: ...

    def default_rule(self) -> Rule: ...

    def crop_bbox(self, ds: xr.Dataset, bbox: BBox) -> xr.Dataset: ...

    def rename_hypercube(self, ds: xr.Dataset) -> list[xr.Dataset]: ...


def _type_of_level(ds: xr.Dataset) -> str:
    for v in ds.data_vars:
        tol = ds[v].attrs.get("GRIB_typeOfLevel")
        if tol:
            return str(tol)
    return "unknown"


def _step_type(ds: xr.Dataset) -> str:
    for v in ds.data_vars:
        st = ds[v].attrs.get("GRIB_stepType")
        if st:
            return str(st)
    return "instant"


def _generic_crop_bbox(ds: xr.Dataset, bbox: BBox) -> xr.Dataset:
    """Crop on regular ``latitude``/``longitude`` coords.

    Handles 0..360 longitude conventions and descending latitude axes.
    """
    if "longitude" not in ds.coords or "latitude" not in ds.coords:
        return ds
    west, east, south, north = bbox
    lon = ds["longitude"]
    if float(lon.max()) > 180.0:
        ds = ds.assign_coords(longitude=(((lon + 180) % 360) - 180)).sortby("longitude")
    ds = ds.sortby("latitude")
    return ds.sel(longitude=slice(west, east), latitude=slice(south, north))


class BaseAdapter:
    """Default implementation driven by a class-level ``RULES`` table.

    Subclasses override ``name``, ``RULES`` and (optionally)
    ``DEFAULT_RULE`` / ``crop_bbox``. The processor calls
    :meth:`rename_hypercube` once per cfgrib hypercube; the default
    implementation looks the type up in the rules table and logs a warning
    when no rule matches so unhandled GRIB layers do not silently produce
    colliding variable names.
    """

    name: ClassVar[str] = "base"
    description: ClassVar[str | None] = None
    RULES: ClassVar[dict[str, Rule]] = {}
    DEFAULT_RULE: ClassVar[Rule] = PassthroughRule()

    def rules(self) -> dict[str, Rule]:
        return dict(self.RULES)

    def default_rule(self) -> Rule:
        return self.DEFAULT_RULE

    def crop_bbox(self, ds: xr.Dataset, bbox: BBox) -> xr.Dataset:
        return _generic_crop_bbox(ds, bbox)

    def rename_hypercube(self, ds: xr.Dataset) -> list[xr.Dataset]:
        tol = _type_of_level(ds)
        stype = _step_type(ds)
        rule = self.rules().get(tol)
        if rule is None:
            _log.warning(
                "unhandled_type_of_level",
                source=self.name,
                type_of_level=tol,
                step_type=stype,
                vars=list(map(str, ds.data_vars)),
            )
            rule = self.default_rule()
        return rule.apply(ds, type_of_level=tol, step_type=stype)
