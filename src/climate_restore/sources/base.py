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


def _vars_by_step_type(ds: xr.Dataset) -> dict[str, list[str]]:
    """Group ``data_vars`` by their per-variable ``GRIB_stepType`` attribute.

    cfgrib happily bundles variables sharing ``(typeOfLevel, level)`` into a
    single hypercube even when their stepTypes differ — IFS HRES, for
    example, ships ``t2m`` (instant) together with ``mx2t3`` (max) and
    ``mn2t3`` (min) at 2 m. The rules expect one stepType per dataset, so
    we split first and apply the rule to each homogeneous subset.
    """
    groups: dict[str, list[str]] = {}
    for v in ds.data_vars:
        st = str(ds[v].attrs.get("GRIB_stepType") or "instant")
        groups.setdefault(st, []).append(str(v))
    return groups


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

    def __init__(self, name: str | None = None) -> None:
        # When an adapter is registered under multiple aliases (e.g.
        # ``ifs-hres`` and ``aifs-single`` both routing to AifsAdapter),
        # the alias used to instantiate it overrides the class-level
        # ``name`` so logs and progress bars reflect the manifest source.
        if name is not None:
            self.name = name

    def rules(self) -> dict[str, Rule]:
        return dict(self.RULES)

    def default_rule(self) -> Rule:
        return self.DEFAULT_RULE

    def crop_bbox(self, ds: xr.Dataset, bbox: BBox) -> xr.Dataset:
        return _generic_crop_bbox(ds, bbox)

    def rename_hypercube(self, ds: xr.Dataset) -> list[xr.Dataset]:
        tol = _type_of_level(ds)
        rule = self.rules().get(tol)
        if rule is None:
            _log.warning(
                "unhandled_type_of_level",
                source=self.name,
                type_of_level=tol,
                vars=list(map(str, ds.data_vars)),
            )
            rule = self.default_rule()
        groups = _vars_by_step_type(ds)
        out: list[xr.Dataset] = []
        for stype, names in groups.items():
            sub = ds[names] if len(groups) > 1 else ds
            out.extend(rule.apply(sub, type_of_level=tol, step_type=stype))
        return out
