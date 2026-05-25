"""Hypercube rename rules.

Each rule transforms a single cfgrib hypercube (an ``xr.Dataset`` sharing one
``typeOfLevel`` and matching variable set) into one or more renamed datasets
ready for cross-step concatenation. Source adapters compose rules in a
``{type_of_level: Rule}`` table; see :mod:`climate_restore.sources.base`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

import xarray as xr


class Rule(Protocol):
    """Transform one cfgrib hypercube into one or more renamed datasets."""

    def apply(
        self, ds: xr.Dataset, *, type_of_level: str, step_type: str
    ) -> list[xr.Dataset]: ...


def _drop_level_coord(ds: xr.Dataset, type_of_level: str) -> xr.Dataset:
    """Drop the scalar level coordinate named after ``type_of_level`` if present."""
    if type_of_level in ds.coords and type_of_level not in ds.dims:
        return ds.drop_vars(type_of_level, errors="ignore")
    return ds


@dataclass(frozen=True)
class PassthroughRule:
    """Drop the scalar level coord; leave variable names untouched."""

    def apply(self, ds, *, type_of_level, step_type):
        return [_drop_level_coord(ds, type_of_level)]


@dataclass(frozen=True)
class HeightSuffixRule:
    """For ``heightAboveGround``-style levels: append a meter suffix.

    - If ``type_of_level`` is also a *dim* on the dataset (cfgrib bundled
      several levels together, e.g. [80, 100]), the dataset is split per
      level and each slice goes through the per-level rename below.
    - Variables whose name already matches ``keep_pattern`` (e.g. ``u10``,
      ``t2m``, ``sh2``) keep their original name. Everything else gets
      ``<var><level><unit>`` appended (e.g. ``pres`` @80m -> ``pres80m``).
    - Non-instant ``step_type`` then appends ``_<step_type>`` to every
      output name. This disambiguates e.g. IFS ``fg10`` (max wind gust at
      10 m) from ``u10`` (instant wind) and lets the same shortName ship
      under multiple stepTypes without colliding (``mx2t3`` -> ``mx2t3_max``).
    """

    unit: str = "m"
    keep_pattern: str = r"\d+[a-z]*$"

    def apply(self, ds, *, type_of_level, step_type):
        if type_of_level in ds.dims:
            outs: list[xr.Dataset] = []
            for lvl in ds[type_of_level].values:
                sub = ds.sel({type_of_level: lvl}).drop_vars(
                    type_of_level, errors="ignore"
                )
                outs.append(self._rename_scalar(sub, float(lvl)))
        else:
            lvl = float(ds[type_of_level].values) if type_of_level in ds.coords else 0.0
            ds = _drop_level_coord(ds, type_of_level)
            outs = [self._rename_scalar(ds, lvl)]
        if step_type != "instant":
            outs = [
                s.rename({v: f"{v}_{step_type}" for v in s.data_vars}) for s in outs
            ]
        return outs

    def _rename_scalar(self, ds: xr.Dataset, level: float) -> xr.Dataset:
        suffix = f"{int(level)}{self.unit}"
        keep = re.compile(self.keep_pattern)
        rename = {
            v: f"{v}{suffix}" for v in ds.data_vars if not keep.search(str(v))
        }
        return ds.rename(rename) if rename else ds


@dataclass(frozen=True)
class DimRenameRule:
    """Rename a level dim/coord and keep variable names intact.

    Used for pressure levels and similar vertical axes: expands a scalar
    coord to a 1-D dim so the schema stays stable when more levels are
    added later.
    """

    dst_dim: str

    def apply(self, ds, *, type_of_level, step_type):
        if type_of_level not in ds.dims and type_of_level in ds.coords:
            ds = ds.expand_dims(type_of_level)
        if type_of_level in ds.dims:
            ds = ds.rename({type_of_level: self.dst_dim})
        return [ds]


@dataclass(frozen=True)
class StepTypeSuffixRule:
    """Drop the scalar level coord; non-instant vars get ``_<stepType>``.

    Optional ``instant_renames`` lets a source rename specific instant vars
    (e.g. ``surface`` rule maps ``t`` to ``t_sfc`` to avoid collisions).
    """

    instant_renames: dict[str, str] = field(default_factory=dict)

    def apply(self, ds, *, type_of_level, step_type):
        ds = _drop_level_coord(ds, type_of_level)
        if step_type != "instant":
            ds = ds.rename({v: f"{v}_{step_type}" for v in ds.data_vars})
        elif self.instant_renames:
            rename = {k: v for k, v in self.instant_renames.items() if k in ds.data_vars}
            if rename:
                ds = ds.rename(rename)
        return [ds]


@dataclass(frozen=True)
class PrefixToSuffixRule:
    """Move cfgrib's ``avg_<v>`` / ``max_<v>`` prefix to a trailing ``_<tag>``.

    cfgrib sometimes bundles instant and time-aggregated variants of the
    same field into one hypercube (e.g. cloud layers carry both ``hcc`` and
    ``avg_hcc``). This rule rewrites the prefix so the schema matches the
    ``<var>_<stepType>`` convention used elsewhere.
    """

    mapping: dict[str, str] = field(
        default_factory=lambda: {"avg_": "_avg", "max_": "_max", "min_": "_min"}
    )

    def apply(self, ds, *, type_of_level, step_type):
        ds = _drop_level_coord(ds, type_of_level)
        rename: dict[str, str] = {}
        for v in ds.data_vars:
            for prefix, suffix in self.mapping.items():
                if str(v).startswith(prefix):
                    rename[v] = f"{str(v)[len(prefix):]}{suffix}"
                    break
        return [ds.rename(rename) if rename else ds]
