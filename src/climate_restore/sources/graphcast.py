"""NOAA NWS GraphCast (aigfs) adapter.

GraphCast ships its forecast as **two** GRIB streams per ``(date, cycle, step)``
on the ``noaa-nws-graphcastgfs-pds`` S3 bucket, each driven by its own
download manifest in this project's upstream:

- ``graphcast-sfc`` – ``aigfs.t{cc}z.sfc.fXXX.grib2``
  Surface / near-surface variables: ``u10`` / ``v10`` (10 m winds),
  ``t2m`` (2 m temperature), ``prmsl`` (mean-sea-level pressure) and
  ``tp`` (accumulated total precipitation).
- ``graphcast-pres`` – ``aigfs.t{cc}z.pres.fXXX.grib2``
  Pressure-level fields. Unlike GFS, each variable family covers its own
  level subset: ``t``/``u``/``v``/``gh`` at [1000, 850, 500] hPa, ``q`` at
  [1000, 850], and ``w`` at [925, 850, 700, 500]. The processor's outer
  merge produces a single ``pressure_level`` axis as the union of these
  sets, with ``NaN`` where a variable is not defined.

Each manifest is processed independently to its own ``.nc`` file (one for
``-sfc``, one for ``-pres``). Downstream code that wants the full state
can ``xr.open_mfdataset`` both files for the same init.
"""

from __future__ import annotations

from typing import ClassVar

from climate_restore.rules import (
    DimRenameRule,
    HeightSuffixRule,
    PassthroughRule,
    Rule,
    StepTypeSuffixRule,
)
from climate_restore.sources.base import BaseAdapter
from climate_restore.sources.registry import register

__all__ = ["GraphCastAdapter"]


@register("graphcast")
@register("graphcast-pres")
@register("graphcast-sfc")
class GraphCastAdapter(BaseAdapter):
    """Restore adapter for the GraphCastGFS ``-sfc`` / ``-pres`` products.

    Variable naming after this adapter runs:

    - ``u10`` / ``v10`` / ``t2m`` keep their original shortNames (already
      level-encoded).
    - Pressure-level variables (``t`` / ``u`` / ``v`` / ``gh`` / ``q`` /
      ``w``) keep their names and carry a ``pressure_level`` dim.
    - ``meanSea`` ``prmsl`` and any surface instant fields pass through
      without a level coord.
    - Surface ``tp`` (``stepType=accum``) becomes ``tp_accum``.

    GraphCast does not publish cloud-layer or atmospheric-column hypercubes,
    so the rule table is intentionally a strict subset of the GFS adapter.
    """

    name: ClassVar[str] = "graphcast"
    description: ClassVar[str | None] = "NOAA NWS GraphCastGFS (aigfs) sfc + pres"

    RULES: ClassVar[dict[str, Rule]] = {
        "heightAboveGround": HeightSuffixRule(unit="m"),
        "isobaricInhPa": DimRenameRule(dst_dim="pressure_level"),
        "meanSea": PassthroughRule(),
        "surface": StepTypeSuffixRule(),
    }
