"""NOAA GFS 0.25° atmos adapter (wgrib2 idx family).

GraphCastGFS shares the same wgrib2 idx protocol but only publishes a
subset of layers, so it lives in :mod:`climate_restore.sources.graphcast`
with its own slimmer rule table rather than reusing this one.
"""

from __future__ import annotations

from typing import ClassVar

from climate_restore.rules import (
    DimRenameRule,
    HeightSuffixRule,
    PassthroughRule,
    PrefixToSuffixRule,
    Rule,
    StepTypeSuffixRule,
)
from climate_restore.sources.base import BaseAdapter
from climate_restore.sources.registry import register

__all__ = ["GfsAdapter"]


@register("gfs-0p25")
class GfsAdapter(BaseAdapter):
    """Restore adapter for NOAA GFS-family products.

    Variable naming after this adapter runs:

    - ``u10``, ``v10``, ``t2m``, ``d2m``, ``sh2``, ``r2`` kept as-is
      (GRIB shortNames already encode the height).
    - ``pres``/``q`` at 80 m become ``pres80m``/``q80m``; ``t``/``u``/``v``
      at 80 m and 100 m split into per-level vars (``t80m``, ``u100m``...).
    - Pressure-level vars (``t``, ``u``, ``v``, ``gh``, ``r``) keep their
      names and carry a ``pressure_level`` dim.
    - Surface instant ``t`` → ``t_sfc``; non-instant surface vars get a
      stepType suffix (``prate`` avg → ``prate_avg``).
    - Cloud layers: cfgrib's ``avg_hcc`` becomes ``hcc_avg``; same for
      mcc/lcc. Atmosphere ``tcc`` avg variant becomes ``tcc_avg``.
    """

    name: ClassVar[str] = "gfs"
    description: ClassVar[str | None] = "NOAA GFS / GraphCastGFS (wgrib2 idx family)"

    RULES: ClassVar[dict[str, Rule]] = {
        "heightAboveGround": HeightSuffixRule(unit="m"),
        "isobaricInhPa": DimRenameRule(dst_dim="pressure_level"),
        "surface": StepTypeSuffixRule(instant_renames={"t": "t_sfc"}),
        "atmosphere": StepTypeSuffixRule(),
        "atmosphereSingleLayer": PassthroughRule(),
        "meanSea": PassthroughRule(),
        "highCloudLayer": PrefixToSuffixRule(),
        "middleCloudLayer": PrefixToSuffixRule(),
        "lowCloudLayer": PrefixToSuffixRule(),
    }
