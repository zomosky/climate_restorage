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

import numpy as np
import xarray as xr

from climate_restore.rules import (
    DimRenameRule,
    HeightSuffixRule,
    PassthroughRule,
    Rule,
    StepTypeSuffixRule,
)
from climate_restore.sources.base import BaseAdapter, BBox
from climate_restore.sources.registry import register

__all__ = ["GraphCastAdapter"]


@register("graphcast")
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
        # The early EAGLE-SOLO archive encodes APCP with paramId 0, so eccodes
        # decodes it as shortName ``unknown`` -> the suffix rule would name it
        # ``unknown_accum``; the newer AIGFS product decodes the same field as
        # ``tp`` -> ``tp_accum``. GraphCast's only surface-accumulated field is
        # precipitation, so mapping ``unknown`` -> ``tp`` here makes precip land
        # as ``tp_accum`` across BOTH eras (a no-op when it already decodes as
        # ``tp``), keeping the archive uniform.
        "surface": StepTypeSuffixRule(renames={"unknown": "tp"}),
    }

    #: native grid resolution (deg); coords are snapped to this before cropping
    GRID_RES: ClassVar[float] = 0.25

    def crop_bbox(self, ds: xr.Dataset, bbox: BBox) -> xr.Dataset:
        """Snap lon/lat onto the exact 0.25° grid, then crop.

        The two GraphCast product eras encode longitude slightly differently:
        the older EAGLE-SOLO *merged* archive carries a tiny floating-point
        drift (global max ``359.750016`` instead of ``359.75``), so the 140°E
        column lands at ~``140.00001`` and slips *outside* a ``slice(70, 140)``
        crop — yielding a 280-column China grid, one short of the newer AIGFS
        product's clean ``359.75`` (281 columns). Rounding each coord onto the
        native ``GRID_RES`` grid before cropping makes **both** eras produce the
        identical 281×161 China grid, so old- and new-format inits stitch into
        one uniform series.
        """
        res = self.GRID_RES
        for c in ("longitude", "latitude"):
            if c in ds.coords:
                snapped = np.round(np.asarray(ds[c].values, dtype="float64") / res) * res
                ds = ds.assign_coords({c: snapped})
        return super().crop_bbox(ds, bbox)
