"""ECMWF open-data adapter (AIFS-single, IFS HRES, ...).

The ECMWF open-data family shares one ``typeOfLevel`` taxonomy and uses
ECMWF shortName conventions, which differ from NCEP/wgrib2 in a few
predictable ways:

- ``entireAtmosphere`` is the analog of GFS's ``atmosphere`` and carries
  vertically-integrated fields (``tcw``, ``tcc``, ...).
- ``mediumCloudLayer`` is spelled with an ``e`` (GFS uses ``middle``).
- Geopotential at pressure levels is ``z`` (GFS ships ``gh``); mean-sea-level
  pressure is ``msl`` (GFS ``prmsl``); skin temperature is ``skt``.
- 100 m winds ship as ``u100`` / ``v100`` (already level-encoded) rather than
  generic ``u`` / ``v`` at a 100 m level. The regex in :class:`HeightSuffixRule`
  treats trailing digits as a level marker so no extra suffix is appended.

Open-data files surveyed so far carry only ``stepType=instant``; the
``StepTypeSuffixRule`` still applies cleanly should accumulated/average
fields appear in a future stream (``oper`` vs ``scda`` vs ``enfo``).
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

__all__ = ["AifsAdapter"]


@register("ifs-hres")
@register("aifs-single")
class AifsAdapter(BaseAdapter):
    """Restore adapter for ECMWF open-data products (AIFS / IFS HRES).

    Variable naming after this adapter runs:

    - ``u10`` / ``v10`` / ``t2m`` / ``d2m`` / ``u100`` / ``v100`` kept as-is
      (shortNames already encode the height).
    - Generic ``heightAboveGround`` variables at a non-encoded height get a
      meter suffix (same convention as the GFS adapter).
    - Pressure-level vars (``z``, ``t``, ``u``, ``v``, ``q``, ...) keep their
      ECMWF shortNames and carry a ``pressure_level`` dim.
    - Cloud-layer scalar level coords (450, 800, ...) are dropped; instant
      ``hcc`` / ``mcc`` / ``lcc`` keep their names. Any ECMWF future
      ``avg_<v>`` style aggregates become ``<v>_avg`` via
      :class:`PrefixToSuffixRule`.
    - Mean-sea-level pressure ``msl`` and integrated fields ``tcw``/``tcc``
      pass through unchanged.
    """

    name: ClassVar[str] = "aifs"
    description: ClassVar[str | None] = "ECMWF AIFS / IFS HRES open-data"

    RULES: ClassVar[dict[str, Rule]] = {
        "heightAboveGround": HeightSuffixRule(unit="m"),
        "isobaricInhPa": DimRenameRule(dst_dim="pressure_level"),
        "surface": StepTypeSuffixRule(),
        "entireAtmosphere": StepTypeSuffixRule(),
        "meanSea": PassthroughRule(),
        "highCloudLayer": PrefixToSuffixRule(),
        "mediumCloudLayer": PrefixToSuffixRule(),
        "lowCloudLayer": PrefixToSuffixRule(),
        "mostUnstableParcel": PassthroughRule(),
        "nominalTop": StepTypeSuffixRule(),
    }
