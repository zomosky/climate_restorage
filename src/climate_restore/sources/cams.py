"""CAMS global: raw cumulative energies and last-three-hour gust semantics.

ADS can label integrated energy as instant. This adapter changes metadata only;
raw floating-point values and source encoding remain available for audit.
"""
from __future__ import annotations

import numpy as np

from climate_restore.sources.aifs import AifsAdapter
from climate_restore.sources.registry import register

ENERGY = frozenset({169, 228129, 228021, 228022, 47, 176, 175, 177, 179})
PRECIP = frozenset({228, 143})
THREE_HOURLY = frozenset({228028, 131, 132, 130, 133, 157, 129})


@register("ec_cams")
class CamsAdapter(AifsAdapter):
    name = "ec_cams"
    description = "CAMS global ADS 0.4° atmospheric composition and weather forecast"

    def rename_hypercube(self, ds):
        normalized = ds.copy(deep=False)
        for name in ds.data_vars:
            a = ds[name]
            p = int(a.attrs.get("GRIB_paramId", -1))
            if a.attrs.get("GRIB_dataType") != "fc":
                raise ValueError("CAMS mapping is forecast-only")
            attrs = {**a.attrs, "cams_native_interval_hours": 3 if p in THREE_HOURLY else 1,
                     "cams_native_horizon_hours": 120, "cams_source_values_changed": 0}
            if p in ENERGY | PRECIP | {228028}:
                units, source_type = a.attrs.get("units"), a.attrs.get("GRIB_stepType")
                if p in ENERGY | PRECIP:
                    allowed = ("J m**-2", "J m-2") if p in ENERGY else ("m", "kg m**-2", "kg m-2")
                    if units not in allowed or source_type not in ("instant", "accum"):
                        raise ValueError(f"Unverified CAMS cumulative units/encoding: {p}")
                    zero = a.where(a.step == np.timedelta64(0, "h"), drop=True)
                    if zero.size and not np.all(zero.values == 0):
                        raise ValueError(f"Nonzero CAMS integrated h0: {p}")
                    kind = "accum"
                    attrs.update(cams_accumulation_origin_hours=0, cams_semantics="forecast-integrated since h0")
                else:
                    if units not in ("m s**-1", "m s-1") or source_type not in ("instant", "max"):
                        raise ValueError("Unverified CAMS gust units/encoding")
                    kind = "max"
                    attrs.update(cams_statistical_window_hours=3, cams_semantics="maximum in last three hours")
                attrs.update(cams_original_GRIB_stepType=source_type, GRIB_stepType=kind)
            normalized[name] = a.copy(deep=False)
            normalized[name].attrs = attrs
        return super().rename_hypercube(normalized)
