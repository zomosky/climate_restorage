"""Independent 0.1-degree IFS source, sharing the established ECMWF rules."""
from __future__ import annotations

from typing import ClassVar
from pathlib import Path
import re

import cfgrib
import numpy as np
import xarray as xr

from climate_restore.sources.aifs import AifsAdapter
from climate_restore.sources.registry import register


@register("ifs-hres-0p1")
class IfsHres0p1Adapter(AifsAdapter):
    name: ClassVar[str] = "ifs-hres-0p1"
    description: ClassVar[str] = "ECMWF IFS 0.1-degree open-data"

    def open_hypercubes(self, path: Path) -> list[xr.Dataset]:
        match = re.fullmatch(r"f(\d+)\.subset\.grib2", path.name)
        if match is None:
            raise ValueError(f"IFS 0.1-degree input requires a canonical fNNN.subset.grib2 name: {path}")
        lead = np.timedelta64(int(match.group(1)), "h")
        result = []
        for dataset in cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""}):
            if "step" not in dataset.coords:
                raise ValueError(f"IFS hypercube has no forecast step: {path}")
            selected = dataset.step.values == lead
            if not np.any(selected):
                dataset.close()
                continue
            if "step" in dataset.dims:
                dataset = dataset.isel(step=np.flatnonzero(selected))
            result.append(dataset)
        # f000 contains zero-filled min/max previews labelled step=3.
        # Keep the canonical file's lead only, so ProcessPool completion
        # order can never let those previews overwrite the real f003 data.
        return result

    def crop_bbox(self, ds: xr.Dataset, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        # cfgrib constructs the decimal grid with floating-point increments:
        # 70 E can become 69.999999999994 and 55 N 55.000000000001.
        # Snap coordinates to GRIB's microdegree precision before slicing;
        # forecast values are neither interpolated nor resampled.
        ds = ds.assign_coords({name: ds[name].round(6)
                               for name in ("latitude", "longitude") if name in ds.coords})
        return super().crop_bbox(ds, bbox)
