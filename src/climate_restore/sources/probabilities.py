"""Decode event probabilities without confusing their underlying statistic with values.

GRIB stepType describes the event (sum, maximum, mean), while the values
are percentages. Same-parameter windows must be split before cfgrib builds
a hypercube, otherwise e.g. instantaneous and multi-day frost collide.
"""
from __future__ import annotations

from pathlib import Path
import json
import re

import cfgrib
import eccodes as ec
import numpy as np
import xarray as xr

from climate_restore.sources.base import BaseAdapter
from climate_restore.sources.registry import register

RAIN = ("tpg1", "tpg5", "tpg10", "tpg20", "tpg25", "tpg50", "tpg100")
MULTIDAY = ("2tl273", "tpl01", "tprl1", "tprg3", "tprg5")
READ_KEYS = ["lengthOfTimeRange", "productDefinitionTemplateNumber", "probabilityType",
             "lowerLimit", "upperLimit", "typeOfStatisticalProcessing"]


def event_name(param: str, statistic: str, hours: int) -> str:
    if hours == 0 or (param in RAIN or param.startswith("10fgg")) and hours == 24:
        return param
    return f"{param}_{statistic}{hours}h"


def event_threshold(param: str) -> tuple[float, str, str]:
    if m := re.fullmatch(r"tpg(\d+)", param):
        return float(m[1]), "mm", "gt"
    if m := re.fullmatch(r"10(?:fgg|spg)(\d+)", param):
        return float(m[1]), "m s-1", "gt"
    if param == "2tl273":
        return 273.15, "K", "lt"
    if param == "tpl01":
        return 0.1, "mm", "lt"
    if m := re.fullmatch(r"tpr([gl])(\d+)", param):
        return float(m[2]), "mm day-1", "gt" if m[1] == "g" else "lt"
    raise ValueError(f"unsupported probability parameter {param!r}")


def expected_samples(source: str) -> dict[str, set[int]]:
    daily = set(range(24, 361, 12))
    if source == "ifs-ens-probabilities":
        return {p: set(daily) for p in ("tpg20", "tpg50", "tpg100", "10fgg15", "10fgg25")}
    result = {p: set(daily) for p in RAIN}
    result.update({p: set(range(12, 361, 12)) for p in ("2tl273", "10spg10", "10spg15")})
    for p in MULTIDAY:
        stat = "max" if p == "2tl273" else "accum" if p == "tpl01" else "avg"
        for hours, ends in ((48, {168}), (72, {240}), (120, {240, 360})):
            result[event_name(p, stat, hours)] = ends
    return result


@register("ifs-ens-probabilities")
@register("aifs-ens-probabilities")
class ProbabilityAdapter(BaseAdapter):
    name = "ifs-ens-probabilities"
    description = "ECMWF ENS event probabilities with original statistical windows"
    historical_availability = None

    def configure_manifest(self, manifest) -> None:
        self.historical_availability = manifest.probability_availability

    def open_hypercubes(self, path: Path) -> list[xr.Dataset]:
        groups: dict[tuple[int, int, int, int], dict] = {}
        with Path(path).open("rb") as handle:
            while (gid := ec.codes_grib_new_from_file(handle)) is not None:
                try:
                    if ec.codes_get(gid, "dataType") != "ep" or ec.codes_get(gid, "units") != "%":
                        raise ValueError("probability source contains a non-probability message")
                    pid = int(ec.codes_get(gid, "paramId"))
                    template = int(ec.codes_get(gid, "productDefinitionTemplateNumber"))
                    start, end = int(ec.codes_get(gid, "startStep")), int(ec.codes_get(gid, "endStep"))
                    if int(ec.codes_get(gid, "stepUnits")) != 1 or end < start:
                        raise ValueError("probability window must use nonnegative hours")
                    width = end - start
                    statistic = -1
                    keys = {"paramId": pid, "productDefinitionTemplateNumber": template}
                    if template == 9:
                        keys["lengthOfTimeRange"] = int(ec.codes_get(gid, "lengthOfTimeRange"))
                        keys["typeOfStatisticalProcessing"] = int(ec.codes_get(gid, "typeOfStatisticalProcessing"))
                        statistic = keys["typeOfStatisticalProcessing"]
                    elif template != 5:
                        raise ValueError(f"unsupported probability template {template}")
                    groups[(pid, template, width, statistic)] = keys
                finally:
                    ec.codes_release(gid)
        output = []
        for (_, _, width, _), keys in groups.items():
            cubes = cfgrib.open_datasets(str(path), backend_kwargs={
                "indexpath": "", "filter_by_keys": keys, "read_keys": READ_KEYS})
            for ds in cubes:
                for name in ds.data_vars:
                    ds[name].attrs["probability_window_hours"] = width
                output.append(ds)
        return output

    def rename_hypercube(self, ds: xr.Dataset) -> list[xr.Dataset]:
        if "step" not in ds.dims:
            ds = ds.expand_dims("step")
        output = []
        for variable in ds.data_vars:
            da = ds[variable]
            param = str(da.attrs["GRIB_shortName"])
            statistic = str(da.attrs.get("GRIB_stepType", "instant"))
            hours = int(da.attrs["probability_window_hours"])
            name = event_name(param, statistic, hours)
            threshold, units, comparison = event_threshold(param)
            ptype = int(da.attrs["GRIB_probabilityType"])
            raw_threshold = float(da.attrs["GRIB_lowerLimit"])
            converted = raw_threshold * 86400 if param.startswith("tpr") else raw_threshold
            if ptype != (3 if comparison == "gt" else 0) or not np.isclose(converted, threshold, rtol=1e-5):
                raise ValueError(f"unexpected probability threshold encoding: {param}")
            start_name, end_name = f"probability_window_start__{name}", f"probability_window_end__{name}"
            result = ds[[variable]].rename({variable: name})
            # Height/surface scalars are already represented in the parameter.
            result = result.drop_vars([c for c in result.coords if c not in
                {"time", "step", "valid_time", "latitude", "longitude"} and not result[c].dims], errors="ignore")
            ends = (result["time"] + result["step"]).values.astype("datetime64[ns]")
            result = result.assign_coords({start_name:("step", ends - np.timedelta64(hours, "h")),
                                           end_name:("step", ends)})
            for coord in (start_name, end_name):
                result[coord].attrs.update(zrread_probability_boundary=1, timezone="UTC")
            result[name].attrs.update(
                zrread_kind="probability", units="%", probability_parameter=param,
                probability_threshold=threshold, probability_threshold_units=units,
                probability_comparison=comparison, probability_statistic=statistic,
                probability_type=ptype, probability_threshold_raw=raw_threshold,
                probability_threshold_raw_units="kg m-2 s-1" if param.startswith("tpr") else units,
                probability_window_hours=hours, probability_window_start_coord=start_name,
                probability_window_end_coord=end_name,
                probability_time_label="window_end" if hours else "instant",
            )
            output.append(result)
        return output

    def validate_product(self, ds: xr.Dataset) -> None:
        full = expected_samples(self.name)
        contract = self.historical_availability
        expected = full
        if contract is not None:
            if contract.get("schema_version") != 1 or contract.get("policy") != "historical_available":
                raise ValueError("invalid historical probability availability contract")
            if {v:set(s) for v,s in contract.get("expected_samples", {}).items()} != full:
                raise ValueError("historical probability catalogue differs from current contract")
            expected = {v:set(s) for v,s in contract.get("available_samples", {}).items()}
            if not expected or any(v not in full or not s or not s <= full[v] for v,s in expected.items()):
                raise ValueError("invalid historical probability samples")
            missing = {v: sorted(s - expected.get(v, set())) for v,s in full.items() if s - expected.get(v,set())}
            if (contract.get("missing_samples") != missing or contract.get("missing_variables") != sorted(set(full)-set(expected))
                or contract.get("status") != ("partial" if missing else "complete")):
                raise ValueError("inconsistent historical missing probability receipt")
            grid = contract.get("grid_resolution_degrees")
            if grid is not None:
                if grid not in (0.25, 0.4) or any(not np.allclose(np.abs(np.diff(ds[c].values)),grid,atol=1e-5)
                    for c in ("latitude","longitude")):
                    raise ValueError("historical probability native grid differs from receipt")
        if set(ds.data_vars) != set(expected):
            raise ValueError(f"incomplete probability variables: expected={sorted(expected)}, got={sorted(ds.data_vars)}")
        for var, wanted in expected.items():
            da = ds[var]
            spatial = [d for d in da.dims if d != "step"]
            present = da.notnull().any(spatial).values if spatial else da.notnull().values
            actual = set((ds.step.values[present] / np.timedelta64(1,"h")).astype(int).tolist())
            if actual != wanted:
                raise ValueError(f"incomplete probability windows for {var}: expected={sorted(wanted)}, got={sorted(actual)}")
            if bool(((da < 0) | (da > 100) | np.isinf(da)).any()):
                raise ValueError(f"probability out of 0..100 range: {var}")
            start = ds[da.attrs["probability_window_start_coord"]].values[present]
            end = ds[da.attrs["probability_window_end_coord"]].values[present]
            if (np.isnat(start).any() or np.isnat(end).any()
                or not np.array_equal(end, (ds.time + ds.step).values[present]) or not np.all(
                (end-start)/np.timedelta64(1,"h") == da.attrs["probability_window_hours"])):
                raise ValueError(f"invalid probability bounds: {var}")
            da.attrs["probability_sample_count"] = len(actual)
        ds.attrs.update(probability_product=1, probability_source=self.name,
                        probability_expected_samples=json.dumps({v:len(s) for v,s in expected.items()}, sort_keys=True),
                        probability_complete=1, probability_catalog_complete=int(expected == full),
                        probability_bundles=[240,360])
        if contract is not None:
            ds.attrs["probability_availability"] = json.dumps(contract, sort_keys=True)
            ds.attrs["probability_full_expected_samples"] = json.dumps({v:len(s) for v,s in full.items()}, sort_keys=True)
            if "grid_resolution_degrees" in contract:
                ds.attrs["probability_native_grid_degrees"] = contract["grid_resolution_degrees"]
