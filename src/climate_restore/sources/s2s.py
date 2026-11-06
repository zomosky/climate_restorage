"""ECMWF S2S: preserve message presence, members and original statistical windows.

Only headers are scanned eagerly. cfgrib values remain lazy until the processor
has cropped the spatial domain. No member is inferred from a field's NaNs.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import cfgrib
import eccodes as ec
import numpy as np
import xarray as xr

from climate_restore.sources.base import BaseAdapter
from climate_restore.sources.registry import register

# ECDS names and ECMWF native cadence. Historical group requests sometimes ask
# for 6h radiation although ECMWF publishes those fields at 24h only.
PARAMETERS = {
    165: ("u10", "10_m_u_component_of_wind", 6),
    166: ("v10", "10_m_v_component_of_wind", 6),
    151: ("msl", "mean_sea_level_pressure", 24),
    134: ("sp", "surface_pressure", 24),
    121: ("mx2t6", "maximum_2_m_temperature_in_the_last_6_hours", 6),
    122: ("mn2t6", "minimum_2_m_temperature_in_the_last_6_hours", 6),
    167: ("t2m", "2_m_temperature", 24),
    168: ("d2m", "2_m_dewpoint_temperature", 24),
    228164: ("tcc", "total_cloud_cover", 24),
    164: ("tcc", "total_cloud_cover", 24),
    169: ("ssrd", "surface_solar_radiation_downwards", 24),
    175: ("strd", "surface_thermal_radiation_downwards", 24),
    228228: ("tp", "total_precipitation", 6),
    228: ("tp", "total_precipitation", 6),
    131: ("u", "u_component_of_wind", 24),
    132: ("v", "v_component_of_wind", 24),
    130: ("t", "temperature", 24),
    156: ("gh", "geopotential_height", 24),
}


EXTREME_STATISTICS = {121: "max", 122: "min"}


def read_headers(path: Path) -> list[dict]:
    records = []
    with Path(path).open("rb") as handle:
        while (
            gid := ec.codes_grib_new_from_file(handle, headers_only=True)
        ) is not None:
            try:

                def get(key):
                    return ec.codes_get(gid, key)

                if int(get("stepUnits")) != 1:
                    raise ValueError("S2S step/window units must be hours")
                init = datetime.strptime(
                    f"{int(get('dataDate')):08d}{int(get('dataTime')):04d}",
                    "%Y%m%d%H%M",
                )
                records.append(
                    dict(
                        param=int(get("paramId")),
                        level_type=str(get("typeOfLevel")),
                        level=float(get("level")),
                        statistic=str(get("stepType")),
                        start=int(get("startStep")),
                        end=int(get("endStep")),
                        member=int(get("number")),
                        forecast_type=str(get("dataType")),
                        ensemble_size=int(get("numberOfForecastsInEnsemble")),
                        init=init.isoformat(),
                        grid=str(get("md5GridSection")),
                    )
                )
            finally:
                ec.codes_release(gid)
    if not records:
        raise ValueError(f"empty S2S GRIB: {path}")
    return records


def _list(value):
    return value if isinstance(value, list) else [value]


@register("s2s-ecmwf")
@register("s2s-ecmwf-perturbed")
@register("s2s-ecmwf-ensemble")
class S2SAdapter(BaseAdapter):
    name = "s2s-ecmwf"
    description = "ECMWF S2S members and original per-message windows"

    def configure_manifest(self, manifest) -> None:
        if manifest.kind != "s2s":
            raise ValueError("S2S adapter requires kind=s2s group manifest")
        self.expected_init = (
            datetime.fromisoformat(manifest.init_time)
            .astimezone(timezone.utc)
            .replace(tzinfo=None)
            .isoformat()
        )
        if (
            self.expected_init
            != datetime.strptime(
                f"{manifest.date}{manifest.cycle:02d}", "%Y%m%d%H"
            ).isoformat()
        ):
            raise ValueError("S2S manifest init/date/cycle disagree")
        wanted = {
            "s2s-ecmwf": "control_forecast",
            "s2s-ecmwf-perturbed": "perturbed_forecast",
            "s2s-ecmwf-ensemble": "ensemble",
        }[self.name]
        if manifest.source_metadata.get("forecast_type") != wanted:
            raise ValueError("S2S source name and forecast_type disagree")
        self.requests = {}
        for entry in manifest.files:
            if entry.path.name in self.requests:
                raise ValueError("ambiguous S2S file basename in manifest")
            request = dict(entry.request)
            kind = request.get("forecast_type", wanted)
            if (
                kind not in ("control_forecast", "perturbed_forecast")
                or wanted != "ensemble"
                and kind != wanted
            ):
                raise ValueError("S2S per-file forecast_type disagrees with source")
            request["forecast_type"] = kind
            self.requests[entry.path.name] = request

    def validate_headers(self, records: list[dict], request: dict) -> None:
        if len({r["init"] for r in records}) != 1:
            raise ValueError("mixed S2S init times")
        if len({r["grid"] for r in records}) != 1:
            raise ValueError("mixed S2S grids")
        if hasattr(self, "expected_init") and records[0]["init"] != self.expected_init:
            raise ValueError("S2S GRIB init differs from manifest")
        for key, fmt in (
            ("year", "%Y"),
            ("month", "%m"),
            ("day", "%d"),
            ("time", "%H:%M"),
        ):
            if key in request:
                init = datetime.fromisoformat(records[0]["init"])
                wanted = {str(v) for v in _list(request[key])}
                actual = init.strftime(fmt)
                if actual not in wanted and (
                    key == "time" or str(int(actual)) not in wanted
                ):
                    raise ValueError("S2S request init differs from GRIB")
        sizes = {r["ensemble_size"] for r in records}
        if len(sizes) != 1 or min(sizes) < 1:
            raise ValueError("inconsistent S2S ensemble size metadata")
        seen = set()
        groups = defaultdict(list)
        for r in records:
            if r["param"] not in PARAMETERS:
                raise ValueError(f"unsupported S2S parameter {r['param']}")
            if r["start"] < 0 or r["end"] < r["start"]:
                raise ValueError("invalid S2S window")
            if r["param"] in EXTREME_STATISTICS:
                if (
                    r["statistic"] != EXTREME_STATISTICS[r["param"]]
                    or not 6 <= r["end"] <= 1104
                    or r["end"] % 6
                    or r["start"] != r["end"] - 6
                    or r["level_type"] != "heightAboveGround"
                    or r["level"] != 2
                ):
                    raise ValueError(
                        "S2S 6-hour extreme requires its original [end-6,end] window, end=6..1104h"
                    )
            elif r["statistic"] not in ("instant", "avg", "accum"):
                raise ValueError("unsupported S2S statistic")
            if r["param"] == 134 and (
                r["statistic"] != "instant"
                or not 0 <= r["end"] <= 1104
                or r["end"] % 24
                or r["level_type"] != "surface"
                or r["level"] != 0
            ):
                raise ValueError(
                    "S2S surface pressure must be instantaneous at surface, 0..1104h every 24h"
                )
            if r["statistic"] == "accum" and r["start"] != 0:
                raise ValueError("S2S accumulation must start at init")
            if r["statistic"] == "instant" and r["start"] != r["end"]:
                raise ValueError("S2S instantaneous window is not a point")
            if (
                r["member"] < 0
                or r["member"] >= r["ensemble_size"]
                or r["forecast_type"] != ("cf" if r["member"] == 0 else "pf")
            ):
                raise ValueError("S2S member/forecast type mismatch")
            key = (r["param"], r["level_type"], r["level"], r["member"], r["end"])
            if key in seen:
                raise ValueError("duplicate S2S message/window conflict")
            seen.add(key)
            groups[(r["param"], r["level_type"], r["level"], r["statistic"])].append(r)
        kind = request.get("forecast_type")
        if kind is None:
            kind = (
                "perturbed_forecast"
                if self.name.endswith("-perturbed")
                else "control_forecast"
                if self.name == "s2s-ecmwf"
                else None
            )
        expected_members = (
            {0}
            if kind == "control_forecast"
            else set(range(1, next(iter(sizes))))
            if kind == "perturbed_forecast"
            else set(range(next(iter(sizes))))
        )
        # ECDS retrieves all members; an explicit member subset is respected when supplied.
        for key in ("number", "member"):
            if key in request:
                subset = {int(v) for v in _list(request[key])}
                if not subset or not subset <= expected_members:
                    raise ValueError(
                        "S2S requested members disagree with forecast type"
                    )
                expected_members = subset
        if any(r["member"] not in expected_members for r in records):
            raise ValueError("S2S unexpected member for requested forecast type")
        actual_vars = {PARAMETERS[r["param"]][1] for r in records}
        if "variable" in request and actual_vars != set(_list(request["variable"])):
            raise ValueError("S2S request variable coverage mismatch")
        for (param, level_type, level, statistic), group in groups.items():
            cadence = PARAMETERS[param][2]
            actual = {(r["member"], r["start"], r["end"]) for r in group}
            if "leadtime_hour" in request:
                windows = set()
                for lead in _list(request["leadtime_hour"]):
                    parts = [int(v) for v in str(lead).split("_")]
                    end = parts[-1]
                    if param in EXTREME_STATISTICS:
                        if len(parts) != 1 or not 6 <= end <= 1104 or end % 6:
                            raise ValueError(
                                "S2S 6-hour extreme request endpoints must be 6..1104h"
                            )
                        start = end - 6
                    else:
                        if end % cadence:
                            continue
                        start = (
                            parts[0]
                            if len(parts) == 2
                            else 0
                            if statistic == "accum"
                            else end
                        )
                    if statistic == "avg" and len(parts) != 2:
                        raise ValueError(
                            "S2S averaged request requires explicit windows"
                        )
                    windows.add((start, end))
            else:
                windows = {(r["start"], r["end"]) for r in group}
            expected = {
                (m, start, end) for m in expected_members for start, end in windows
            }
            if actual != expected:
                raise ValueError(
                    f"S2S missing/unexpected member or window coverage: parameter={param} level={level}; missing={len(expected - actual)} extra={len(actual - expected)}"
                )
        if "level_value" in request:
            expected_levels = {
                float(str(v).split("_")[0]) for v in _list(request["level_value"])
            }
            for param in {r["param"] for r in records}:
                if {
                    r["level"] for r in records if r["param"] == param
                } != expected_levels:
                    raise ValueError("S2S pressure level coverage mismatch")

    def open_hypercubes(self, path: Path) -> list[xr.Dataset]:
        records = read_headers(path)
        requests = getattr(self, "requests", None)
        if requests is not None and path.name not in requests:
            raise ValueError("S2S file is not declared by configured manifest")
        self.validate_headers(
            records, requests[path.name] if requests is not None else {}
        )
        groups = defaultdict(list)
        for r in records:
            groups[(r["param"], r["level_type"], r["statistic"])].append(r)
        output = []
        for (param, level_type, statistic), headers in groups.items():
            cubes = cfgrib.open_datasets(
                str(path),
                backend_kwargs={
                    "indexpath": "",
                    "filter_by_keys": {
                        "paramId": param,
                        "typeOfLevel": level_type,
                        "stepType": statistic,
                    },
                },
            )
            represented = set()
            for ds in cubes:
                # cfgrib allocates full spatial fields before applying a bbox.
                # Limit each load to one member so a global 101-member cube is
                # never materialized even though the outer slice is regional.
                members = np.atleast_1d(ds.number.values)
                hours = set(
                    (np.atleast_1d(ds.step.values) / np.timedelta64(1, "h")).astype(int)
                )
                levels = set(np.atleast_1d(ds[level_type].values))
                if len(ds.data_vars) != 1:
                    raise ValueError("S2S parameter decoding did not produce one field")
                for member in members:
                    chunk = ds.sel(number=[member]) if "number" in ds.dims else ds
                    selected = [
                        r
                        for r in headers
                        if r["member"] == int(member)
                        and r["end"] in hours
                        and r["level"] in levels
                    ]
                    keys = {(r["member"], r["end"], r["level"]) for r in selected}
                    if not selected or represented & keys:
                        raise ValueError(
                            "S2S cfgrib duplicated or lost message coordinates"
                        )
                    represented.update(keys)
                    chunk.attrs = dict(
                        ds.attrs,
                        _s2s_headers=json.dumps(selected),
                        s2s_grid_hash=records[0]["grid"],
                    )
                    output.append(chunk)
            expected = {(r["member"], r["end"], r["level"]) for r in headers}
            if represented != expected:
                raise ValueError("S2S cfgrib lost message coordinates")
        return output

    def rename_hypercube(self, ds: xr.Dataset) -> list[xr.Dataset]:
        headers = json.loads(ds.attrs["_s2s_headers"])
        if "number" in ds.coords:
            ds = ds.rename({"number": "member"})
        if "member" not in ds.coords:
            raise ValueError("S2S cfgrib lost member coordinate")
        for dim in ("member", "step"):
            if dim not in ds.dims:
                ds = ds.expand_dims(dim)
        level_type = headers[0]["level_type"]
        if level_type == "isobaricInhPa":
            if level_type not in ds.dims:
                ds = ds.expand_dims(level_type)
            ds = ds.rename({level_type: "pressure_level"})
        elif level_type in ds.coords:
            ds = ds.drop_vars(level_type)
        output = []
        for old in ds.data_vars:
            stat = headers[0]["statistic"]
            base = PARAMETERS[int(ds[old].attrs["GRIB_paramId"])][0]
            name = base if stat == "instant" else f"{base}_{stat}"
            result = (
                ds[[old]].rename({old: name}).drop_vars("valid_time", errors="ignore")
            )
            result.attrs.pop("_s2s_headers", None)
            dims = [
                d
                for d in ("step", "member", "pressure_level")
                if d in result[name].dims
            ]
            shape = tuple(result.sizes[d] for d in dims)
            present = np.full(shape, np.nan)
            starts = np.full(shape, np.datetime64("NaT", "ns"))
            ends = starts.copy()
            indices = {d: {v: i for i, v in enumerate(result[d].values)} for d in dims}
            init = np.datetime64(headers[0]["init"], "ns")
            for r in headers:
                point = {
                    "step": np.timedelta64(r["end"], "h").astype("timedelta64[ns]"),
                    "member": r["member"],
                    "pressure_level": r["level"],
                }
                if any(point[d] not in indices[d] for d in dims):
                    raise ValueError("S2S renamed cube lost message coordinates")
                idx = tuple(indices[d][point[d]] for d in dims)
                present[idx] = 1
                starts[idx] = init + np.timedelta64(r["start"], "h")
                ends[idx] = init + np.timedelta64(r["end"], "h")
            p, s, e = (
                f"s2s_present__{name}",
                f"s2s_window_start__{name}",
                f"s2s_window_end__{name}",
            )
            result = result.assign_coords(
                {p: (dims, present), s: (dims, starts), e: (dims, ends)}
            )
            for c in (s, e):
                result[c].attrs.update(zrread_time_boundary=1, timezone="UTC")
            result[name].attrs.update(
                s2s_sample_present_coord=p,
                s2s_window_start_coord=s,
                s2s_window_end_coord=e,
                s2s_ensemble_size_metadata=headers[0]["ensemble_size"],
            )
            output.append(result)
        return output

    def merge_hypercubes(self, cubes: list[xr.Dataset]) -> xr.Dataset:
        if not cubes:
            raise ValueError("no S2S fields inside bbox")
        if len({str(ds.time.values) for ds in cubes}) != 1:
            raise ValueError("mixed S2S init times")
        if len({ds.attrs.get("s2s_grid_hash") for ds in cubes}) != 1:
            raise ValueError("mixed S2S grids")
        for ds in cubes[1:]:
            for dim in ("latitude", "longitude"):
                if not np.array_equal(cubes[0][dim].values, ds[dim].values):
                    raise ValueError("mixed S2S cropped grid")
        by_variable = defaultdict(list)
        for ds in cubes:
            for name in ds.data_vars:
                by_variable[name].append(ds[[name]])
        merged = []
        for name, parts in by_variable.items():
            p = parts[0][name].attrs["s2s_sample_present_coord"]
            aligned = xr.align(*[ds[p] for ds in parts], join="outer")
            counts = sum(a.fillna(0).astype(int) for a in aligned)
            if bool((counts > 1).any()):
                raise ValueError(f"duplicate S2S samples/window conflict: {name}")
            # Absence uses NaN until merge, so a different file may fill a hole.
            parts = [ds.assign_coords({p: ds[p].where(ds[p] == 1)}) for ds in parts]
            merged.append(
                xr.merge(
                    parts, compat="no_conflicts", join="outer", combine_attrs="override"
                )
            )
        final = (
            xr.merge(
                merged, compat="no_conflicts", join="outer", combine_attrs="override"
            )
            .sortby("step")
            .sortby("member")
        )
        for name in final.data_vars:
            p = final[name].attrs["s2s_sample_present_coord"]
            final = final.assign_coords({p: final[p].fillna(0).astype(bool)})
        return final.assign_coords(valid_time=final.time + final.step)

    def validate_product(self, ds: xr.Dataset) -> None:
        coverage = {}
        for name in ds.data_vars:
            da = ds[name]
            p = ds[da.attrs["s2s_sample_present_coord"]]
            start = ds[da.attrs["s2s_window_start_coord"]]
            end = ds[da.attrs["s2s_window_end_coord"]]
            if bool(
                (
                    p
                    & (
                        start.isnull()
                        | end.isnull()
                        | (end < start)
                        | (end != ds.valid_time)
                    )
                ).any()
            ):
                raise ValueError(f"invalid S2S original windows: {name}")
            param = da.attrs.get("GRIB_paramId")
            if param in EXTREME_STATISTICS and (
                da.attrs.get("GRIB_stepType") != EXTREME_STATISTICS[param]
                or bool((p & ((end - start) != np.timedelta64(6, "h"))).any())
            ):
                raise ValueError(f"invalid S2S 6-hour extreme window: {name}")
            if bool((~p & (start.notnull() | end.notnull())).any()):
                raise ValueError(f"S2S absent sample has a window: {name}")
            if bool((da.notnull() & ~p).any()):
                raise ValueError(f"S2S values without message presence: {name}")
            members = (
                ds.member.where(p.any([d for d in p.dims if d != "member"]), drop=True)
                .values.astype(int)
                .tolist()
            )
            # Explicit float encoding preserves NaT through Zarr/NetCDF CF decoding.
            # Integer-hour encoding without a fill value can overflow on read.
            for coord in (start, end):
                coord.encoding.update(
                    dtype="float64",
                    units="hours since 1970-01-01 00:00:00",
                    _FillValue=np.nan,
                )
            coverage[name] = {"members": members, "sample_count": int(p.sum())}
            da.attrs["s2s_sample_count"] = int(p.sum())
            # cfgrib's serialized coordinate list still names number/surface
            # and omits the S2S presence/window coordinates added above. Let
            # xarray rebuild each field's association from the actual dataset,
            # so opening one field with drop_variables retains its metadata.
            da.encoding.pop("coordinates", None)
            # CF-only fields and mixed native cadences leave whole member/time
            # chunks empty. Write those NaN chunks explicitly so the existing
            # physical chunk-integrity check still detects real interrupted writes.
            da.encoding["write_empty_chunks"] = True
        ds.attrs.update(
            s2s_product=1,
            s2s_source=self.name,
            s2s_message_coverage=json.dumps(coverage, sort_keys=True),
            s2s_complete=1,
        )
