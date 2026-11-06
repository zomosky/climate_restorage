"""NOAA GEFS selected members through day 35, retaining reset/expanding GRIB windows.

This is a standard per-step source, not an ECMWF ECDS group request. Only
coordinate merging and serialization of generic sample/window metadata are
shared with S2SAdapter; ECMWF member/window validation is never used.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import cfgrib
import eccodes as ec
import numpy as np
import xarray as xr
from collections import defaultdict

from climate_restore.sources.base import BaseAdapter
from climate_restore.sources.registry import register
from climate_restore.sources.s2s import S2SAdapter

# Explicit GFS-family naming prevents cfgrib cube grouping from changing 100m
# wind names. The actual NOAA 100m sample decodes as u100/v100.
FIELDS = {
    (167, "heightAboveGround", 2, "instant"): ("t2m", "K"),
    (168, "heightAboveGround", 2, "instant"): ("d2m", "K"),
    (260242, "heightAboveGround", 2, "instant"): ("r2", "%"),
    (165, "heightAboveGround", 10, "instant"): ("u10", "m s**-1"),
    (166, "heightAboveGround", 10, "instant"): ("v10", "m s**-1"),
    (131, "heightAboveGround", 100, "instant"): ("u100", "m s**-1"),
    (132, "heightAboveGround", 100, "instant"): ("v100", "m s**-1"),
    (134, "surface", 0, "instant"): ("sp", "Pa"),
    (260074, "meanSea", 0, "instant"): ("prmsl", "Pa"),
    (3015, "heightAboveGround", 2, "max"): ("tmax2m_max", "K"),
    (3016, "heightAboveGround", 2, "min"): ("tmin2m_min", "K"),
    (228228, "surface", 0, "accum"): ("tp_accum", "kg m**-2"),
    (260087, "surface", 0, "avg"): ("sdswrf_avg", "W m**-2"),
    (260097, "surface", 0, "avg"): ("sdlwrf_avg", "W m**-2"),
    (228164, "atmosphere", 0, "avg"): ("tcc_avg", "%"),
}
STEPS = tuple(range(0, 241, 3)) + tuple(range(246, 841, 6))
ALL_FIELDS = {name for name, unit in FIELDS.values()}
INSTANT_FIELDS = {name for key, (name, unit) in FIELDS.items() if key[-1] == "instant"}


def field_key(record):
    return (record["param"], record["level_type"], record["level"], record["statistic"])


def read_gefs_headers(path: Path) -> list[dict]:
    records = []
    with Path(path).open("rb") as stream:
        while (
            handle := ec.codes_grib_new_from_file(stream, headers_only=True)
        ) is not None:
            try:

                def get(key):
                    return ec.codes_get(handle, key)

                if int(get("stepUnits")) != 1:
                    raise ValueError("GEFS windows must use hours")
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
                        ensemble_type=int(get("typeOfEnsembleForecast")),
                        perturbed_size=int(get("numberOfForecastsInEnsemble")),
                        centre=str(get("centre")),
                        units=str(get("units")),
                        init=init.isoformat(),
                        grid=str(get("md5GridSection")),
                        grid_type=str(get("gridType")),
                        grid_spacing=[
                            float(get("iDirectionIncrementInDegrees")),
                            float(get("jDirectionIncrementInDegrees")),
                        ],
                    )
                )
            finally:
                ec.codes_release(handle)
    if not records:
        raise ValueError(f"empty GEFS GRIB: {path}")
    return records


@register("gefs-s2s")
class GefsS2SAdapter(BaseAdapter):
    name = "gefs-s2s"
    description = "NOAA GEFS 0.5 degree selected members, original windows to day 35"
    expected_members = (0,)
    def merge_hypercubes(self, cubes):
        """Compact per-member time blocks before aligning full member axes.

        Avoid expanding each single message to the complete step/member/grid
        array. The generic merger receives one block per variable/member.
        """
        if not cubes:
            raise ValueError("no GEFS fields inside bbox")
        if len({str(ds.time.values) for ds in cubes}) != 1:
            raise ValueError("mixed GEFS init times")
        if len({ds.attrs.get("s2s_grid_hash") for ds in cubes}) != 1:
            raise ValueError("mixed GEFS grids")
        for ds in cubes[1:]:
            for dim in ("latitude", "longitude"):
                if not np.array_equal(cubes[0][dim].values, ds[dim].values):
                    raise ValueError("mixed GEFS cropped grid")
        groups = defaultdict(list)
        seen = set()
        for ds in cubes:
            if ds.sizes["member"] != 1 or ds.sizes["step"] != 1:
                raise ValueError("GEFS decoder must yield one member/time per message")
            member = int(ds.member.item())
            step = int(ds.step.values[0] / np.timedelta64(1, "h"))
            for name in ds.data_vars:
                key = (name, member, step)
                if key in seen:
                    raise ValueError(f"duplicate GEFS samples/window conflict: {key}")
                seen.add(key)
                groups[(name, member)].append(ds[[name]])
        compact = [xr.concat(parts, dim="step", coords="minimal", compat="equals", join="exact").sortby("step")
                   for parts in groups.values()]
        return S2SAdapter.merge_hypercubes(self, compact)

    def configure_manifest(self, manifest) -> None:
        if manifest.kind == "s2s" or manifest.source_name != self.name:
            raise ValueError(
                "GEFS requires its own standard per-step manifest, not ECDS kind=s2s"
            )
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
            raise ValueError("GEFS manifest init/date/cycle mismatch")
        metadata = manifest.source_metadata
        members = metadata.get("members", [metadata.get("member", 0)])
        if (not isinstance(members, list) or not members
                or any(type(m) is not int or not 0 <= m <= 30 for m in members)
                or len(set(members)) != len(members)):
            raise ValueError("GEFS members must be a nonempty unique integer list in 0..30")
        if "member" in metadata and members != [metadata["member"]]:
            raise ValueError("GEFS member and members declarations conflict")
        self.expected_members = tuple(sorted(members))
        self.file_steps = {}
        for entry in manifest.files:
            if entry.step_hours not in STEPS or entry.path.name in self.file_steps:
                raise ValueError("invalid GEFS manifest step or duplicate file")
            self.file_steps[entry.path.name] = entry.step_hours
        if len(set(self.file_steps.values())) != len(self.file_steps):
            raise ValueError("duplicate GEFS manifest step")
        self.expected_fields = (
            {item["name"] for item in manifest.variables}
            if manifest.variables
            else ALL_FIELDS.copy()
        )
        if not self.expected_fields or not self.expected_fields <= ALL_FIELDS:
            raise ValueError(
                "GEFS manifest variables must declare supported output names"
            )

    def validate_headers(self, records, step=None):
        if not records:
            raise ValueError("empty GEFS headers")
        if len({r["init"] for r in records}) != 1:
            raise ValueError("mixed GEFS init times")
        if len({r["grid"] for r in records}) != 1:
            raise ValueError("mixed GEFS grid")
        if hasattr(self, "expected_init") and records[0]["init"] != self.expected_init:
            raise ValueError("GEFS GRIB init differs from manifest")
        ends = {r["end"] for r in records}
        if len(ends) != 1 or (step is not None and ends != {step}):
            raise ValueError("GEFS message step differs from per-step file")
        seen = set()
        for r in records:
            if (
                r["member"] not in self.expected_members
                or r["forecast_type"] != ("cf" if r["member"] == 0 else "pf")
                or r["ensemble_type"] != (1 if r["member"] == 0 else 3)
            ):
                raise ValueError(
                    "GEFS member/control identity differs from declared members"
                )
            if r["centre"] != "kwbc" or r["perturbed_size"] != 30:
                raise ValueError("GEFS control centre/ensemble metadata mismatch")
            if r["grid_type"] != "regular_ll" or not np.allclose(
                r["grid_spacing"], [0.5, 0.5], atol=1e-8, rtol=0
            ):
                raise ValueError(
                    "GEFS native grid must be regular latitude/longitude at 0.5 degrees"
                )
            key = field_key(r)
            if key not in FIELDS:
                raise ValueError(f"unsupported GEFS field: {key}")
            name, units = FIELDS[key]
            if r["units"] != units:
                raise ValueError(f"GEFS units mismatch: {name}")
            if r["end"] not in STEPS or r["start"] < 0 or r["start"] > r["end"]:
                raise ValueError(f"invalid GEFS step/window: {name}")
            if (r["statistic"] == "instant" and r["start"] != r["end"]) or (
                r["statistic"] != "instant" and r["start"] == r["end"]
            ):
                raise ValueError(f"invalid GEFS statistical window: {name}")
            sample = (r["member"], name)
            if sample in seen:
                raise ValueError(f"duplicate GEFS member/message/window: {sample}")
            seen.add(sample)
        if hasattr(self, "expected_fields"):
            wanted = (
                self.expected_fields
                if next(iter(ends)) != 0
                else self.expected_fields & INSTANT_FIELDS
            )
            wanted = {(member, name) for member in self.expected_members for name in wanted}
            if seen != wanted:
                raise ValueError(
                    f"GEFS per-step field coverage mismatch: expected={sorted(wanted)}, got={sorted(seen)}"
                )

    def open_hypercubes(self, path):
        records = read_gefs_headers(path)
        step = None
        if hasattr(self, "file_steps"):
            if path.name not in self.file_steps:
                raise ValueError("GEFS file is not in configured manifest")
            step = self.file_steps[path.name]
        self.validate_headers(records, step)
        output = []
        for record in records:
            keys = {
                "number": record["member"],
                "paramId": record["param"],
                "typeOfLevel": record["level_type"],
                "level": record["level"],
                "stepType": record["statistic"],
            }
            cubes = cfgrib.open_datasets(
                str(path),
                backend_kwargs={
                    "indexpath": "",
                    "filter_by_keys": keys,
                    "read_keys": [
                        "numberOfForecastsInEnsemble",
                        "typeOfEnsembleForecast",
                        "startStep",
                        "endStep",
                    ],
                },
            )
            if len(cubes) != 1 or len(cubes[0].data_vars) != 1:
                raise ValueError("GEFS decoder did not preserve one message per field")
            ds = cubes[0]
            ds.attrs["_gefs_header"] = json.dumps(record)
            ds.attrs["s2s_grid_hash"] = record["grid"]
            output.append(ds)
        return output

    def rename_hypercube(self, ds):
        record = json.loads(ds.attrs["_gefs_header"])
        name = FIELDS[field_key(record)][0]
        if (
            "number" not in ds.coords
            or ds.number.size != 1
            or int(ds.number.values.item()) != record["member"]
            or np.atleast_1d(ds.step.values).size != 1
            or np.atleast_1d(ds.step.values)[0] != np.timedelta64(record["end"], "h")
            or np.datetime64(ds.time.values, "ns")
            != np.datetime64(record["init"], "ns")
        ):
            raise ValueError("GEFS decoder lost message coordinates")
        old = next(iter(ds.data_vars))
        result = ds.rename({old: name, "number": "member"}).drop_vars(
            ["valid_time", record["level_type"]], errors="ignore"
        )
        for dim in ("step", "member"):
            if dim not in result.dims:
                result = result.expand_dims(dim)
        result = result.transpose("step", "member", "latitude", "longitude")
        result.attrs.pop("_gefs_header", None)
        p, s, e = (
            f"s2s_present__{name}",
            f"s2s_window_start__{name}",
            f"s2s_window_end__{name}",
        )
        init = np.datetime64(record["init"], "ns")
        result = result.assign_coords(
            {
                p: (("step", "member"), [[1.0]]),
                s: (
                    ("step", "member"),
                    np.array([[init + np.timedelta64(record["start"], "h")]]),
                ),
                e: (
                    ("step", "member"),
                    np.array([[init + np.timedelta64(record["end"], "h")]]),
                ),
            }
        )
        for coord in (s, e):
            result[coord].attrs.update(zrread_time_boundary=1, timezone="UTC")
        result[name].attrs.update(
            s2s_sample_present_coord=p,
            s2s_window_start_coord=s,
            s2s_window_end_coord=e,
            gefs_window_semantics="instant"
            if record["statistic"] == "instant"
            else "original_grib_interval",
            gefs_perturbed_members_metadata=record["perturbed_size"],
        )
        return [result]

    def validate_product(self, ds):
        if not np.array_equal(ds.member.values, self.expected_members):
            raise ValueError("GEFS product member coverage differs from manifest")
        if hasattr(self, "expected_init") and np.datetime64(
            ds.time.values, "ns"
        ) != np.datetime64(self.expected_init, "ns"):
            raise ValueError("GEFS product init differs from manifest")
        if hasattr(self, "file_steps"):
            actual = (ds.step.values / np.timedelta64(1, "h")).astype(int).tolist()
            if actual != sorted(self.file_steps.values()):
                raise ValueError("GEFS product step coverage differs from manifest")
            wanted = (
                self.expected_fields
                if any(actual)
                else self.expected_fields & INSTANT_FIELDS
            )
            if set(ds.data_vars) != wanted:
                raise ValueError("GEFS product field coverage differs from manifest")
        for name in ds.data_vars:
            da = ds[name]
            p = ds[da.attrs["s2s_sample_present_coord"]]
            start = ds[da.attrs["s2s_window_start_coord"]]
            end = ds[da.attrs["s2s_window_end_coord"]]
            statistic = da.attrs.get("GRIB_stepType")
            invalid = (start != end) if statistic == "instant" else (start >= end)
            if bool((p & invalid).any()):
                raise ValueError(f"invalid GEFS original window: {name}")
        # Generic presence/NaT, CF coordinate associations and empty-chunk
        # serialization only; no ECMWF header/init-cumulative assumption here.
        S2SAdapter.validate_product(self, ds)
        # A single cfgrib message attribute must not describe a mixed ensemble.
        if 0 in self.expected_members and len(self.expected_members) > 1:
            for attrs in [ds.attrs, *(ds[name].attrs for name in ds.data_vars)]:
                for key in ("GRIB_dataType", "GRIB_typeOfEnsembleForecast"):
                    attrs.pop(key, None)
        ds.attrs.update(
            gefs_member_types=json.dumps({str(m): "cf" if m == 0 else "pf" for m in self.expected_members}),
            gefs_control_only=int(self.expected_members == (0,)),
            gefs_actual_member_count=len(self.expected_members),
            gefs_selected_members=json.dumps(list(self.expected_members)),
            gefs_full_ensemble=int(self.expected_members == tuple(range(31))),
            gefs_complete_horizon=int(
                set(ds.data_vars) == ALL_FIELDS
                and np.array_equal(
                    ds.step.values, np.array(STEPS, dtype="timedelta64[h]")
                )
                and all(
                    int(ds[ds[name].attrs["s2s_sample_present_coord"]].sum())
                    == len(self.expected_members) * (181 if name in INSTANT_FIELDS else 180)
                    for name in ds.data_vars
                )
            ),
        )


@register("gefs-ens")
class GefsEnsAdapter(GefsS2SAdapter):
    """Canonical member product, retaining legacy manifest support separately."""
    name = "gefs-ens"
