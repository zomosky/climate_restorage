"""Control identity, raw reset windows, manifest coverage and single-field reads."""

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import xarray as xr

from climate_restore.sources.gefs_s2s import GefsS2SAdapter, read_gefs_headers
from climate_restore.processor import _decode_one_grib, process_init

SAMPLES = (
    Path(__file__).resolve().parents[2]
    / "docs/analysis/s2s-pilot-20261104/alternatives/gefs-s2s"
)


def manifest(paths, steps, names):
    return SimpleNamespace(
        kind=None,
        source_name="gefs-s2s",
        init_time="2026-10-29T00:00:00+00:00",
        date="20261029",
        cycle=0,
        source_metadata={"name": "gefs-s2s"},
        variables=[{"name": n} for n in names],
        files=[SimpleNamespace(path=p, step_hours=s) for p, s in zip(paths, steps)],
    )


def sample(name):
    path = SAMPLES / name
    if not path.exists():
        pytest.skip("local genuine GEFS sample absent")
    return path


def test_genuine_reset_windows_and_perturbed_rejection():
    adapter = GefsS2SAdapter()
    cubes = []
    for step in (6, 9):
        for variable in ("APCP", "DSWRF"):
            path = sample(f"c00_f{step:03d}_{variable}_surface.grib2")
            cubes.extend(
                ds
                for _, _, ds in _decode_one_grib(
                    (path, adapter, (116.0, 118.0, 38.0, 40.0))
                )
            )
    ds = adapter.merge_hypercubes(cubes)
    adapter.validate_product(ds)
    assert set(ds.data_vars) == {"tp_accum", "sdswrf_avg"}
    assert ds.member.values.tolist() == [0]
    for name in ds.data_vars:
        start = ds[ds[name].attrs["s2s_window_start_coord"]]
        end = ds[ds[name].attrs["s2s_window_end_coord"]]
        np.testing.assert_array_equal(
            (start - ds.time) / np.timedelta64(1, "h"), [[0], [6]]
        )
        np.testing.assert_array_equal(
            (end - ds.time) / np.timedelta64(1, "h"), [[6], [9]]
        )
    assert ds.sdswrf_avg.attrs["units"] == "W m**-2"
    assert ds.tp_accum.attrs["units"] == "kg m**-2"
    with pytest.raises(ValueError, match="control"):
        adapter.open_hypercubes(sample("p01_f840_TMP_2_m_above_ground.grib2"))


def test_manifest_coverage_step_and_ecds_kind_rejection(tmp_path):
    path = sample("c00_f006_APCP_surface.grib2")
    adapter = GefsS2SAdapter()
    m = manifest([path], [6], ["tp_accum", "sdswrf_avg"])
    adapter.configure_manifest(m)
    with pytest.raises(ValueError, match="coverage"):
        adapter.open_hypercubes(path)
    m = manifest([path], [9], ["tp_accum"])
    adapter.configure_manifest(m)
    with pytest.raises(ValueError, match="step"):
        adapter.open_hypercubes(path)
    m.kind = "s2s"
    with pytest.raises(ValueError, match="manifest"):
        adapter.configure_manifest(m)


@pytest.mark.parametrize(
    "change,match",
    [
        ("duplicate", "duplicate"),
        ("init", "init"),
        ("grid", "grid"),
        ("member", "control"),
        ("window", "window"),
    ],
)
def test_bad_header_contracts(change, match):
    records = read_gefs_headers(sample("c00_f006_APCP_surface.grib2"))
    if change == "duplicate":
        records = records * 2
    elif change == "init":
        records = [records[0], dict(records[0], init="2026-10-30T00:00:00")]
    elif change == "grid":
        records = [records[0], dict(records[0], grid="wrong")]
    elif change == "member":
        records = [dict(records[0], member=1)]
    else:
        records = [dict(records[0], start=7)]
    with pytest.raises(ValueError, match=match):
        GefsS2SAdapter().validate_headers(records)


def make_grib(path, step, member=0):
    """Synthetic 15-field NOAA-shaped GRIB, with interval fields absent at init."""
    import eccodes as ec

    fields = [
        (167, "heightAboveGround", 2, "instant"),
        (168, "heightAboveGround", 2, "instant"),
        (260242, "heightAboveGround", 2, "instant"),
        (165, "heightAboveGround", 10, "instant"),
        (166, "heightAboveGround", 10, "instant"),
        (131, "heightAboveGround", 100, "instant"),
        (132, "heightAboveGround", 100, "instant"),
        (134, "surface", 0, "instant"),
        (260074, "meanSea", 0, "instant"),
        (3015, "heightAboveGround", 2, "max"),
        (3016, "heightAboveGround", 2, "min"),
        (228228, "surface", 0, "accum"),
        (260087, "surface", 0, "avg"),
        (260097, "surface", 0, "avg"),
        (228164, "atmosphere", 0, "avg"),
    ]
    with path.open("wb") as output:
        for param, level_type, level, statistic in fields:
            if step == 0 and statistic != "instant":
                continue
            h = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            try:
                settings = dict(
                    centre="kwbc",
                    productDefinitionTemplateNumber=1 if statistic == "instant" else 11,
                    Ni=2,
                    Nj=2,
                    latitudeOfFirstGridPointInDegrees=40.0,
                    latitudeOfLastGridPointInDegrees=39.5,
                    longitudeOfFirstGridPointInDegrees=117.0,
                    longitudeOfLastGridPointInDegrees=117.5,
                    iDirectionIncrementInDegrees=0.5,
                    jDirectionIncrementInDegrees=0.5,
                    dataDate=20261029,
                    dataTime=0,
                    paramId=param,
                    typeOfLevel=level_type,
                    level=level,
                    stepType=statistic,
                    numberOfForecastsInEnsemble=30,
                    perturbationNumber=member,
                    typeOfEnsembleForecast=1 if member == 0 else 3,
                    startStep=step if statistic == "instant" else (step - 1) // 6 * 6,
                    endStep=step,
                )
                for key, value in settings.items():
                    if key == "paramId" and param in (3015, 3016):
                        ec.codes_set(h, "discipline", 0)
                        ec.codes_set(h, "parameterCategory", 0)
                        ec.codes_set(h, "parameterNumber", 4 if param == 3015 else 5)
                    else:
                        ec.codes_set(h, key, value)
                ec.codes_set(h, "dataType", "cf" if member == 0 else "pf")
                ec.codes_set_values(h, np.arange(4, dtype=float) + step + member * 100)
                ec.codes_write(h, output)
            finally:
                ec.codes_release(h)


@pytest.mark.parametrize("fmt,suffix", [("zarr", ".zarr"), ("netcdf", ".nc")])
def test_synthetic_full_field_contract_and_single_field_reads(tmp_path, fmt, suffix):
    names = {
        "t2m",
        "d2m",
        "r2",
        "u10",
        "v10",
        "u100",
        "v100",
        "sp",
        "prmsl",
        "tmax2m_max",
        "tmin2m_min",
        "tp_accum",
        "sdswrf_avg",
        "sdlwrf_avg",
        "tcc_avg",
    }
    steps = [0, 3, 6, 9, 12, 240, 246, 840]
    paths = [tmp_path / f"f{s:03d}.grib2" for s in steps]
    for p, s in zip(paths, steps):
        make_grib(p, s)
    adapter = GefsS2SAdapter()
    adapter.configure_manifest(manifest(paths, steps, sorted(names)))
    out = tmp_path / "gefs-s2s" / f"20261029_00z_gefs-s2s{suffix}"
    process_init(
        paths,
        adapter=adapter,
        bbox=(116, 119, 38, 41),
        out_path=out,
        workers=1,
        output_format=fmt,
    )
    opener = xr.open_zarr if fmt == "zarr" else xr.open_dataset
    with opener(out) as ds:
        assert set(ds.data_vars) == names
        assert ds.sizes["member"] == 1 and ds.member.item() == 0
        assert (
            json.loads(ds.attrs["s2s_message_coverage"])["tp_accum"]["sample_count"]
            == 7
        )
        for n in ("tp_accum", "sdswrf_avg", "tmax2m_max", "tcc_avg"):
            assert (
                not ds[ds[n].attrs["s2s_sample_present_coord"]]
                .sel(step=np.timedelta64(0, "h"))
                .any()
            )
            assert ds[n].sel(step=np.timedelta64(0, "h")).isnull().all()
    for name in names:
        with opener(out, drop_variables=sorted(names - {name})) as selected:
            for attr in (
                "s2s_sample_present_coord",
                "s2s_window_start_coord",
                "s2s_window_end_coord",
            ):
                assert selected[name].attrs[attr] in selected.coords


def test_wrong_native_grid_and_duplicate_file_samples_rejected(tmp_path):
    path = sample("c00_f006_APCP_surface.grib2")
    records = read_gefs_headers(path)
    with pytest.raises(ValueError, match="grid"):
        GefsS2SAdapter().validate_headers([dict(records[0], grid_spacing=[0.25, 0.25])])
    with pytest.raises(ValueError, match="duplicate"):
        process_init(
            [path, path],
            adapter=GefsS2SAdapter(),
            bbox=(116, 118, 38, 40),
            out_path=tmp_path / "duplicate.zarr",
            output_format="zarr",
            workers=1,
        )
    assert not (tmp_path / "duplicate.zarr").exists()


def test_manifest_duplicate_or_non_native_steps_rejected():
    paths = [Path("f006.grib2"), Path("f009.grib2")]
    for steps in ([6, 6], [6, 241]):
        with pytest.raises(ValueError, match="step"):
            GefsS2SAdapter().configure_manifest(manifest(paths, steps, ["tp_accum"]))


def test_full_horizon_marker_requires_every_native_sample():
    from climate_restore.sources.gefs_s2s import FIELDS, STEPS, INSTANT_FIELDS

    init = np.datetime64("2026-10-29", "ns")
    steps = np.array(STEPS, dtype="timedelta64[h]")
    ds = xr.Dataset(
        coords={
            "step": steps,
            "member": [0],
            "latitude": [39.0],
            "longitude": [117.0],
            "time": init,
            "valid_time": ("step", init + steps),
        }
    )
    for key, (name, units) in FIELDS.items():
        present = np.ones((181, 1), dtype=bool)
        if name not in INSTANT_FIELDS:
            present[0] = False
        end = np.broadcast_to((init + steps)[:, None], (181, 1)).copy()
        start = end.copy() if name in INSTANT_FIELDS else end - np.timedelta64(3, "h")
        start[~present] = np.datetime64("NaT")
        end[~present] = np.datetime64("NaT")
        p, s, e = (
            f"s2s_present__{name}",
            f"s2s_window_start__{name}",
            f"s2s_window_end__{name}",
        )
        values = np.ones((181, 1, 1, 1))
        values[~present] = np.nan
        ds[name] = (
            ("step", "member", "latitude", "longitude"),
            values,
            {
                "GRIB_stepType": key[-1],
                "s2s_sample_present_coord": p,
                "s2s_window_start_coord": s,
                "s2s_window_end_coord": e,
            },
        )
        ds = ds.assign_coords(
            {
                p: (("step", "member"), present),
                s: (("step", "member"), start),
                e: (("step", "member"), end),
            }
        )
    adapter = GefsS2SAdapter()
    adapter.validate_product(ds)
    assert ds.attrs["gefs_complete_horizon"] == 1
    ds.s2s_present__tp_accum.values[-1, 0] = False
    ds.s2s_window_start__tp_accum.values[-1, 0] = np.datetime64("NaT")
    ds.s2s_window_end__tp_accum.values[-1, 0] = np.datetime64("NaT")
    ds.tp_accum.values[-1] = np.nan
    adapter.validate_product(ds)
    assert ds.attrs["gefs_complete_horizon"] == 0


@pytest.mark.parametrize("members", [[0,1,2,3,4], [1,7,30]])
def test_member_subset_manifest_and_roundtrip(tmp_path, members):
    from climate_restore.sources.gefs_s2s import ALL_FIELDS
    paths=[]
    for step in [0,3,6,9]:
        path=tmp_path/f"f{step:03d}.grib2"
        parts=[]
        for member in members:
            part=tmp_path/f"m{member}_f{step}.grib2"
            make_grib(part,step,member=member);parts.append(part.read_bytes())
        path.write_bytes(b"".join(parts));paths.append(path)
    m=manifest(paths,[0,3,6,9],sorted(ALL_FIELDS));m.source_metadata["members"]=members
    adapter=GefsS2SAdapter();adapter.configure_manifest(m)
    out=tmp_path/"ensemble.zarr"
    process_init(paths,adapter=adapter,bbox=(116,119,38,41),out_path=out,workers=1,output_format="zarr")
    with xr.open_zarr(out) as ds:
        np.testing.assert_array_equal(ds.member,members)
        assert ds.attrs["gefs_actual_member_count"]==len(members)
        assert ds.attrs["gefs_control_only"]==0
        for member in members:
            np.testing.assert_array_equal(ds.t2m.sel(member=member,latitude=40.,longitude=117.),np.array([0,3,6,9])+member*100)
        assert int(ds.s2s_present__tp_accum.sum())==3*len(members)
    headers=read_gefs_headers(paths[1])
    with pytest.raises(ValueError,match="coverage"):
        adapter.validate_headers([r for r in headers if r["member"]!=members[-1]],step=3)
    with pytest.raises(ValueError,match="duplicate"):
        adapter.validate_headers(headers+[headers[0]],step=3)


@pytest.mark.parametrize("members", [[],[0,0],[-1],[31],[True],[1.0],["1"]])
def test_bad_member_declaration_rejected(members):
    m=manifest([Path("f003.grib2")],[3],["t2m"]);m.source_metadata["members"]=members
    with pytest.raises(ValueError,match="member"):
        GefsS2SAdapter().configure_manifest(m)



def test_compact_merge_matches_reference_with_sparse_member_times(monkeypatch):
    from climate_restore.sources.s2s import S2SAdapter
    cubes=[]
    init=np.datetime64("2026-10-29", "ns")
    for member,steps in [(0,[0,3,6]),(1,[0,6]),(4,[3,6])]:
        for hour in steps:
            step=np.timedelta64(hour,"h")
            for name in ["t2m","u10"]:
                p,s,e=[prefix+name for prefix in ["s2s_present__","s2s_window_start__","s2s_window_end__"]]
                cube=xr.Dataset({name:(("step","member","latitude","longitude"),np.full((1,1,1,1),member*100+hour,dtype='float32'),{'s2s_sample_present_coord':p,'s2s_window_start_coord':s,'s2s_window_end_coord':e})},coords={'step':[step],'member':[member],'latitude':[39.],'longitude':[117.],'time':init,p:(("step","member"),[[1.]]),s:(("step","member"),[[init+step]]),e:(("step","member"),[[init+step]])},attrs={'s2s_grid_hash':'same-grid'})
                cubes.append(cube)
    adapter=GefsS2SAdapter();original=S2SAdapter.merge_hypercubes
    expected=original(adapter,cubes);observed=[]
    def check_compact(self,parts):
        observed.append(len(parts));return original(self,parts)
    monkeypatch.setattr(S2SAdapter,'merge_hypercubes',check_compact)
    actual=adapter.merge_hypercubes(cubes)
    xr.testing.assert_equal(actual,expected)
    assert observed==[6]
    with pytest.raises(ValueError,match='duplicate'):
        adapter.merge_hypercubes(cubes+[cubes[0]])
    with pytest.raises(ValueError,match='grid'):
        adapter.merge_hypercubes([cubes[0],cubes[1].assign_coords(longitude=[118.])])


def test_canonical_gefs_manifest_adapter():
    from climate_restore.sources.registry import get_source
    m = manifest([Path('f000.subset.grib2')], [0], ['t2m'])
    m.source_name = 'gefs-ens'
    m.source_metadata = {'name': 'gefs-ens', 'members': [0, 1, 2, 3, 4]}
    adapter = get_source('gefs-ens')()
    adapter.configure_manifest(m)
    assert adapter.name == 'gefs-ens'
    assert adapter.expected_members == (0, 1, 2, 3, 4)
    assert get_source('gefs-s2s') is GefsS2SAdapter
