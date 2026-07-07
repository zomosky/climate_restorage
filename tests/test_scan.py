"""Tests for the ``scan-once`` idempotent sweep and its completeness check.

No GRIB or network: ``_process_one`` is monkeypatched to a recorder so we can
assert which manifests get routed to processing vs skipped vs deferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from climate_restore import cli
from climate_restore.config import ZarrOptions


def _write_manifest(
    root: Path,
    *,
    date: str,
    cycle: int,
    source: str = "gfs-0p25",
    completed_at: str | None = "2026-06-23T00:00:00+00:00",
    failures: list | None = None,
) -> Path:
    """Drop a minimal but loadable ``*.manifest.json`` under ``root``."""
    d = root / source / date / f"{cycle:02d}z"
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source": {"name": source},
        "init_time": f"{date[:4]}-{date[4:6]}-{date[6:]}T{cycle:02d}:00:00+00:00",
        "date": date,
        "cycle": cycle,
        "completed_at": completed_at,
        "variables": [],
        "failures": failures or [],
        "files": [
            {
                "step_hours": 0,
                "path": f"output/{source}/{date}/{cycle:02d}z/gfs_f000.grib2",
                "size_bytes": 1,
                "sha256": "",
                "records_selected": 0,
            }
        ],
    }
    p = d / f"{date}_{cycle:02d}z_{source}.manifest.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _make_complete_zarr(output_dir: Path, *, date: str, cycle: int, source: str = "gfs-0p25") -> Path:
    """Create a store that ``_output_complete`` accepts (consolidated marker present)."""
    store = output_dir / source / f"{date}_{cycle:02d}z_{source}.zarr"
    store.mkdir(parents=True, exist_ok=True)
    (store / ".zmetadata").write_text("{}", encoding="utf-8")
    return store


def _scan_args(download_root: Path, output_dir: Path, **over) -> argparse.Namespace:
    base = dict(
        config=None,
        download_root=str(download_root),
        output_dir=str(output_dir),
        output_format="zarr",
        source=None,
        force=False,
        bbox=None,
        source_type=None,
        workers=None,
        zarr_chunks=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


# ── _output_complete ────────────────────────────────────────────────────────


def test_output_complete_zarr_consolidated(tmp_path: Path):
    store = tmp_path / "x.zarr"
    store.mkdir()
    opts = ZarrOptions(consolidated=True)
    # bare dir without the consolidation marker is NOT complete (partial write)
    assert cli._output_complete(store, "zarr", opts) is False
    (store / ".zmetadata").write_text("{}")
    assert cli._output_complete(store, "zarr", opts) is True


def test_output_complete_zarr_unconsolidated(tmp_path: Path):
    store = tmp_path / "x.zarr"
    store.mkdir()
    opts = ZarrOptions(consolidated=False)
    assert cli._output_complete(store, "zarr", opts) is False
    (store / ".zgroup").write_text("{}")
    assert cli._output_complete(store, "zarr", opts) is True


def test_output_complete_netcdf(tmp_path: Path):
    out = tmp_path / "x.nc"
    opts = ZarrOptions()
    assert cli._output_complete(out, "netcdf", opts) is False
    out.write_bytes(b"")  # empty = not done
    assert cli._output_complete(out, "netcdf", opts) is False
    out.write_bytes(b"\x89HDF")
    assert cli._output_complete(out, "netcdf", opts) is True


# ── cmd_scan routing ────────────────────────────────────────────────────────


def test_scan_routes_new_done_and_not_ready(tmp_path: Path, monkeypatch):
    download_root = tmp_path / "download"
    output_dir = tmp_path / "out"

    m_new = _write_manifest(download_root, date="20260623", cycle=0)
    m_done = _write_manifest(download_root, date="20260622", cycle=12)
    m_pending = _write_manifest(download_root, date="20260621", cycle=0, completed_at=None)
    m_failed = _write_manifest(download_root, date="20260620", cycle=0, failures=[{"phase": "download"}])

    # m_done already has a complete output on disk -> must be skipped
    _make_complete_zarr(output_dir, date="20260622", cycle=12)

    processed: list[Path] = []
    monkeypatch.setattr(cli, "_process_one", lambda mp, **kw: processed.append(Path(mp)) or Path("ok"))

    rc = cli.cmd_scan(_scan_args(download_root, output_dir))

    assert rc == 0  # no hard failures
    assert processed == [m_new]  # only the ready, not-yet-built init
    assert m_done not in processed
    assert m_pending not in processed
    assert m_failed not in processed


def test_scan_force_reprocesses_existing(tmp_path: Path, monkeypatch):
    download_root = tmp_path / "download"
    output_dir = tmp_path / "out"
    m_done = _write_manifest(download_root, date="20260622", cycle=12)
    _make_complete_zarr(output_dir, date="20260622", cycle=12)

    processed: list[Path] = []
    monkeypatch.setattr(cli, "_process_one", lambda mp, **kw: processed.append(Path(mp)) or Path("ok"))

    rc = cli.cmd_scan(_scan_args(download_root, output_dir, force=True))

    assert rc == 0
    assert processed == [m_done]  # --force ignores the existing output


def test_scan_returns_nonzero_on_processing_failure(tmp_path: Path, monkeypatch):
    download_root = tmp_path / "download"
    output_dir = tmp_path / "out"
    _write_manifest(download_root, date="20260623", cycle=0)

    def boom(mp, **kw):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(cli, "_process_one", boom)
    rc = cli.cmd_scan(_scan_args(download_root, output_dir))
    assert rc == 1  # a ready manifest failed -> alertable exit code
