"""Optional job-config schema (YAML) for ``climate_restore``.

Most CLI flags can be set here; the CLI flags override the YAML values.
A job YAML is *optional* — the simplest workflow is just
``climate_restore run --manifest <path>`` with defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


BBoxTuple = tuple[float, float, float, float]
DEFAULT_WORKERS: int = min(os.cpu_count() or 1, 4)

OutputFormat = Literal["netcdf", "zarr"]
DEFAULT_OUTPUT_FORMAT: OutputFormat = "netcdf"

# Default chunk policy favors spatial slicing: one chunk along ``step``
# (whole time series), 64x64 along latitude/longitude, full pressure_level.
# ``-1`` means "single chunk for that dim" (resolved at write time).
DEFAULT_ZARR_CHUNKS: dict[str, int] = {
    "step": -1,
    "latitude": 64,
    "longitude": 64,
    "pressure_level": -1,
}


class ZarrOptions(BaseModel):
    """Tuning knobs for the Zarr writer.

    chunks: dim->chunk size; ``-1`` means "whole dim in one chunk".
        Unknown dims in the dataset are ignored, so a single default is
        safe across products.
    compressor: numcodecs Blosc cname; ``"none"`` disables compression
        (fastest reads, biggest files).
    clevel: Blosc compression level (1..9).
    consolidated: write a ``.zmetadata`` file for fast ``open_zarr``.
    zarr_format: 2 for max ecosystem compat, 3 for the newer spec.
    """

    model_config = ConfigDict(extra="forbid")

    chunks: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_ZARR_CHUNKS))
    compressor: Literal["zstd", "lz4", "blosclz", "zlib", "none"] = "zstd"
    clevel: int = Field(default=3, ge=0, le=9)
    consolidated: bool = True
    zarr_format: Literal[2, 3] = 2


class JobConfig(BaseModel):
    """Restore job configuration.

    Fields
    ------
    download_root: project root of the download sub-project (so manifest
        ``files[].path`` like ``output/gfs-0p25/...`` can be resolved). Defaults
        to ``../download`` relative to the restorage project.
    output_dir: where to write produced output files.
    bbox: spatial crop window as (west, east, south, north) in degrees.
    verify_sha256: if true, also verify sha256 (slow) on top of size check.
    workers: parallel decode workers; ``1`` disables the process pool.
    output_format: ``"netcdf"`` (default, ``.nc``) or ``"zarr"`` (``.zarr`` dir).
    zarr: writer options used when ``output_format == "zarr"``.
    """

    model_config = ConfigDict(extra="forbid")

    download_root: Path = Field(default=Path("../download"))
    output_dir: Path = Field(default=Path("output"))
    bbox: BBoxTuple = (70.0, 140.0, 15.0, 55.0)
    verify_sha256: bool = False
    workers: int = Field(default=DEFAULT_WORKERS, ge=1)
    output_format: OutputFormat = DEFAULT_OUTPUT_FORMAT
    zarr: ZarrOptions = Field(default_factory=ZarrOptions)

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v: BBoxTuple) -> BBoxTuple:
        west, east, south, north = v
        if not (-180.0 <= west < east <= 360.0):
            raise ValueError(f"invalid lon range: west={west}, east={east}")
        if not (-90.0 <= south < north <= 90.0):
            raise ValueError(f"invalid lat range: south={south}, north={north}")
        return v


def load_job(path: Path) -> JobConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "bbox" in raw and isinstance(raw["bbox"], list):
        raw["bbox"] = tuple(raw["bbox"])
    return JobConfig.model_validate(raw)


def parse_bbox_str(text: str) -> BBoxTuple:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'west,east,south,north'")
    return parts[0], parts[1], parts[2], parts[3]


def parse_chunks_str(text: str) -> dict[str, int]:
    """Parse a ``--zarr-chunks`` flag of the form ``step=-1,lat=64,lon=64``."""
    out: dict[str, int] = {}
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"bad chunk spec '{piece}', expected 'dim=int'")
        k, v = piece.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out
