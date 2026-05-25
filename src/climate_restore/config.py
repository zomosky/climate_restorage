"""Optional job-config schema (YAML) for ``climate_restore``.

Most CLI flags can be set here; the CLI flags override the YAML values.
A job YAML is *optional* — the simplest workflow is just
``climate_restore run --manifest <path>`` with defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


BBoxTuple = tuple[float, float, float, float]
DEFAULT_WORKERS: int = min(os.cpu_count() or 1, 4)


class JobConfig(BaseModel):
    """Restore job configuration.

    Fields
    ------
    download_root: project root of the download sub-project (so manifest
        ``files[].path`` like ``output/gfs-0p25/...`` can be resolved). Defaults
        to ``../download`` relative to the restorage project.
    output_dir: where to write produced ``.nc`` files.
    bbox: spatial crop window as (west, east, south, north) in degrees.
    verify_sha256: if true, also verify sha256 (slow) on top of size check.
    workers: parallel decode workers; ``1`` disables the process pool.
    """

    model_config = ConfigDict(extra="forbid")

    download_root: Path = Field(default=Path("../download"))
    output_dir: Path = Field(default=Path("output"))
    bbox: BBoxTuple = (70.0, 140.0, 15.0, 55.0)
    verify_sha256: bool = False
    workers: int = Field(default=DEFAULT_WORKERS, ge=1)

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
