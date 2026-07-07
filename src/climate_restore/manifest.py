"""Load and verify download-side manifest.json files.

The download sub-project writes one manifest per (date, cycle) with
``schema_version=1``, a ``files[]`` array of step files and a top-level
``failures[]`` array. A download is considered usable iff
``completed_at`` is set and ``failures`` is empty; per-file size/sha
fields are no longer trusted (they may be placeholders while the
manifest is being refreshed). Actual file readability is left to the
cfgrib open step in the processor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestFile:
    step_hours: int
    path: Path
    size_bytes: int
    sha256: str
    records_selected: int


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    source_name: str
    init_time: str
    date: str
    cycle: int
    completed_at: str | None
    variables: list[dict[str, Any]]
    files: list[ManifestFile]
    failures: list[Any]
    manifest_path: Path

    @property
    def root_dir(self) -> Path:
        """Directory the manifest lives in; file paths in the manifest are
        recorded relative to the download project root, so we also need a
        ``download_root`` for absolute resolution."""
        return self.manifest_path.parent


class ManifestHasFailures(Exception):
    """Raised by :func:`verify` when the manifest reports download-side
    failures. Carries the original ``failures`` payload so the caller can
    log it and decide to skip the manifest."""

    def __init__(self, manifest_path: Path, failures: list[Any]) -> None:
        self.manifest_path = manifest_path
        self.failures = failures
        super().__init__(
            f"manifest {manifest_path} reports {len(failures)} failure(s)"
        )


def load_manifest(manifest_path: Path) -> Manifest:
    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError(
            f"unsupported manifest schema_version: {raw.get('schema_version')!r}"
        )
    files = [
        ManifestFile(
            step_hours=int(f["step_hours"]),
            path=Path(f["path"]),
            size_bytes=int(f["size_bytes"]),
            sha256=str(f["sha256"]),
            records_selected=int(f.get("records_selected", 0)),
        )
        for f in raw.get("files", [])
    ]
    files.sort(key=lambda f: f.step_hours)
    return Manifest(
        schema_version=int(raw["schema_version"]),
        source_name=str(raw["source"]["name"]),
        init_time=str(raw["init_time"]),
        date=str(raw["date"]),
        cycle=int(raw["cycle"]),
        completed_at=raw.get("completed_at"),
        variables=list(raw.get("variables", [])),
        files=files,
        failures=list(raw.get("failures", [])),
        manifest_path=Path(manifest_path).resolve(),
    )


def resolve_file(manifest: Manifest, download_root: Path, entry: ManifestFile) -> Path:
    """Resolve a manifest ``files[].path`` to an absolute filesystem path.

    The manifest stores paths like ``output/gfs-0p25/20260501/12z/f000.subset.grib2``
    which are relative to the download project root. We first try
    ``download_root / entry.path``; if that does not exist we fall back to
    ``manifest.root_dir / basename`` (same directory as the manifest)."""
    cand = (download_root / entry.path).resolve()
    if cand.is_file():
        return cand
    sibling = (manifest.root_dir / entry.path.name).resolve()
    if sibling.is_file():
        return sibling
    return cand


def verify(
    manifest: Manifest,
    download_root: Path,
) -> list[Path]:
    """Resolve and return the manifest's step files in step order.

    Completion gate: a manifest is considered usable only when
    ``completed_at`` is set and ``failures`` is empty. If ``failures`` is
    non-empty, :class:`ManifestHasFailures` is raised so the batch driver
    can log the payload and skip the init.

    File-level size / sha256 fields are not checked here -- they may be
    placeholder values while the manifest is being refreshed. Actual file
    readability is left to the cropping step which opens each GRIB via
    cfgrib; any unreadable file naturally fails there with a clear error.
    """
    if manifest.completed_at is None:
        raise ValueError(
            f"manifest {manifest.manifest_path} has no completed_at; download not finished"
        )
    if manifest.failures:
        raise ManifestHasFailures(manifest.manifest_path, manifest.failures)
    resolved: list[Path] = []
    for entry in manifest.files:
        abs_path = resolve_file(manifest, download_root, entry)
        if not abs_path.is_file():
            raise FileNotFoundError(f"manifest file missing on disk: {abs_path}")
        resolved.append(abs_path)
    return resolved
