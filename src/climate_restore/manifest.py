"""Load and verify download-side manifest.json files.

The download sub-project writes one manifest per (date, cycle) with
``schema_version=1`` and a ``files[]`` array carrying ``path``,
``size_bytes`` and ``sha256``. We treat the presence of ``completed_at``
+ all listed files matching size (optionally sha256) as the trigger
condition for downstream processing.
"""

from __future__ import annotations

import hashlib
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
    manifest_path: Path

    @property
    def root_dir(self) -> Path:
        """Directory the manifest lives in; file paths in the manifest are
        recorded relative to the download project root, so we also need a
        ``download_root`` for absolute resolution."""
        return self.manifest_path.parent


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


def _sha256_of(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(
    manifest: Manifest,
    download_root: Path,
    *,
    check_sha256: bool = False,
) -> list[Path]:
    """Verify every file in the manifest exists and matches size (and
    optionally sha256). Returns the list of resolved absolute paths
    in step order. Raises ``FileNotFoundError`` / ``ValueError`` on
    the first failure."""
    if manifest.completed_at is None:
        raise ValueError(
            f"manifest {manifest.manifest_path} has no completed_at; download not finished"
        )
    resolved: list[Path] = []
    for entry in manifest.files:
        abs_path = resolve_file(manifest, download_root, entry)
        if not abs_path.is_file():
            raise FileNotFoundError(f"manifest file missing on disk: {abs_path}")
        size = abs_path.stat().st_size
        if size != entry.size_bytes:
            raise ValueError(
                f"size mismatch for {abs_path}: expected {entry.size_bytes}, got {size}"
            )
        if check_sha256:
            actual = _sha256_of(abs_path)
            if actual != entry.sha256:
                raise ValueError(
                    f"sha256 mismatch for {abs_path}: expected {entry.sha256}, got {actual}"
                )
        resolved.append(abs_path)
    return resolved
