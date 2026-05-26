"""Poll a download output tree for new ``*.manifest.json`` files."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from pathlib import Path


def discover_manifests(
    download_root: Path,
    *,
    source: str | None = None,
) -> list[Path]:
    """Find every ``*.manifest.json`` under ``download_root``.

    The download sub-project lays out output as
    ``<source>/<date>/<cycle>z/<date>_<cycle>z_<source>.manifest.json``.
    If ``source`` is given, only that source subtree is scanned.
    """
    base = Path(download_root).resolve()
    if not base.is_dir():
        return []
    if source:
        base = base / source
        if not base.is_dir():
            return []
    return sorted(base.rglob("*.manifest.json"))


def watch_manifests(
    download_root: Path,
    *,
    source: str | None = None,
    interval_seconds: float = 30.0,
    seen: set[Path] | None = None,
) -> Iterator[Path]:
    """Yield each new manifest as it appears. Runs forever; caller breaks."""
    seen = set(seen) if seen else set()
    # Seed with whatever already exists so first iteration only emits new files.
    for existing in discover_manifests(download_root, source=source):
        seen.add(existing.resolve())
    while True:
        for m in discover_manifests(download_root, source=source):
            r = m.resolve()
            if r not in seen:
                seen.add(r)
                yield r
        time.sleep(interval_seconds)


def _seed_seen(paths: Iterable[Path]) -> set[Path]:
    return {p.resolve() for p in paths}
