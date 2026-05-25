"""``climate_restore`` CLI entrypoint: ``run`` (single manifest) + ``watch`` (poll)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import JobConfig, load_job, parse_bbox_str
from .logging_setup import configure_logging, get_logger
from .manifest import Manifest, load_manifest, verify
from .processor import DEFAULT_BBOX, DEFAULT_WORKERS, process_init
from .sources import get_source, list_sources
from .sources.base import BaseAdapter
from .watcher import watch_manifests

_log = get_logger(__name__)


def _auto_download_root(manifest_path: Path) -> Path | None:
    """Walk up from a manifest path looking for an ``output`` directory; its
    parent is the download project root."""
    for parent in manifest_path.resolve().parents:
        if parent.name == "output" and parent.parent.is_dir():
            return parent.parent
    return None


def _resolve_download_root(cli_value: Path | None, job: JobConfig, manifest_path: Path) -> Path:
    if cli_value is not None:
        return cli_value.resolve()
    auto = _auto_download_root(manifest_path)
    if auto is not None:
        return auto
    return job.download_root.resolve()


def _out_path(output_dir: Path, manifest: Manifest) -> Path:
    # Flat layout: filename already encodes ``<date>_<cycle>z_<source>``,
    # so a per-source folder is enough to keep different inits separated.
    fname = f"{manifest.date}_{manifest.cycle:02d}z_{manifest.source_name}.nc"
    return (output_dir / manifest.source_name / fname).resolve()


def _resolve_adapter(manifest: Manifest, cli_source_type: str | None) -> BaseAdapter:
    """Look up the source adapter for this manifest.

    ``--source-type`` (CLI) overrides ``manifest.source.name`` so a user can
    force a specific adapter (e.g. testing a new product against the GFS
    rules). Unknown names raise ``KeyError`` with the list of registered
    sources for a clearer error message than a plain stack trace.
    """
    key = cli_source_type or manifest.source_name
    adapter_cls = get_source(key)
    return adapter_cls(name=key)


def _process_one(
    manifest_path: Path,
    *,
    job: JobConfig,
    cli_download_root: Path | None,
    cli_output_dir: Path | None,
    cli_bbox: tuple[float, float, float, float] | None,
    cli_source_type: str | None,
    verify_sha256: bool,
    cli_workers: int | None,
) -> Path:
    manifest = load_manifest(manifest_path)
    download_root = _resolve_download_root(cli_download_root, job, manifest_path)
    output_dir = (cli_output_dir or job.output_dir).resolve()
    bbox = cli_bbox or job.bbox
    workers = cli_workers if cli_workers is not None else job.workers
    adapter = _resolve_adapter(manifest, cli_source_type)

    _log.info(
        "verify_start",
        manifest=str(manifest_path),
        source=manifest.source_name,
        adapter=adapter.name,
        date=manifest.date,
        cycle=manifest.cycle,
        files=len(manifest.files),
        check_sha256=verify_sha256,
    )
    grib_paths = verify(manifest, download_root, check_sha256=verify_sha256)
    _log.info("verify_ok", files=len(grib_paths))

    out_path = _out_path(output_dir, manifest)
    variables = process_init(
        grib_paths, adapter=adapter, bbox=bbox, out_path=out_path, workers=workers
    )
    _log.info("done", out_path=str(out_path), variables=variables)
    return out_path


def _load_job(args: argparse.Namespace) -> JobConfig:
    if getattr(args, "config", None):
        return load_job(Path(args.config))
    return JobConfig()


def _cli_bbox(args: argparse.Namespace) -> tuple[float, float, float, float] | None:
    if getattr(args, "bbox", None):
        return parse_bbox_str(args.bbox)
    return None


def cmd_run(args: argparse.Namespace) -> int:
    job = _load_job(args)
    out = _process_one(
        Path(args.manifest),
        job=job,
        cli_download_root=Path(args.download_root) if args.download_root else None,
        cli_output_dir=Path(args.output_dir) if args.output_dir else None,
        cli_bbox=_cli_bbox(args),
        cli_source_type=args.source_type,
        verify_sha256=args.verify_sha256,
        cli_workers=args.workers,
    )
    print(out)
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    job = _load_job(args)
    download_root = (Path(args.download_root) if args.download_root else job.download_root).resolve()
    _log.info(
        "watch_start",
        download_root=str(download_root),
        source=args.source,
        interval=args.interval,
    )
    for manifest_path in watch_manifests(
        download_root, source=args.source, interval_seconds=args.interval
    ):
        try:
            _process_one(
                manifest_path,
                job=job,
                cli_download_root=download_root,
                cli_output_dir=Path(args.output_dir) if args.output_dir else None,
                cli_bbox=_cli_bbox(args),
                cli_source_type=args.source_type,
                verify_sha256=args.verify_sha256,
                cli_workers=args.workers,
            )
        except Exception as exc:
            _log.error("process_failed", manifest=str(manifest_path), error=str(exc))
    return 0


def cmd_list_sources(args: argparse.Namespace) -> int:
    for name in list_sources():
        print(name)
    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="optional job YAML")
    p.add_argument("--download-root", help="path to download project root")
    p.add_argument("--output-dir", help="where to write .nc outputs")
    p.add_argument("--bbox", help="west,east,south,north (degrees), overrides job/default")
    p.add_argument("--source-type",
                   help="override the source adapter (default: manifest source.name)")
    p.add_argument("--verify-sha256", action="store_true",
                   help="also verify sha256 (slower) in addition to size check")
    p.add_argument("--workers", type=int, default=None,
                   help=f"parallel decode workers; overrides job YAML "
                        f"(YAML default {DEFAULT_WORKERS}; set 1 to disable the pool)")
    p.add_argument("--log-level", default="INFO")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="climate_restore")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="process one manifest into NetCDF")
    p_run.add_argument("--manifest", required=True, help="path to a download manifest.json")
    _add_common(p_run)
    p_run.set_defaults(func=cmd_run)

    p_watch = sub.add_parser("watch", help="poll download root for new manifests")
    p_watch.add_argument("--source", help="restrict to one source subtree, e.g. gfs-0p25")
    p_watch.add_argument("--interval", type=float, default=30.0,
                         help="polling interval in seconds (default 30)")
    _add_common(p_watch)
    p_watch.set_defaults(func=cmd_watch)

    p_ls = sub.add_parser("list-sources", help="list registered source adapters")
    p_ls.add_argument("--log-level", default="INFO")
    p_ls.set_defaults(func=cmd_list_sources)

    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
