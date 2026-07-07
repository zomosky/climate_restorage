"""``climate_restore`` CLI entrypoint: ``run`` (single manifest) + ``watch`` (poll) +
``scan-once`` (idempotent one-shot sweep, for cron)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (
    JobConfig,
    OutputFormat,
    ZarrOptions,
    load_job,
    parse_bbox_str,
    parse_chunks_str,
)
from .logging_setup import configure_logging, get_logger
from .manifest import Manifest, load_manifest, verify
from .processor import (
    DEFAULT_BBOX,
    DEFAULT_WORKERS,
    format_suffix,
    process_init,
)
from .sources import get_source, list_sources
from .sources.base import BaseAdapter
from .watcher import discover_manifests, watch_manifests

_log = get_logger(__name__)


def _resolve_download_root(cli_value: Path | None, job: JobConfig, manifest_path: Path) -> Path:
    if cli_value is not None:
        return cli_value.resolve()
    return job.download_root.resolve()


def _out_path(output_dir: Path, manifest: Manifest, fmt: OutputFormat) -> Path:
    # Flat layout: filename already encodes ``<date>_<cycle>z_<source>``,
    # so a per-source folder is enough to keep different inits separated.
    fname = f"{manifest.date}_{manifest.cycle:02d}z_{manifest.source_name}{format_suffix(fmt)}"
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
    cli_workers: int | None,
    cli_output_format: OutputFormat | None,
    cli_zarr_chunks: dict[str, int] | None,
) -> Path:
    manifest = load_manifest(manifest_path)
    download_root = _resolve_download_root(cli_download_root, job, manifest_path)
    output_dir = (cli_output_dir or job.output_dir).resolve()
    bbox = cli_bbox or job.bbox
    workers = cli_workers if cli_workers is not None else job.workers
    output_format = cli_output_format or job.output_format
    zarr_options = job.zarr.model_copy()
    if cli_zarr_chunks is not None:
        zarr_options = zarr_options.model_copy(update={"chunks": cli_zarr_chunks})
    adapter = _resolve_adapter(manifest, cli_source_type)

    _log.info(
        "verify_start",
        manifest=str(manifest_path),
        source=manifest.source_name,
        adapter=adapter.name,
        date=manifest.date,
        cycle=manifest.cycle,
        files=len(manifest.files),
    )
    grib_paths = verify(manifest, download_root)
    _log.info("verify_ok", files=len(grib_paths))

    out_path = _out_path(output_dir, manifest, output_format)
    variables = process_init(
        grib_paths,
        adapter=adapter,
        bbox=bbox,
        out_path=out_path,
        workers=workers,
        output_format=output_format,
        zarr_options=zarr_options,
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


def _cli_zarr_chunks(args: argparse.Namespace) -> dict[str, int] | None:
    raw = getattr(args, "zarr_chunks", None)
    if not raw:
        return None
    return parse_chunks_str(raw)


def cmd_run(args: argparse.Namespace) -> int:
    job = _load_job(args)
    out = _process_one(
        Path(args.manifest),
        job=job,
        cli_download_root=Path(args.download_root) if args.download_root else None,
        cli_output_dir=Path(args.output_dir) if args.output_dir else None,
        cli_bbox=_cli_bbox(args),
        cli_source_type=args.source_type,
        cli_workers=args.workers,
        cli_output_format=args.output_format,
        cli_zarr_chunks=_cli_zarr_chunks(args),
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
                cli_workers=args.workers,
                cli_output_format=args.output_format,
                cli_zarr_chunks=_cli_zarr_chunks(args),
            )
        except Exception as exc:
            _log.error("process_failed", manifest=str(manifest_path), error=str(exc))
    return 0


def _output_complete(out_path: Path, fmt: OutputFormat, zarr_options: ZarrOptions) -> bool:
    """Has ``out_path`` already been fully written?

    The processor writes in place (``_remove_output`` + ``to_zarr``/``to_netcdf``),
    so a crash mid-write can leave a *partial* store. We therefore key off a
    "done last" marker rather than mere existence:

    - zarr + ``consolidated`` (the job default): ``.zmetadata`` is written last
      by xarray's consolidation pass, so its presence means the store is whole.
    - zarr without consolidation: best-effort — the root group marker exists.
    - netcdf: a single ``to_netcdf`` call, so a non-empty file is good enough.
    """
    if format_suffix(fmt) == ".zarr":
        if not out_path.is_dir():
            return False
        if getattr(zarr_options, "consolidated", True):
            return (out_path / ".zmetadata").is_file()
        return (out_path / ".zgroup").is_file() or (out_path / "zarr.json").is_file()
    return out_path.is_file() and out_path.stat().st_size > 0


def cmd_scan(args: argparse.Namespace) -> int:
    """One-shot, idempotent sweep over every manifest under the download root.

    Unlike ``watch`` (a long-running poll whose "already seen" set lives only in
    memory and is re-seeded from disk on restart — so manifests that appear while
    it is down are skipped forever), ``scan-once`` keys idempotency off the
    on-disk output: any init whose ``.zarr``/``.nc`` is already complete is
    skipped, everything else is processed, then the process exits. That makes it
    safe to drive from cron at any interval. Manifests whose download has not
    finished (no ``completed_at`` or recorded ``failures``) are left for a later
    sweep. Exit code is non-zero iff at least one ready manifest failed to
    process, so a scheduler can alert on it.
    """
    job = _load_job(args)
    download_root = (Path(args.download_root) if args.download_root else job.download_root).resolve()
    output_dir = (Path(args.output_dir) if args.output_dir else job.output_dir).resolve()
    output_format = args.output_format or job.output_format
    manifests = discover_manifests(download_root, source=args.source)
    _log.info(
        "scan_start",
        download_root=str(download_root),
        source=args.source,
        manifests=len(manifests),
        output_format=output_format,
        force=args.force,
    )

    processed = skipped = not_ready = failed = 0
    for manifest_path in manifests:
        try:
            manifest = load_manifest(manifest_path)
        except Exception as exc:
            _log.error("scan_manifest_load_failed", manifest=str(manifest_path), error=str(exc))
            failed += 1
            continue

        if manifest.completed_at is None or manifest.failures:
            _log.info(
                "scan_not_ready",
                manifest=str(manifest_path),
                completed_at=manifest.completed_at,
                failures=len(manifest.failures),
            )
            not_ready += 1
            continue

        out_path = _out_path(output_dir, manifest, output_format)
        if not args.force and _output_complete(out_path, output_format, job.zarr):
            _log.info("scan_skip_done", manifest=str(manifest_path), out_path=str(out_path))
            skipped += 1
            continue

        try:
            _process_one(
                manifest_path,
                job=job,
                cli_download_root=download_root,
                cli_output_dir=output_dir,
                cli_bbox=_cli_bbox(args),
                cli_source_type=args.source_type,
                cli_workers=args.workers,
                cli_output_format=args.output_format,
                cli_zarr_chunks=_cli_zarr_chunks(args),
            )
            processed += 1
        except Exception as exc:
            _log.error("scan_process_failed", manifest=str(manifest_path), error=str(exc))
            failed += 1

    _log.info(
        "scan_done",
        processed=processed,
        skipped=skipped,
        not_ready=not_ready,
        failed=failed,
    )
    return 1 if failed else 0


def cmd_list_sources(args: argparse.Namespace) -> int:
    for name in list_sources():
        print(name)
    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="optional job YAML")
    p.add_argument("--download-root", help="path to download project root")
    p.add_argument("--output-dir", help="where to write outputs (.nc / .zarr)")
    p.add_argument("--bbox", help="west,east,south,north (degrees), overrides job/default")
    p.add_argument("--source-type",
                   help="override the source adapter (default: manifest source.name)")
    p.add_argument("--workers", type=int, default=None,
                   help=f"parallel decode workers; overrides job YAML "
                        f"(YAML default {DEFAULT_WORKERS}; set 1 to disable the pool)")
    p.add_argument("--output-format", choices=("netcdf", "zarr"), default=None,
                   help="output writer; overrides job YAML (default: netcdf)")
    p.add_argument("--zarr-chunks", default=None,
                   help="comma list of dim=size for zarr chunking, e.g. "
                        "'step=-1,latitude=64,longitude=64'; -1 means 'whole dim'")
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

    p_scan = sub.add_parser(
        "scan-once",
        help="process every not-yet-processed manifest once, then exit (idempotent; for cron)",
    )
    p_scan.add_argument("--source", help="restrict to one source subtree, e.g. gfs-0p25")
    p_scan.add_argument("--force", action="store_true",
                        help="reprocess even if the output already exists")
    _add_common(p_scan)
    p_scan.set_defaults(func=cmd_scan)

    p_ls = sub.add_parser("list-sources", help="list registered source adapters")
    p_ls.add_argument("--log-level", default="INFO")
    p_ls.set_defaults(func=cmd_list_sources)

    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
