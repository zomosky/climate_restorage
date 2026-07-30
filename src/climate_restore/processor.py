"""Crop downloaded GRIB subsets and time-concatenate one (date, cycle) into NetCDF.

Strategy
--------
For each ``f###.subset.grib2`` file in a manifest:

1. ``cfgrib.open_datasets`` splits the heterogeneous GRIB into one
   ``xr.Dataset`` per coherent hypercube.
2. The source adapter (:mod:`climate_restore.sources`) crops each
   hypercube to the configured bbox and applies its rename rules,
   yielding one or more datasets with friendly variable names.
3. Renamed hypercubes are bucketed by their renamed variable signature,
   concatenated across step files, then merged into a single flat
   ``xr.Dataset`` written to one NetCDF4 file (no groups).

The processor itself contains no source-specific logic; everything that
varies between GFS / AIFS / HRRR / ... lives in the adapter.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import warnings
from collections import OrderedDict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# cfgrib's internal hypercube builder calls xr.merge without an explicit
# ``compat`` argument and triggers a FutureWarning on every call. Filter at
# module import so ProcessPoolExecutor workers (which re-import this module
# but do not run configure_logging) are also covered.
warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")

import cfgrib  # noqa: E402
import xarray as xr  # noqa: E402
from tqdm import tqdm  # noqa: E402

from .config import DEFAULT_WORKERS, OutputFormat, ZarrOptions
from .logging_setup import get_logger
from .sources.base import BaseAdapter

DEFAULT_BBOX: tuple[float, float, float, float] = (70.0, 140.0, 15.0, 55.0)

# Suffix per format; ``.zarr`` is a directory store, ``.nc`` a single file.
_FORMAT_SUFFIX: dict[str, str] = {"netcdf": ".nc", "zarr": ".zarr"}


def format_suffix(fmt: OutputFormat) -> str:
    try:
        return _FORMAT_SUFFIX[fmt]
    except KeyError as e:
        raise ValueError(f"unknown output_format: {fmt!r}") from e

_log = get_logger(__name__)


def _open_hypercubes(grib_path: Path) -> list[xr.Dataset]:
    return cfgrib.open_datasets(
        str(grib_path),
        backend_kwargs={"indexpath": ""},
    )


def _ensure_step_dim(ds: xr.Dataset) -> xr.Dataset:
    if "step" in ds.dims:
        return ds
    if "step" in ds.coords:
        return ds.expand_dims("step")
    return ds


# Module-level so a ProcessPoolExecutor can pickle it. Returns a list of
# (vars_sig, dim_sig, eagerly-loaded ds) tuples; ``.load()`` is required
# so the dataset survives the worker exit (lazy cfgrib refs would dangle).
def _decode_one_grib(
    args: tuple[Path, BaseAdapter, tuple[float, float, float, float]],
) -> list[tuple[tuple[str, ...], tuple[str, ...], xr.Dataset]]:
    grib_path, adapter, bbox = args
    out: list[tuple[tuple[str, ...], tuple[str, ...], xr.Dataset]] = []
    for ds in _open_hypercubes(grib_path):
        cropped = adapter.crop_bbox(ds, bbox)
        if cropped.sizes.get("latitude", 0) == 0 or cropped.sizes.get("longitude", 0) == 0:
            continue
        cropped = _ensure_step_dim(cropped)
        for renamed in adapter.rename_hypercube(cropped):
            renamed = renamed.load()
            vars_sig = tuple(sorted(map(str, renamed.data_vars)))
            dim_sig = tuple(sorted(d for d in renamed.dims if d != "step"))
            out.append((vars_sig, dim_sig, renamed))
    return out


def process_init(
    grib_paths: Iterable[Path],
    *,
    adapter: BaseAdapter,
    bbox: tuple[float, float, float, float],
    out_path: Path,
    workers: int = DEFAULT_WORKERS,
    output_format: OutputFormat = "netcdf",
    zarr_options: ZarrOptions | None = None,
) -> list[str]:
    """Process one (date, cycle): crop + time-concat all step files into ``out_path``.

    ``workers`` controls the per-file decode pool. ``workers <= 1`` runs the
    decode loop in-process (useful for debugging); higher values use a
    :class:`ProcessPoolExecutor` to parallelize cfgrib / eccodes which holds
    process-global state and is not thread-safe.

    ``output_format`` picks the writer (``netcdf`` -> single ``.nc`` file,
    ``zarr`` -> ``.zarr`` directory store); ``zarr_options`` is consulted
    only when ``output_format == "zarr"``.

    Returns the sorted list of data-variable names in the written output.
    """
    grib_paths = list(grib_paths)
    if not grib_paths:
        raise ValueError("no GRIB files supplied")

    # Bucket renamed hypercubes by (sorted data_vars, sorted non-step dims).
    buckets: "OrderedDict[tuple[tuple[str, ...], tuple[str, ...]], list[xr.Dataset]]" = OrderedDict()
    workers = max(1, min(workers, len(grib_paths)))
    bar_kwargs = dict(
        desc=f"[{adapter.name}] {out_path.stem}",
        unit="file",
        disable=not sys.stderr.isatty(),
        leave=False,
        total=len(grib_paths),
    )
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_decode_one_grib, (p, adapter, bbox)) for p in grib_paths
            ]
            for fut in tqdm(as_completed(futures), **bar_kwargs):
                for vars_sig, dim_sig, renamed in fut.result():
                    buckets.setdefault((vars_sig, dim_sig), []).append(renamed)
    else:
        for p in tqdm(grib_paths, **bar_kwargs):
            for vars_sig, dim_sig, renamed in _decode_one_grib((p, adapter, bbox)):
                buckets.setdefault((vars_sig, dim_sig), []).append(renamed)

    bucket_dss: list[xr.Dataset] = []
    for (vars_sig, _dim_sig), ds_list in buckets.items():
        merged = xr.concat(
            ds_list, dim="step", coords="minimal", compat="override", join="outer"
        ).sortby("step")
        # ECMWF ships some accumulated/min/max records (e.g. mx2t3 at step=3h)
        # in both the f000 preview and the canonical step file. Keep the last
        # occurrence per step; meteorological semantics make the values
        # identical, but the later forecast pass is the authoritative one.
        if "step" in merged.dims:
            merged = merged.drop_duplicates("step", keep="last")
        # Drop per-bucket valid_time; rebuilt on the merged step axis below
        # (concat with coords='minimal' may already have dropped it).
        merged = merged.drop_vars("valid_time", errors="ignore")
        bucket_dss.append(merged)
        _log.debug(
            "concat_bucket",
            vars=list(vars_sig),
            steps=int(merged.sizes.get("step", 0)),
            lat=int(merged.sizes.get("latitude", 0)),
            lon=int(merged.sizes.get("longitude", 0)),
        )

    final = xr.merge(bucket_dss, compat="no_conflicts", join="outer")
    if "time" in final.coords and "step" in final.coords:
        final = final.assign_coords(valid_time=final["time"] + final["step"])

    write_dataset(
        final, out_path, fmt=output_format, zarr_options=zarr_options,
    )

    variables = sorted(map(str, final.data_vars))
    _log.info(
        "processed_init",
        out_path=str(out_path),
        source=adapter.name,
        variables=len(variables),
        steps=int(final.sizes.get("step", 0)),
        fmt=output_format,
    )
    return variables


def _remove_output(out_path: Path) -> None:
    """Drop a prior output if it exists.

    Files are unlinked; ``.zarr`` directories are removed recursively. Any
    other directory is refused to avoid wiping unrelated content.
    """
    if not out_path.exists():
        return
    if out_path.is_dir():
        if out_path.suffix != ".zarr":
            raise RuntimeError(
                f"refusing to remove non-zarr directory: {out_path}"
            )
        shutil.rmtree(out_path)
    else:
        out_path.unlink()


def _clear_chunk_encoding(ds: xr.Dataset) -> None:
    """cfgrib copies NetCDF chunk hints into ``encoding`` which conflict
    with the explicit per-variable ``chunks`` we set for ``to_zarr``."""
    stale = ("chunks", "preferred_chunks", "contiguous", "original_shape")
    for var in list(ds.variables):
        enc = ds[var].encoding
        for key in stale:
            enc.pop(key, None)


def _resolve_dim_chunks(
    ds: xr.Dataset, chunks: dict[str, int]
) -> dict[str, int]:
    """Map a user-facing dim->chunk dict to concrete chunk sizes.

    ``-1`` and ``0`` collapse to the full dim length; oversized requests are
    clipped to the dim length; dims not present in ``ds`` are silently
    dropped so a single default dict works across products.
    """
    resolved: dict[str, int] = {}
    for dim, want in chunks.items():
        if dim not in ds.dims:
            continue
        size = int(ds.sizes[dim])
        if want is None or want <= 0:
            resolved[dim] = size
        else:
            resolved[dim] = min(int(want), size)
    return resolved


def _build_zarr_encoding(
    ds: xr.Dataset, opts: ZarrOptions
) -> dict[str, dict]:
    """Build a per-variable encoding dict for ``to_zarr`` (zarr v2 style).

    Per-variable ``chunks`` are derived from the dim->size map in ``opts``
    so callers don't need dask installed; dims absent from a variable fall
    through to its full size.
    """
    dim_chunks = _resolve_dim_chunks(ds, opts.chunks)
    if opts.compressor == "none":
        compressor = None
    else:
        from numcodecs import Blosc

        compressor = Blosc(
            cname=opts.compressor, clevel=opts.clevel, shuffle=Blosc.SHUFFLE
        )

    encoding: dict[str, dict] = {}
    for var in ds.data_vars:
        da = ds[var]
        chunk_shape = tuple(
            dim_chunks.get(dim, int(da.sizes[dim])) for dim in da.dims
        )
        spec: dict = {"chunks": chunk_shape}
        if compressor is not None:
            spec["compressor"] = compressor
        encoding[var] = spec
    return encoding


def write_dataset(
    ds: xr.Dataset,
    out_path: Path,
    *,
    fmt: OutputFormat = "netcdf",
    zarr_options: ZarrOptions | None = None,
) -> None:
    """Persist ``ds`` as NetCDF4 or Zarr, replacing any prior output.

    For Zarr the dataset is rechunked according to ``zarr_options.chunks``
    (defaulting to ``ZarrOptions()``), compressed per ``zarr_options``, and
    optionally consolidated for fast metadata reads.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_output(out_path)

    if fmt == "netcdf":
        ds.to_netcdf(out_path, engine="netcdf4")
        return

    if fmt == "zarr":
        opts = zarr_options or ZarrOptions()
        _clear_chunk_encoding(ds)
        encoding = _build_zarr_encoding(ds, opts)
        ds.to_zarr(
            out_path,
            mode="w",
            consolidated=opts.consolidated,
            zarr_format=opts.zarr_format,
            encoding=encoding,
        )
        # 写后 chunk 完整性校验:截断(部分 chunk 没落盘)→ 删库 + 报错,
        # 让上层重切,绝不把损坏库留成"已完成"(下游读到静默 NaN 或 500)。
        truncated = _verify_zarr_chunks(out_path)
        if truncated:
            _remove_output(out_path)
            raise ChunkIntegrityError(
                f"{out_path.name} zarr 写入截断(部分 chunk 缺失),已删除待重切: "
                + "; ".join(truncated)
            )
        return

    raise ValueError(f"unknown output_format: {fmt!r}")


class ChunkIntegrityError(RuntimeError):
    """新写的 zarr 库有数组截断:元数据说该有 N 个 chunk,磁盘只写出 M<N 个
    (静默 chunk 写失败 / 写入中断)。库被删除以便重切,不留成损坏的"已完成"产品。"""


def _verify_zarr_chunks(store: Path) -> list[str]:
    """返回 ["<数组>: <写出>/<应有>", ...] —— 写出 chunk 数 **少于** 元数据蕴含
    数量(截断)的数组。全 fill 数组(写出=0)不报(可能合法);只报**部分写入**
    (0<写出<应有):对中国区空间稠密的预报场,这意味着写没完成。step 是单 chunk,
    故稀疏 step 维不会跳 chunk、不会误报。仅 consolidated(有 .zmetadata)时校验。"""
    zm = store / ".zmetadata"
    if not zm.is_file():
        return []
    try:
        meta = json.loads(zm.read_text())["metadata"]
    except Exception:
        return []
    bad: list[str] = []
    for key, za in meta.items():
        if not key.endswith("/.zarray"):
            continue
        arr = key[: -len("/.zarray")]
        shape, chunks = za.get("shape"), za.get("chunks")
        if not shape or not chunks:
            continue
        expected = 1
        for s, c in zip(shape, chunks):
            expected *= max(1, math.ceil(s / c))
        d = store / arr
        if not d.is_dir():
            continue
        got = sum(1 for f in d.iterdir() if not f.name.startswith("."))
        if 0 < got < expected:
            bad.append(f"{arr}: {got}/{expected}")
    return bad
