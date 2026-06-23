"""batch_restore.py — 批量拼接气象源已下载文件的独立运行脚本。

用法：
    python batch_restore.py          # 使用脚本内 CONFIG 中的设置
    python batch_restore.py --dry-run  # 只打印待处理 manifest，不实际执行

所有配置集中在脚本顶部的 "====== 用户配置区 ======" 内修改。
可选：指定一个 job YAML 作为基底配置，脚本内字段若非 None 则覆盖 YAML 值。
"""

from __future__ import annotations

# ============================================================
# ====== 用户配置区（所有参数在此修改）========================
# ============================================================

# --- 基底 Job YAML（可选，设为 None 则完全使用下方参数）------
# 例：CONFIG_YAML = "config/jobs/ifs_china.yaml"
# CONFIG_YAML: str | None = "/Users/zmy/pycharm/climate_pipeline/climate_restorage/config/jobs/ifs_zarr_china.yaml"
CONFIG_YAML: str | None = "/Users/zmy/pycharm/climate_pipeline/climate_restorage/config/jobs/gfs_china.yaml"

# --- 气象源根目录（已下载数据的 download project root）--------
# manifest 路径格式：<DOWNLOAD_ROOT>/output/<source>/<date>/<cycle>z/...manifest.json
# 设为 None 则使用 YAML 中的 download_root
DOWNLOAD_ROOT: str | None = "/Users/zmy/pycharm/climate_pipeline/climate_data/"

# --- 输出目录（.nc / .zarr 写入位置）--------------------------
# 设为 None 则使用 YAML 中的 output_dir
OUTPUT_DIR: str | None = "/Users/zmy/pycharm/climate_pipeline/climate_data_storage/zarr"

# --- 输出格式（None = 使用 YAML 中的 output_format，缺省 netcdf）
# "netcdf" -> 单个 .nc 文件；"zarr" -> .zarr 目录 store
OUTPUT_FORMAT: str | None = "zarr"

# --- Zarr chunks 覆盖（None = 使用 YAML / 内置默认）-----------
# 例：ZARR_CHUNKS = {"step": -1, "latitude": 64, "longitude": 64}
ZARR_CHUNKS: dict[str, int] | None = None

# --- 时间范围过滤（YYYYMMDD 字符串，包含两端）-----------------
DATE_START: str = "20260201"   # 起始日期
DATE_END:   str = "20260206"   # 截止日期（含）

# --- 起报周期过滤（None = 不过滤；例如只取 [0, 12] UTC 起报）--
# 例：CYCLES: list[int] | None = [0, 12]
CYCLES: list[int] | None = None

# --- 数据源名称过滤（None = 不过滤；例如只处理某一源）----------
# 例：SOURCE_FILTER: str | None = "gfs-0p25"
# SOURCE_FILTER: str | None = "graphcast-history"
SOURCE_FILTER: str | None = "gfs-0p25"

# --- 强制使用某个 adapter（None = 自动从 manifest.source.name 推断）
SOURCE_TYPE_OVERRIDE: str | None = None

# --- 并行解码进程数（覆盖 YAML；1 = 关闭进程池，方便调试）------
WORKERS: int = 10

# --- 是否同时验证 SHA-256（慢，默认只验 size）------------------
VERIFY_SHA256: bool = False

# --- 裁剪框 (west, east, south, north)，None = 使用 YAML / 内置默认
BBOX: tuple[float, float, float, float] | None = None  # e.g. (70.0, 140.0, 15.0, 55.0)

# --- 跳过已存在输出文件（True = 断点续跑）----------------------
SKIP_EXISTING: bool = True

# --- 日志级别 --------------------------------------------------
LOG_LEVEL: str = "INFO"

# ============================================================
# ====== 以下为执行逻辑，通常无需修改 =========================
# ============================================================

import os
import sys
from pathlib import Path

# ── 自动切换到 .venv Python（避免 Anaconda/系统 Python 与 venv 包冲突）──────
# 如果当前解释器不是本项目 .venv 内的 Python，则用 .venv/bin/python 重启自身。
_REPO = Path(__file__).resolve().parent
_VENV_PY = _REPO / ".venv" / "bin" / "python"
if _VENV_PY.is_file() and not str(_REPO / ".venv").startswith(sys.prefix):
    os.execv(str(_VENV_PY), [str(_VENV_PY)] + sys.argv)

# ── 把 src/ 加入搜索路径（editable install 替代）────────────────────────────
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
from datetime import datetime


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%Y%m%d")


def _out_path_for(manifest, output_dir: Path, suffix: str) -> Path:
    fname = f"{manifest.date}_{manifest.cycle:02d}z_{manifest.source_name}{suffix}"
    return (output_dir / manifest.source_name / fname).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description="批量 climate_restore 拼接工具")
    ap.add_argument("--dry-run", action="store_true", help="只列出待处理 manifest，不执行")
    ap.add_argument("--log-level", default=LOG_LEVEL)
    args = ap.parse_args()

    # --- 初始化日志 ---
    from climate_restore.logging_setup import configure_logging
    configure_logging(args.log_level)
    import structlog
    log = structlog.get_logger("batch_restore")

    # --- 加载基底配置 ---
    from climate_restore.config import JobConfig, load_job
    job: JobConfig = load_job(Path(CONFIG_YAML)) if CONFIG_YAML else JobConfig()

    # --- 覆盖配置 ---
    download_root = Path(DOWNLOAD_ROOT).resolve() if DOWNLOAD_ROOT else job.download_root.resolve()
    output_dir    = Path(OUTPUT_DIR).resolve()    if OUTPUT_DIR    else job.output_dir.resolve()
    bbox          = BBOX if BBOX is not None else job.bbox
    workers       = WORKERS
    verify_sha256 = VERIFY_SHA256
    output_format = OUTPUT_FORMAT or job.output_format
    zarr_options  = job.zarr.model_copy(
        update={"chunks": ZARR_CHUNKS}) if ZARR_CHUNKS is not None else job.zarr

    from climate_restore.processor import format_suffix
    out_suffix = format_suffix(output_format)

    date_start = _parse_date(DATE_START)
    date_end   = _parse_date(DATE_END)

    log.info("batch_start",
             download_root=str(download_root),
             output_dir=str(output_dir),
             date_start=DATE_START, date_end=DATE_END,
             source_filter=SOURCE_FILTER, cycles=CYCLES,
             workers=workers, skip_existing=SKIP_EXISTING,
             output_format=output_format)

    # --- 发现 manifest ---
    from climate_restore.watcher import discover_manifests
    all_manifests = discover_manifests(download_root, source=SOURCE_FILTER)
    log.info("discovered", total=len(all_manifests), search_root=str(download_root))

    # --- 加载并过滤 ---
    from climate_restore.manifest import ManifestHasFailures, load_manifest, verify
    from climate_restore.sources import get_source
    from climate_restore.processor import process_init

    candidates: list[Path] = []
    for mp in all_manifests:
        try:
            m = load_manifest(mp)
        except Exception as exc:
            log.warning("manifest_load_error", path=str(mp), error=str(exc))
            continue
        if m.completed_at is None:
            log.debug("skip_incomplete", path=str(mp))
            continue
        try:
            mdate = _parse_date(m.date)
        except ValueError:
            log.warning("skip_bad_date", path=str(mp), date=m.date)
            continue
        if not (date_start <= mdate <= date_end):
            continue
        if CYCLES is not None and m.cycle not in CYCLES:
            continue
        candidates.append(mp)

    candidates.sort()
    log.info("filtered", count=len(candidates))

    if args.dry_run:
        for mp in candidates:
            print(mp)
        return 0

    # --- 批量处理 ---
    ok = failed = skipped = with_failures = 0
    for mp in candidates:
        try:
            m = load_manifest(mp)
            out_path = _out_path_for(m, output_dir, out_suffix)
            if SKIP_EXISTING and out_path.exists():
                log.info("skip_existing", out=str(out_path))
                skipped += 1
                continue

            source_key = SOURCE_TYPE_OVERRIDE or m.source_name
            adapter_cls = get_source(source_key)
            adapter = adapter_cls(name=source_key)

            try:
                grib_paths = verify(m, download_root, check_sha256=verify_sha256)
            except ManifestHasFailures as exc:
                log.warning("skip_manifest_failures", manifest=str(mp),
                            date=m.date, cycle=m.cycle,
                            failures_count=len(exc.failures),
                            failures=exc.failures)
                with_failures += 1
                continue

            log.info("processing", manifest=str(mp),
                     date=m.date, cycle=m.cycle, files=len(grib_paths))

            process_init(grib_paths, adapter=adapter,
                         bbox=bbox, out_path=out_path, workers=workers,
                         output_format=output_format, zarr_options=zarr_options)
            log.info("done", out=str(out_path))
            ok += 1

        except Exception as exc:
            log.error("process_failed", manifest=str(mp), error=str(exc))
            failed += 1

    log.info("batch_done", ok=ok, skipped=skipped,
             failed=failed, failures=with_failures)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
