#!/bin/bash
# forecast-coldmove.sh — 把超过保留窗(默认 30 天)的预报原始 GRIB 逐 init 挪到
# /save_data 冷备。zarr 是下游唯一消费的产品(留在热端不动),原始 GRIB 只是
# 重建原料——某 init 的 zarr 已固化(.zmetadata)后其原始档即可冷存。
#
# 结构:热端 /training_data/climate_data/{源}/{YYYYMMDD}/{NN}z/*.grib*
#       zarr /training_data/climate_data/climate_data_storage/zarr/{源}/{date}_{NN}z_{源}[-part].zarr
# 判定:一个 {date} 目录整体挪走,前提是它下面**每个** {NN}z 都有对应且全部
#       固化的 zarr(分部 -part 也要各自 .zmetadata);任一 cycle 的 zarr 没好
#       就整天跳过(原始还得留着重切)。
# 概率源另验 manifest、原始 SHA-256、全部变量/窗口及数据分块可读性;
#       复制后逐文件 SHA-256 一致才删热端,与概率下载共用锁。Zarr 始终不移动。
# 复制:tar 管道流式(顺序写,免 rsync 逐文件 fsync/rename 开销;实测 NVMe→机械盘
#       页缓存吸收可达数 GB/s、稳态受 HDD 回写约 200MB/s)。
#       ⚠️ 本机 I/O 调度器是 none/mq-deadline,**不要用 ionice**——blk-mq 仍认
#       IOPRIO_CLASS_IDLE,-c3 会把 tar 饿到 ~2MB/s(实测踩过坑)。
# 权限:原始 GRIB 由 zhangmy-dev 容器(root)写,复制用宿主机(读 644 源、写
#       /save_data),删除借容器 root(容器挂 /training_data/climate_data→/climate_data)。
# 幂等:重复运行安全;复制/核验/删除任一步不过只报错保留源(tar 覆盖式重解,
#       上一轮中断的半拉冷端副本会被补全)。
#
# 安装(crontab,每天 07:00,躲开下载/重整高峰):
#   0 7 * * * /bin/bash /home/zhangmingyu/operation/forecast-coldmove.sh
# 手动演练: DRY_RUN=1 bash forecast-coldmove.sh
set -u

SOURCES="${SOURCES:-gfs-0p25 ifs-hres aifs-single dwd-icon ifs-ens-probabilities aifs-ens-probabilities}"
KEEP_DAYS="${KEEP_DAYS:-30}"
HOT="${HOT:-/training_data/climate_data}"
COLD="${COLD:-/save_data/climate_data}"
ZARR="${ZARR:-/training_data/climate_data/climate_data_storage/zarr}"
CONTAINER="${CONTAINER:-zhangmy-dev}"
CHOT="${CHOT:-/climate_data}"   # 容器内热端根(删除用)
CZARR="${CZARR:-/climate_data/climate_data_storage/zarr}"
CCOLD="${CCOLD:-$COLD}"         # /save_data 在容器内外路径相同
RESTORE_DIR="${RESTORE_DIR:-/workspace/climate_restorage}"
PROBABILITY_LOCK="${PROBABILITY_LOCK:-/home/zhangmingyu/operation/lock/ens_probabilities.lock}"
LOCK="${LOCK:-/home/zhangmingyu/operation/lock/forecast_coldmove.lock}"
LOG="${LOG:-/home/zhangmingyu/operation/logs/forecast-coldmove.log}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$(dirname "$LOCK")" "$(dirname "$LOG")"
[ "$DRY_RUN" = "1" ] || exec >>"$LOG" 2>&1
ts() { date '+%F %T'; }
probability_check() {
  docker exec -w "$RESTORE_DIR" "$CONTAINER" env \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    uv run --no-sync python deploy/probability_coldmove_check.py \
    --raw-root "$CHOT" --zarr-root "$CZARR" --source "$s" --date "$d" "$@"
}

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(ts)] 上一轮仍在运行,跳过"
  exit 0
fi

cutoff=$(date -u -d "${KEEP_DAYS} days ago" +%Y%m%d)
echo "[$(ts)] START keep=${KEEP_DAYS}d cutoff=$cutoff(date < cutoff 的 init 挪 /save_data)dry_run=$DRY_RUN"

rc=0
for s in $SOURCES; do
  [ -d "$HOT/$s" ] || continue
  probability=0
  case "$s" in
    ifs-ens-probabilities|aifs-ens-probabilities)
      probability=1
      mkdir -p "$(dirname "$PROBABILITY_LOCK")"
      exec 8>"$PROBABILITY_LOCK"
      if ! flock -n 8; then
        echo "[$(ts)] 跳过 $s: 概率下载/处理正在运行"
        exec 8>&-
        continue
      fi
      ;;
  esac
  for dir in "$HOT/$s"/2*/; do
    [ -d "$dir" ] || continue
    d=$(basename "$dir")
    case "$d" in 2[0-9][0-9][0-9][0-9][0-9][0-9][0-9]) ;; *) continue ;; esac
    [ "$d" \< "$cutoff" ] || continue

    # 该 init 各 cycle 的 zarr 是否都固化
    safe=1
    # Probability history has a richer gate below: a proven upstream-no-data
    # receipt may legitimately have indexes and no probability GRIB/Zarr.
    if [ "$probability" != "1" ]; then
    for cyc in "$dir"*z; do
      [ -d "$cyc" ] || continue
      nn=$(basename "$cyc"); nn=${nn%z}
      zn=0; zdone=0
      for z in "$ZARR/$s/${d}_${nn}z_${s}"*.zarr; do
        [ -d "$z" ] || continue
        zn=$((zn+1)); [ -f "$z/.zmetadata" ] && zdone=$((zdone+1))
      done
      if [ "$zn" -eq 0 ] || [ "$zdone" -lt "$zn" ]; then safe=0; break; fi
    done
    fi
    if [ "$safe" != 1 ]; then
      echo "[$(ts)] 跳过 $s/$d: 有 cycle 的 zarr 未固化,原始先留着"
      continue
    fi
    if [ "$probability" = "1" ] && ! probability_check; then
      echo "[$(ts)] 跳过 $s/$d: 概率产品完整性核验失败,原始先留着"
      rc=1
      continue
    fi

    src_n=$(find "$dir" -type f | wc -l)
    if [ "$DRY_RUN" = "1" ]; then
      echo "[$(ts)] (演练) 将挪 $s/$d($src_n 文件)→ $COLD/$s/$d,删除经容器 $CONTAINER:$CHOT/$s/$d"
      continue
    fi
    mkdir -p "$COLD/$s/$d"
    tar -C "$dir" -cf - . | tar -C "$COLD/$s/$d" -xpf -
    st=("${PIPESTATUS[@]}")
    if [ "${st[0]}" -ne 0 ] || [ "${st[1]}" -ne 0 ]; then
      echo "[$(ts)] tar 复制失败 $s/$d(读 ${st[0]}/写 ${st[1]}),保留源"; rc=1; continue
    fi
    dst_n=$(find "$COLD/$s/$d" -type f | wc -l)
    if [ "$dst_n" -ge "$src_n" ]; then
      if [ "$probability" = "1" ] && ! probability_check --cold-root "$CCOLD"; then
        echo "[$(ts)] 冷端校验失败 $s/$d,保留热端原始文件"
        rc=1
        continue
      fi
      if docker exec "$CONTAINER" rm -rf "$CHOT/$s/$d"; then
        echo "[$(ts)] moved $s/$d($src_n 文件)"
      else
        echo "[$(ts)] 冷端就绪但容器删热端失败 $s/$d,保留源"; rc=1
      fi
    else
      echo "[$(ts)] 核验失败 $s/$d(源 $src_n vs 冷端 $dst_n),保留源"; rc=1
    fi
  done
  if [ "$probability" = "1" ]; then
    flock -u 8
    exec 8>&-
  fi
done
echo "[$(ts)] DONE rc=$rc"
exit "$rc"
