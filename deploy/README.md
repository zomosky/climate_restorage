# deploy/ — scheduling the restore (slicing → Zarr) step

`restore_scan.sh` is a **host-side cron entry** that drives
`climate_restore scan-once` *inside the long-running dev container*, turning
every newly-downloaded init under the shared data root into a per-source Zarr
store. It mirrors the existing "host crontab → `.sh` → `docker exec`" pattern
used for the download jobs.

**Lock placement:** `flock` runs **on the host** and wraps `docker exec`, so the
lock file lives on the host (not inside the container). Two overlapping cron
ticks can't stack two scans while the host `docker exec` client is attached (it
stays up until the in-container run exits). All other paths (`RESTORE_DIR`,
`DATA_ROOT`, `OUT_DIR`, `JOB`) are **container-internal**; only `LOCK` is a host
path.

## Why `scan-once` (not `watch`)

`scan-once` makes one idempotent pass and exits, so it fits `docker exec` from
cron perfectly. Its idempotency key is the **on-disk output**: any init whose
`.zarr` is already complete is skipped, everything else is processed. A `watch`
loop instead keeps its "already seen" set in memory and re-seeds it from disk on
restart, so manifests that appear while it is down are skipped forever —
`scan-once` has no such gap and survives container restarts.

## Prerequisites

- The dev container (`zhangmy-dev`) is running and **stays up** — cron reaches
  it via `docker exec`.
- The cron user can run `docker` (member of the `docker` group, or root) —
  otherwise `docker ps` / `docker exec` fail with permission denied.
- Restore deps synced once inside it:
  ```sh
  docker exec -w /workspace/climate_restorage zhangmy-dev uv sync
  ```

## Install

1. Copy the script to the host and make it executable:
   ```sh
   install -m755 restore_scan.sh /srv/climate/restore_scan.sh
   ```
2. Add to the host crontab (`crontab -e`):
   ```cron
   */10 * * * * /srv/climate/restore_scan.sh >> /var/log/climate/restore_scan.log 2>&1
   ```

## Configuration (environment-overridable)

Defaults match the `zhangmy-dev` container; override per host as needed.

| var           | default                                     | meaning                                   |
|---------------|---------------------------------------------|-------------------------------------------|
| `CONTAINER`   | `zhangmy-dev`                               | dev container name                        |
| `RESTORE_DIR` | `/workspace/climate_restorage`              | restore project dir **inside the container** |
| `DATA_ROOT`   | `/climate_data`                             | **inside**: parent of all source subtrees to scan |
| `OUT_DIR`     | `/climate_data/climate_data_storage/zarr`   | **inside**: where `<source>/<init>.zarr` is written |
| `JOB`         | `config/jobs/gfs_zarr_china.yaml`           | **inside** (relative to `RESTORE_DIR`): bbox / workers / zarr knobs (shared, source-agnostic) |
| `SOURCE`      | *(empty)*                                   | empty = scan all sources; e.g. `gfs-0p25` to scope one subtree |
| `LOCK`        | `/tmp/restore_scan.lock`                    | flock path **on the host** (its parent dir is auto-created) |

`DATA_ROOT=/climate_data` is the common parent of `gfs-0p25`, `ifs-hres`,
`graphcast-history`, and `aifs-single`, so one scan handles all four — the
adapter for each init is chosen from its manifest's `source.name`.

**The `JOB` config does NOT select a source** — it only carries bbox / workers /
zarr knobs, which are shared across sources. So `JOB=…/gfs_zarr_china.yaml` still
processes ifs/aifs/graphcast too; the filename is just a label. To restrict to
one source, set `SOURCE` (→ `--source`), which is the *only* thing that narrows
the scan to a single subtree. For genuinely per-source knobs, run one instance
per source, pairing `SOURCE=<name>` with its own `JOB=<name>.yaml`.

## Behaviour & exit codes

- **Idempotent**: completed Zarr stores are skipped (a consolidated store is
  "done" once its `.zmetadata` exists, so a crash mid-write is re-processed, not
  mistaken for complete).
- **Not-ready** manifests (download still running: no `completed_at`, or
  recorded `failures`) are left for a later tick — not an error.
- **Exit code** `0` if everything succeeded or was skipped (including the
  container being down, and a concurrent run holding the lock); `1` if at least
  one *ready* manifest failed to process — alert on that in the log.

## Run it once by hand (don't wait for cron)

```sh
docker exec -w /workspace/climate_restorage zhangmy-dev \
  uv run climate_restore scan-once \
    --download-root /climate_data \
    --output-dir    /climate_data/climate_data_storage/zarr \
    --output-format zarr --config config/jobs/gfs_zarr_china.yaml
```

Check the closing `scan_done` log line: `processed / skipped / not_ready / failed`.

## 预报原始数据 30 天转冷存

`forecast-coldmove.sh` 是服务器现有冷存脚本的版本化副本。每天北京时间 07:00
由宿主机 crontab 执行，按 UTC 日期判断 `date < today_utc - 30 days`。
默认覆盖 `gfs-0p25`、`ifs-hres`、`aifs-single`、`dwd-icon`、
`ifs-ens-probabilities`、`aifs-ens-probabilities`。

原始文件及 manifest 从 `/training_data/climate_data/<source>/<date>/` 转至
`/save_data/climate_data/<source>/<date>/`；Zarr 始终保留在
`/training_data/climate_data/climate_data_storage/zarr/<source>/`，查询路径不变。

普通源沿用原有完成检查；两个概率源使用 `probability_coldmove_check.py` 额外检查：

1. 当天每个已下载 cycle 的 manifest 均已完成、无失败，并包含 240/360 两包；
   原始文件大小及 SHA-256 与清单一致。
2. Zarr 不早于 manifest，概率验收属性完整；各概率变量的所有物理分块存在，
   全量读取可解码，字段、原生时刻、0–100 范围和窗口起止通过概率适配器校验。
   坐标可以合法地使用 Zarr 的默认填充值省略分块，按实际解码结果校验。
3. 复制后再次校验产品，并逐文件比较热端和冷端的路径集合与 SHA-256；一致才
   删除热端原始日期目录。任一步失败保留热端，记录日志并退出非零。

概率源冷存与 `probabilities_run.py` 共用
`/home/zhangmingyu/operation/lock/ens_probabilities.lock`，下载/处理运行时跳过冷存。
原有冷存全局锁与 `KEEP_DAYS=30` 保持不变。

安装时先将校验器放入容器挂载的项目目录，再安装宿主机脚本：

```sh
install -m 0644 deploy/probability_coldmove_check.py \
  /home/zhangmingyu/dev-docker/project/climate_restorage/deploy/probability_coldmove_check.py
install -m 0755 deploy/forecast-coldmove.sh /home/zhangmingyu/operation/forecast-coldmove.sh
```

只读演练（不复制、不删除气象数据；仍检查符合保留窗的产品）：

```sh
SOURCES='ifs-ens-probabilities aifs-ens-probabilities' DRY_RUN=1 \
  bash /home/zhangmingyu/operation/forecast-coldmove.sh
```

没有超过 30 天的数据时演练不会触发迁移。可在容器内对某个现有日期直接执行
`uv run --no-sync python deploy/probability_coldmove_check.py --raw-root /climate_data
--zarr-root /climate_data/climate_data_storage/zarr --source ifs-ens-probabilities
--date YYYYMMDD`，该校验器自身只读。

日志为 `/home/zhangmingyu/operation/logs/forecast-coldmove.log`。
新增路径变量 `CZARR`、`CCOLD` 分别用于容器内的 Zarr、冷端根目录；`CHOT`
是容器内原始根目录。默认对应现有挂载，不允许把生产 Zarr 改为冷存目标。
回滚恢复备份的宿主机脚本即可；校验器不修改数据，也不需要重建服务镜像。
