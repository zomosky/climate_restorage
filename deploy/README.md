# deploy/ — scheduling the restore (slicing → Zarr) step

`restore_scan.sh` is a **host-side cron entry** that drives
`climate_restore scan-once` *inside the long-running dev container*, turning
every newly-downloaded init under the shared data root into a per-source Zarr
store. It mirrors the existing "host crontab → `.sh` → `docker exec`" pattern
used for the download jobs.

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
| `RESTORE_DIR` | `/workspace/climate_restorage`              | restore project dir inside the container  |
| `DATA_ROOT`   | `/climate_data`                             | parent of all source subtrees to scan     |
| `OUT_DIR`     | `/climate_data/climate_data_storage/zarr`   | where `<source>/<init>.zarr` is written   |
| `JOB`         | `config/jobs/gfs_china.yaml`                | bbox / workers / zarr knobs (shared)      |
| `LOCK`        | `/tmp/restore_scan.lock`                    | flock path (in-container)                 |

`DATA_ROOT=/climate_data` is the common parent of `gfs-0p25`, `ifs-hres`,
`graphcast-history`, and `aifs-single`, so one scan handles all four — the
adapter for each init is chosen from its manifest's `source.name`.

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
    --output-format zarr --config config/jobs/gfs_china.yaml
```

Check the closing `scan_done` log line: `processed / skipped / not_ready / failed`.
