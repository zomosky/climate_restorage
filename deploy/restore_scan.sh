#!/usr/bin/env sh
# restore_scan.sh — host-side cron entry for the climate_restore slicing step.
#
# Idempotently sweeps every *.manifest.json under the download data root *inside
# the long-running dev container* and turns each finished init into a per-source
# Zarr store. Safe to run on any schedule: already-built outputs are skipped
# (keyed off the on-disk .zarr), so re-runs are cheap and crash-safe — exactly
# what `scan-once` is for (unlike `watch`, whose in-memory "seen" set drops any
# manifest that appears while it is down).
#
# Install (host crontab), every 10 minutes:
#   */10 * * * * /srv/climate/restore_scan.sh >> /var/log/climate/restore_scan.log 2>&1
#
# Prerequisites:
#   - the dev container is running and stays up (cron drives it via `docker exec`)
#   - restore deps synced once inside it:
#       docker exec -w /workspace/climate_restorage zhangmy-dev uv sync
#
# Any value below can be overridden from the environment if your layout differs.
set -eu

CONTAINER="${CONTAINER:-zhangmy-dev}"
RESTORE_DIR="${RESTORE_DIR:-/workspace/climate_restorage}"
DATA_ROOT="${DATA_ROOT:-/climate_data}"
OUT_DIR="${OUT_DIR:-/climate_data/climate_data_storage/zarr}"
JOB="${JOB:-config/jobs/gfs_china.yaml}"
LOCK="${LOCK:-/tmp/restore_scan.lock}"

# Skip quietly (exit 0) if the container is down — scan-once is idempotent, so
# the next tick after it comes back processes the whole backlog; no data is lost.
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "restore_scan: container '$CONTAINER' not running; skipping this tick." >&2
  exit 0
fi

# flock (inside the container) stops a slow scan from overlapping the next cron
# tick: two in-place .zarr rewrites of the same init would corrupt it. `-E 0`
# makes "lock already held" a clean exit 0 rather than a spurious failure
# (needs util-linux >= 2.27; drop `-E 0` if your container's flock is older).
exec docker exec -w "$RESTORE_DIR" "$CONTAINER" \
  flock -n -E 0 "$LOCK" \
  uv run climate_restore scan-once \
    --download-root "$DATA_ROOT" \
    --output-dir    "$OUT_DIR" \
    --output-format zarr \
    --config "$JOB"
