#!/usr/bin/env python3
"""Generate manifest files for historical GFS data downloaded by gfsdown.

This script scans existing gfsdown output directories and generates
climate_restore-compatible manifest.json files for batch processing.

Usage:
    # Generate manifests for all data in output directory
    python generate_manifest.py

    # Generate for specific date range
    python generate_manifest.py --date-range 2026-01-01:2026-02-01

    # Generate for specific date
    python generate_manifest.py --date 2026-01-01

    # Dry-run mode (show what would be generated)
    python generate_manifest.py --dry-run

    # Compute SHA-256 hashes (slow but more accurate)
    python generate_manifest.py --compute-sha256
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gfsdown.manifest import discover_dates, generate_manifest, write_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> str:
    """Parse date string to YYYY-MM-DD format."""
    # Try YYYY-MM-DD
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass

    # Try YYYYMMDD
    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass

    raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD or YYYYMMDD")


def parse_date_range(date_range_str: str) -> tuple[str, str]:
    """Parse date range string: START:END or START:END:STEP."""
    parts = date_range_str.split(":")
    if len(parts) == 2:
        start, end = parts
    elif len(parts) == 3:
        start, end = parts[0], parts[1]
        # step is ignored for now
    else:
        raise ValueError(f"Invalid date range format: {date_range_str}. Use START:END or START:END:STEP")

    return parse_date(start), parse_date(end)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate manifest files for historical gfsdown data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="GFS data output directory (default: output)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="gfs-0p25",
        help="Source name (default: gfs-0p25)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Generate manifest for specific date (YYYY-MM-DD or YYYYMMDD)",
    )
    parser.add_argument(
        "--date-range",
        type=str,
        default=None,
        metavar="START:END",
        help="Generate manifests for date range (e.g., 2026-01-01:2026-02-01)",
    )
    parser.add_argument(
        "--cycle",
        type=int,
        default=None,
        choices=[0, 6, 12, 18],
        help="Generate manifest for specific cycle hour (0, 6, 12, or 18)",
    )
    parser.add_argument(
        "--compute-sha256",
        action="store_true",
        help="Compute SHA-256 hashes for each file (slow)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing files",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    source_name = args.source

    # Determine which dates to process
    if args.date:
        date_str = parse_date(args.date)
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        date_yyyymmdd = date_obj.strftime("%Y%m%d")
        cycles = [args.cycle] if args.cycle is not None else [0, 6, 12, 18]
        dates_to_process = [(date_yyyymmdd, c) for c in cycles]
    elif args.date_range:
        start_date, end_date = parse_date_range(args.date_range)
        start_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_obj = datetime.strptime(end_date, "%Y-%m-%d")

        dates_to_process = []
        current = start_obj
        while current <= end_obj:
            date_yyyymmdd = current.strftime("%Y%m%d")
            cycles = [args.cycle] if args.cycle is not None else [0, 6, 12, 18]
            dates_to_process.extend([(date_yyyymmdd, c) for c in cycles])
            current = current.replace(day=current.day + 1) if current.day < 28 else \
                      current.replace(month=current.month + 1, day=1) if current.month < 12 else \
                      current.replace(year=current.year + 1, month=1, day=1)

            # Use timedelta for proper date arithmetic
            from datetime import timedelta
            current = start_obj + timedelta(days=(current - start_obj).days + 1)
            if current > end_obj:
                break
    else:
        # Discover all available dates
        logger.info(f"Discovering data in {output_dir / source_name}...")
        dates_to_process = discover_dates(output_dir, source_name)

    if not dates_to_process:
        logger.error("No data found to process")
        return 1

    logger.info(f"Found {len(dates_to_process)} (date, cycle) combinations to process")

    generated = 0
    skipped = 0
    errors = 0

    for date_str, cycle in dates_to_process:
        date_yyyymmdd = date_str.replace("-", "") if "-" in date_str else date_str
        cycle_dir = output_dir / source_name / date_yyyymmdd / f"{cycle:02d}z"
        manifest_path = cycle_dir / f"{date_yyyymmdd}_{cycle:02d}z_{source_name}.manifest.json"

        if not cycle_dir.exists():
            logger.debug(f"Skipping {date_yyyymmdd} {cycle:02d}Z: directory not found")
            continue

        if manifest_path.exists():
            logger.info(f"Manifest already exists: {manifest_path}")
            skipped += 1
            continue

        logger.info(f"Processing: {date_yyyymmdd} {cycle:02d}Z")
        logger.info(f"  Directory: {cycle_dir}")
        logger.info(f"  Manifest: {manifest_path}")

        if args.dry_run:
            logger.info(f"  [DRY-RUN] Would generate manifest")
            generated += 1
            continue

        try:
            manifest = generate_manifest(
                output_dir=output_dir,
                date_str=date_yyyymmdd,
                cycle=cycle,
                source_name=source_name,
                compute_hash=args.compute_sha256,
                include_source_in_path=True,  # Data is in output_dir/source_name/date/cycle
            )

            write_manifest(manifest, manifest_path)
            logger.info(f"  ✓ Generated manifest with {len(manifest['files'])} files")
            generated += 1

        except Exception as exc:
            logger.error(f"  ✗ Error: {exc}")
            errors += 1

    logger.info(f"\nSummary:")
    logger.info(f"  Generated: {generated}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Errors: {errors}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
