"""DWD ICON icosahedral → regular lat/lon nearest-neighbour remap.

ICON global GRIB is on an **unstructured icosahedral grid**: each message is a
1-D array of ``values`` indexed by cell (R3B07 ≈ 2.95M cells), and the cell
centre lat/lon are **not** in the data message — they live in DWD's separate
time-invariant ``CLAT`` / ``CLON`` files. This module:

1. downloads + decodes those cell centres once (cached as ``.npz``);
2. builds a nearest-neighbour remap from the cells to a regular lat/lon target
   grid over a bbox, using a KD-tree on unit-sphere xyz (correct across the
   dateline / poles), caching the per-(bbox,res) index.

``remap_values(values, remap)`` then turns any ``(…, values)`` array into a
regular ``(…, lat, lon)`` array by fancy-indexing — cheap once the index exists.
"""

from __future__ import annotations

import bz2
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from climate_restore.logging_setup import get_logger

_log = get_logger(__name__)

# Cell count of the operational global grid (R3B07). Used only as a sanity gate.
_R3B07_CELLS = 2_949_120
_DWD_BASE = "https://opendata.dwd.de/weather/nwp/icon/grib"


def cache_dir() -> Path:
    """Where decoded coords + remap indices are cached (env-overridable)."""
    root = os.environ.get("CLIMATE_RESTORE_CACHE")
    d = Path(root) if root else (Path.home() / ".cache" / "climate_restore")
    d = d / "icon"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass(frozen=True)
class Remap:
    """A built nearest-neighbour remap onto a regular grid."""

    lats: np.ndarray          # (nlat,) ascending target latitudes
    lons: np.ndarray          # (nlon,) ascending target longitudes
    index: np.ndarray         # (nlat*nlon,) source cell index for each target point

    @property
    def shape(self) -> tuple[int, int]:
        return (self.lats.size, self.lons.size)


# --- cell coordinates ------------------------------------------------------

def _decode_grib_values(path: Path) -> np.ndarray:
    """Read the single message's ``values`` array from a GRIB file."""
    from eccodes import (
        codes_grib_new_from_file,
        codes_get_array,
        codes_release,
    )

    with open(path, "rb") as fh:
        gid = codes_grib_new_from_file(fh)
        if gid is None:
            raise ValueError(f"no GRIB message in {path}")
        try:
            return np.asarray(codes_get_array(gid, "values"), dtype="f8")
        finally:
            codes_release(gid)


def _http_get(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (trusted host)
        return resp.read()


def _find_clatclon_urls() -> tuple[str, str]:
    """Locate a currently-published CLAT/CLON pair on the DWD server.

    The files are time-invariant (same grid every run), so any available cycle
    works; we scan cycles for the first one that has both.
    """
    for cc in ("00", "06", "12", "18"):
        urls = {}
        for var in ("clat", "clon"):
            try:
                text = _http_get(f"{_DWD_BASE}/{cc}/{var}/", timeout=30.0).decode(
                    "utf-8", "replace"
                )
            except OSError:
                break
            m = re.search(
                rf'icon_global_icosahedral_time-invariant_\d+_{var.upper()}\.grib2\.bz2',
                text,
            )
            if not m:
                break
            urls[var] = f"{_DWD_BASE}/{cc}/{var}/{m.group(0)}"
        if len(urls) == 2:
            return urls["clat"], urls["clon"]
    raise RuntimeError("could not locate CLAT/CLON on the DWD open-data server")


def load_cell_coords() -> tuple[np.ndarray, np.ndarray]:
    """Return (clat, clon) cell centres in degrees, downloading once + caching."""
    npz = cache_dir() / "cell_coords_r3b07.npz"
    if npz.is_file():
        d = np.load(npz)
        return d["clat"], d["clon"]

    _log.info("icon_grid_download_coords", dest=str(npz))
    clat_url, clon_url = _find_clatclon_urls()
    coords = {}
    for name, url in (("clat", clat_url), ("clon", clon_url)):
        tmp = cache_dir() / f"{name}.grib2"
        tmp.write_bytes(bz2.decompress(_http_get(url, timeout=120.0)))
        coords[name] = _decode_grib_values(tmp)
        tmp.unlink(missing_ok=True)
    clat, clon = coords["clat"], coords["clon"]
    if clat.size != clon.size:
        raise ValueError(f"CLAT/CLON size mismatch: {clat.size} vs {clon.size}")
    np.savez(npz, clat=clat, clon=clon)
    _log.info("icon_grid_coords_ready", cells=int(clat.size))
    return clat, clon


# --- remap building --------------------------------------------------------

def _lonlat_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Unit-sphere xyz so KD-tree distances are geodesic-correct everywhere."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    cl = np.cos(lat)
    return np.column_stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)])


def build_remap(
    bbox: tuple[float, float, float, float], res: float
) -> Remap:
    """Nearest-neighbour remap of the ICON cells onto a regular grid, cached.

    ``bbox`` is (west, east, south, north) in degrees; ``res`` the target grid
    spacing. Source cells are pre-filtered to the bbox (+1° margin) so the
    KD-tree stays small, then every target point is matched to its nearest cell.
    """
    from scipy.spatial import cKDTree

    west, east, south, north = bbox
    key = f"remap_{west}_{east}_{south}_{north}_{res}.npz"
    npz = cache_dir() / key
    if npz.is_file():
        d = np.load(npz)
        return Remap(lats=d["lats"], lons=d["lons"], index=d["index"])

    clat, clon = load_cell_coords()
    # Target grid (ascending), inclusive of the north/east edge.
    lats = np.round(np.arange(south, north + res / 2, res), 6)
    lons = np.round(np.arange(west, east + res / 2, res), 6)
    glon, glat = np.meshgrid(lons, lats)  # (nlat, nlon)

    # Pre-filter source cells to the bbox + margin so the tree is small.
    m = 1.0
    sel = (
        (clat >= south - m) & (clat <= north + m)
        & (clon >= west - m) & (clon <= east + m)
    )
    src_idx = np.nonzero(sel)[0]
    if src_idx.size == 0:
        raise RuntimeError(f"no ICON cells inside bbox {bbox}")
    tree = cKDTree(_lonlat_to_xyz(clat[sel], clon[sel]))
    _, nn = tree.query(_lonlat_to_xyz(glat.ravel(), glon.ravel()))
    index = src_idx[nn].astype("i8")  # map back to full cell index

    np.savez(npz, lats=lats, lons=lons, index=index)
    _log.info(
        "icon_remap_built", bbox=bbox, res=res,
        target=f"{lats.size}x{lons.size}", src_cells=int(src_idx.size),
    )
    return Remap(lats=lats, lons=lons, index=index)


def remap_values(values: np.ndarray, remap: Remap) -> np.ndarray:
    """Map a ``(…, values)`` array to ``(…, nlat, nlon)`` by nearest cell."""
    nlat, nlon = remap.shape
    picked = values[..., remap.index]           # (…, nlat*nlon)
    return picked.reshape(values.shape[:-1] + (nlat, nlon))
