"""restore 后 chunk 完整性校验:截断(部分 chunk 缺)必须被检出并报错。"""
from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

from climate_restore.processor import (
    ChunkIntegrityError,
    _verify_zarr_chunks,
    write_dataset,
)
from climate_restore.config import ZarrOptions


def _make_ds():
    return xr.Dataset(
        {"t2m": (("step", "latitude", "longitude"),
                 np.random.default_rng(0).random((3, 128, 128)).astype("f4"))},
        coords={"step": [0, 6, 12],
                "latitude": np.arange(128.0), "longitude": np.arange(128.0)},
    )


def test_complete_write_no_truncation(tmp_path):
    p = tmp_path / "ok.zarr"
    _make_ds().to_zarr(p, consolidated=True, zarr_format=2,
                       encoding={"t2m": {"chunks": (3, 64, 64)}})
    assert _verify_zarr_chunks(p) == []  # 完整:4 chunk 全在,不误报


def test_truncated_write_detected(tmp_path):
    p = tmp_path / "trunc.zarr"
    _make_ds().to_zarr(p, consolidated=True, zarr_format=2,
                       encoding={"t2m": {"chunks": (3, 64, 64)}})
    # 删掉一个 t2m 数据 chunk 文件 → 模拟写截断
    cdir = p / "t2m"
    data_chunks = [f for f in os.listdir(cdir) if not f.startswith(".")]
    os.remove(cdir / data_chunks[0])
    bad = _verify_zarr_chunks(p)
    assert any("t2m" in b for b in bad), bad


def test_write_dataset_raises_on_truncation(tmp_path, monkeypatch):
    # 让 write_dataset 写完后 mock 出截断,验证它删库 + 抛 ChunkIntegrityError
    p = tmp_path / "wd.zarr"
    import climate_restore.processor as proc
    orig = proc._verify_zarr_chunks
    monkeypatch.setattr(proc, "_verify_zarr_chunks", lambda s: ["t2m: 3/4"])
    with pytest.raises(ChunkIntegrityError, match="截断"):
        write_dataset(_make_ds(), p, fmt="zarr", zarr_options=ZarrOptions())
    assert not p.exists()  # 坏库已删
