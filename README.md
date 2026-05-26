# climate_restore

气象数据的再加工子项目，把下载后输出的多 step `*.subset.grib2` 文件按一次起报（`date + cycle`）合并成 **一个扁平 NetCDF / Zarr**：裁剪到中国区域、按 step 时间拼接、所有变量重命名为阅读友好的 shortName 风格。

输出格式可在 YAML 或 CLI 切换（`output_format: netcdf | zarr`），下游可直接 `xr.open_dataset(...)['u100m']` 或 `xr.open_zarr(...)['u100m']` 拿到 `(step, lat, lon)` 的整段时间序列，无需再处理 GRIB / cfgrib。

---

## 1. 安装

Python ≥ 3.11，依赖管理使用 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync           # 运行依赖
uv sync --dev     # 同时安装 pytest
uv run pytest     # 跑测试 (22 个)
```

---

## 2. 快速开始

```bash
# 1. 看注册了哪些源 adapter
uv run climate_restore list-sources
# aifs-single / gfs-0p25 / graphcast / graphcast-pres / graphcast-sfc / ifs-hres

# 2. 处理一个已下载完成的 manifest
uv run climate_restore run \
    --config config/jobs/gfs_china.yaml \
    --manifest ../download/output/gfs-0p25/20260501/12z/20260501_12z_gfs-0p25.manifest.json

# 3. 长驻：轮询 download 输出，发现新 manifest 就处理
uv run climate_restore watch \
    --config config/jobs/gfs_china.yaml \
    --source gfs-0p25 --interval 30
```

产物路径（扁平：filename 已经带 `<date>_<cycle>z_<source>`，所以一次起报落一个文件即可）：

```
<output_dir>/<source>/<date>_<cycle>z_<source>.nc
# e.g. output/gfs-0p25/20260501_12z_gfs-0p25.nc
```

读取示例：

```python
import xarray as xr
ds = xr.open_dataset("output/gfs-0p25/20260501_12z_gfs-0p25.nc")
ds["u10"].sel(latitude=slice(20, 40), longitude=slice(100, 120))   # (step, lat, lon)
ds["t"].sel(pressure_level=850)                                      # 压力层取片
ds.valid_time.values                                                 # 每个 step 的绝对时刻
```

---

## 3. 作业配置（YAML）

`config/jobs/gfs_china.yaml`：

```yaml
download_root: ../download                 # download 项目根；manifest 路径可自动反推
output_dir: output                         # 输出根目录（.nc 或 .zarr 落于其下）
bbox: [70.0, 140.0, 15.0, 55.0]            # west, east, south, north
verify_sha256: false                       # true 时全文件哈希校验（慢）
workers: 4                                 # 并行解码 GRIB 的进程数；1 关闭进程池
output_format: netcdf                      # netcdf (默认, .nc) | zarr (.zarr 目录)
zarr:                                      # 仅 output_format=zarr 时生效
  chunks: {step: -1, latitude: 64, longitude: 64, pressure_level: -1}
  compressor: zstd                         # zstd | lz4 | blosclz | zlib | none
  clevel: 3
  consolidated: true
  zarr_format: 2
```

所有字段都可被 CLI flag 覆盖：`--download-root` / `--output-dir` / `--bbox 70,140,15,55` / `--verify-sha256` / `--workers` / `--output-format {netcdf,zarr}` / `--zarr-chunks step=-1,latitude=64,longitude=64`。

### Zarr 输出

切换到 zarr 后产物为目录 store：

```
<output_dir>/<source>/<date>_<cycle>z_<source>.zarr/
```

读取（`consolidated=True` 时一次拿全部元数据，开启快）：

```python
import xarray as xr

ds = xr.open_zarr("output/ifs-hres/20260101_12z_ifs-hres.zarr", consolidated=True)

# 取某省/电网区域的所有变量、整段预报时序 —— 只解 1-4 个 chunk，极快
region = ds[["u100m", "v100m", "t2m", "ssrd"]].sel(
    latitude=slice(35.0, 45.0),
    longitude=slice(110.0, 120.0),
).load()   # 一次 I/O 全部读入内存，后续循环/切片走纯 numpy，无再次 I/O

# 之后训练循环直接 .values，无额外开销
X = region["u100m"].values   # shape: (step, lat, lon)
```

默认 chunk 策略 `{step: -1, latitude: 64, longitude: 64, pressure_level: -1}` 偏向**区域时序**场景：16°×16° 以内的区域整段时序只命中 1 个 chunk（~800 KB），`.load()` 一次解压完毕。如果下游主要按时间窗口扫描全图（如气象大模型训练），把 `step` 改小（如 `12`）并把 `latitude/longitude` 设为 `-1` 更合适。`-1` 表示该维一整块；未出现在数据集中的维会被忽略，因此一份默认 chunks 可跨产品复用。

---

## 4. CLI 参考

```
climate_restore run         --manifest PATH [common]
climate_restore watch       [--source NAME] [--interval SEC] [common]
climate_restore list-sources
```

公共 flag（`common`）：

| flag | 含义 |
|---|---|
| `--config FILE` | 加载作业 YAML（不传则用内置默认） |
| `--download-root DIR` | 覆盖 download 项目根 |
| `--output-dir DIR` | 覆盖输出目录 |
| `--bbox W,E,S,N` | 覆盖裁剪框 |
| `--source-type NAME` | 强制使用某个 adapter（默认看 `manifest.source.name`） |
| `--verify-sha256` | 启用 sha256 校验（默认只查 size） |
| `--workers N` | 覆盖 YAML 的 `workers`（默认 `min(cpu_count, 4)`；`1` 关闭进程池）。85 文件实测：1→70s、4→23s、8→13s |
| `--output-format {netcdf,zarr}` | 覆盖 YAML 的 `output_format`（默认 `netcdf`） |
| `--zarr-chunks SPEC` | 覆盖 zarr chunks，格式 `dim=size,dim=size`，`-1` 表示整维一块 |
| `--log-level LVL` | 日志级别（默认 INFO，JSON 行格式输出到 stderr；TTY 下会显示 tqdm 进度条） |

---

## 5. 已支持的源

| 注册名 | adapter | 说明 |
|---|---|---|
| `gfs-0p25` | `GfsAdapter` | NOAA GFS 0.25° atmos forecast (wgrib2 idx) |
| `graphcast-sfc` / `graphcast-pres` / `graphcast` | `GraphCastAdapter` | NOAA NWS GraphCastGFS (aigfs)，sfc + pres 两份独立 manifest |
| `aifs-single` / `ifs-hres` | `AifsAdapter` | ECMWF AIFS / IFS HRES (open-data) |

GraphCast 每次起报会落两份 manifest（`-sfc` / `-pres`），各自处理成各自的 `.nc`；下游若要"一次起报的完整状态"，可对同一 (date, cycle) 的两个文件 `xr.open_mfdataset` 合并。`-pres` 内部不同变量覆盖的压力层不一致（`t/u/v/gh` 在 [1000, 850, 500]，`q` 仅 [1000, 850]，`w` 在 [925, 850, 700, 500]），合并后 `pressure_level` 取并集 `[500, 700, 850, 925, 1000]`，未覆盖位置为 NaN。

变量命名约定（重命名后）：

- `u10` / `v10` / `t2m` / `d2m` / `sh2` / `r2` 等已自带数字的 shortName 保留原名
- 通用变量在 ≥ 80 m 高度自动加米后缀：`pres80m` / `q80m` / `t100m` / `u100m`
- 压力层变量保留原名并携带 `pressure_level` 维度（`t` / `u` / `v` / `gh` / `z` / `q` / `w` …）
- 非 instant 量加 stepType 后缀：`prate` → `prate_avg`，`tp` (accum) → `tp_accum`，cfgrib 的 `avg_hcc` → `hcc_avg`
- 地面瞬时 `t` 在 GFS 下改名为 `t_sfc`，避免与压力层 `t` 冲突

---

## 6. 完整性与失败语义

下游 sensor 是用 **per-init manifest** 触发的，不是用 `_runs/` 下的批次日志。两者的语义区别要分清：

| 文件 | 粒度 | 写入时机 | 作用 |
|---|---|---|---|
| `<source>/<date>/<cycle>z/<date>_<cycle>z_<source>.manifest.json` | 一次起报（date, cycle, source） | `_run_init` 走完后**原子**写入（tmp + `os.replace`），且**只有至少一个 step 成功**才写 | 下游 ready 信号 |
| `_runs/run_<ts>.json` | 一次 CLI 调用（可覆盖多个 init） | 整次跑完后写入，含 `succeeded / failed / total` + `failures[]` | 批次审计，不适合做实时触发 |

restorage 只看 manifest，因此实际行为：

| 失败情形 | manifest | restorage |
|---|---|---|
| 整个 init 列表失败 / 全部 step 失败 | 不写 | watch 看不到，自然不处理 ✅ |
| 部分 step 失败（49 个挂 2 个） | 写入，但 `files[]` 只含成功的 47 个，**无 incomplete 标记** | 照常处理 47 个 step；输出的 `step` 维稀疏 ⚠️ |
| 上游还没发够 step（`init_steps_missing`） | 同上 | 同上 ⚠️ |
| manifest 自己写失败 | 无 manifest | 不处理 ✅ |
| 落盘后磁盘损坏 | size 不匹配 | `manifest.verify()` size 校验拦截；`--verify-sha256` 加哈希一层 ✅ |

**推荐做法**：消费产物时先检查 `len(ds.step)` 和 `ds.valid_time` 是否连续，缺帧时回查 `_runs/` 下相近时间的 run report 看是否有 `failures[]`。等真出现过一次因部分失败导致下游困惑，再考虑在 download manifest 里加 `expected_steps` / `missing_steps` 字段。

---

## 7. 新增数据源

只要在 `src/climate_restore/sources/` 下加一个文件，挑选合适的 Rule 组合即可。例如：

```python
# src/climate_restore/sources/myproduct.py
from climate_restore.rules import DimRenameRule, HeightSuffixRule, StepTypeSuffixRule
from climate_restore.sources.base import BaseAdapter
from climate_restore.sources.registry import register

@register("my-source-name")    # 必须匹配 manifest 里的 source.name
class MyAdapter(BaseAdapter):
    name = "myproduct"
    RULES = {
        "heightAboveGround": HeightSuffixRule(unit="m"),
        "isobaricInhPa": DimRenameRule(dst_dim="pressure_level"),
        "surface": StepTypeSuffixRule(),
    }
```

再去 `sources/__init__.py` 加一行 `from . import myproduct as _myproduct  # noqa: F401`，不用动 `processor.py` / `cli.py`。未覆盖的 `typeOfLevel` 第一次跑会打 `unhandled_type_of_level` warning 并走 passthrough 兜底。

可复用的 Rule 见 `src/climate_restore/rules.py`：`HeightSuffixRule` / `DimRenameRule` / `StepTypeSuffixRule` / `PrefixToSuffixRule` / `PassthroughRule`。

---

## 8. 测试

```bash
uv run pytest -q
```

包含：

- `tests/test_rules.py` — 五个 Rule 类的单元测试（in-memory `xr.Dataset`，不依赖 GRIB）
- `tests/test_registry.py` — 注册 / 查找 / 重名抛错 / 排序
- `tests/test_gfs_golden.py` — 对 `../download/output/.../f001.subset.grib2` 跑 GFS adapter，断言产出的变量名集合（fixture 缺失时自动 skip）z
