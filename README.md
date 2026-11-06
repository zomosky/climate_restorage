# climate_restore

GEFS 命名已统一为 `gefs-ens`（成员预测）和 `gefs-ctrl`（控制预测）。旧 `gefs-s2s` / `gefs-s2s-control` 产品仍可读取；同时存在新旧名称时自动目录优先展示新名称。下载命令 `gefs-s2s` 保留为 `gefs-ens` 的兼容别名，新产物使用新名称。

## ENS 事件概率

`ifs-ens-probabilities` 和 `aifs-ens-probabilities` 复用每起报 manifest 处理链。
240h/360h 是两个多时间下载包；IFS 合计 5 个字段、145 个窗口样本，AIFS 合计
25 个字段、313 个窗口样本。日常要求两个包及全部预期窗口齐备。
历史回填使用 manifest 的 `probability_availability`，记录上游实际可用及缺少的字段/窗口，
逐项核验实际可用集合后发布；网络/处理失败仍不得作为上游缺失。IFS旧0.4°网格与0.25°网格分别按原网格保留，核验与回执中声明的间距一致。

概率始终保留百分比 `%`，`GRIB_stepType=accum/max/avg` 描述的是事件的统计方式，
不能把概率本身当作降水累计量或阵风最大值进行差分。每个变量写入
`zrread_kind=probability`、阈值及单位、`probability_type`、比较方式、原始阈值、
`probability_statistic` 和 `probability_window_hours`。
`probability_window_start_coord/end_coord` 指向该变量逐样本的 UTC 窗口边界坐标。
瞬时字段起止相同；原 24h 降水/阵风字段保持参数名。多日字段按统计方式和时长拆分，
例如 `2tl273_max48h/max72h/max120h`、`tpl01_accum120h`、`tprg3_avg120h`。
其中多日 `2tl273` 是**区间最高气温低于 0℃**的概率。

重整先按 GRIB 参数、模板、统计类型和时长拆分解码，再裁剪/合并。适配器可提供
`open_hypercubes(path)` 与 `validate_product(dataset)` 扩展点；后者在写盘前检查
字段、全部原始窗口、0–100 范围以及时间边界，失败不写完整标记。
根属性 `probability_complete=1`、JSON 字符串 `probability_expected_samples` 和
逐变量 `probability_sample_count` 是监控使用的验收回执。
历史产品的 `probability_complete=1` 指声明的可用数据已完整处理，
`probability_catalog_complete=0` 则明确它不具备当前完整目录；缺项不能补零。
`scripts/restore_probability_history.py` 校验原始 SHA-256，在隐藏暂存目录处理、全量读回后发布。

```sh
uv run climate_restore scan-once --download-root /climate_data --source ifs-ens-probabilities --output-dir /climate_data/climate_data_storage/zarr --output-format zarr --workers 1
```

气象数据的再加工子项目，把下载后输出的多 step `*.subset.grib2` 文件按一次起报（`date + cycle`）合并成 **一个扁平 NetCDF / Zarr**：裁剪到中国区域、按 step 时间拼接、所有变量重命名为阅读友好的 shortName 风格。

输出格式可在 YAML 或 CLI 切换（`output_format: netcdf | zarr`），下游可直接 `xr.open_dataset(...)['u100m']` 或 `xr.open_zarr(...)['u100m']` 拿到 `(step, lat, lon)` 的整段时间序列，无需再处理 GRIB / cfgrib。

---

## 1. 安装

Python ≥ 3.11，依赖管理使用 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync           # 运行依赖
uv sync --dev     # 同时安装 pytest
uv run pytest     # 跑测试 (30 个；含 3 个在缺 GRIB fixture 时自动 skip)
```

---

## 2. 快速开始

```bash
# 1. 看注册了哪些源 adapter
uv run climate_restore list-sources
# aifs-single / gfs-0p25 / graphcast / graphcast-history / graphcast-pres / graphcast-sfc / ifs-hres

# 2. 处理一个已下载完成的 manifest
uv run climate_restore run \
    --config config/jobs/gfs_zarr_china.yaml \
    --manifest ../download/output/gfs-0p25/20260501/12z/20260501_12z_gfs-0p25.manifest.json

# 3. 长驻：轮询 download 输出，发现新 manifest 就处理
uv run climate_restore watch \
    --config config/jobs/gfs_zarr_china.yaml \
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

`config/jobs/gfs_zarr_china.yaml`（`config/jobs/` 下现有 `gfs_zarr_china` / `ifs_zarr_china` / `aifs_zarr_china` / `graphcast_china` 四份，均输出 zarr）：

```yaml
download_root: ../download                 # download 项目根；manifest 路径可自动反推
output_dir: output                         # 输出根目录（.nc 或 .zarr 落于其下）
bbox: [70.0, 140.0, 15.0, 55.0]            # west, east, south, north
workers: 4                                 # 并行解码 GRIB 的进程数；1 关闭进程池（默认 min(cpu,4)）
output_format: zarr                        # netcdf (代码默认, .nc) | zarr (.zarr 目录)
zarr:                                      # 仅 output_format=zarr 时生效
  chunks: {step: -1, latitude: 64, longitude: 64, pressure_level: -1}
  compressor: zstd                         # zstd | lz4 | blosclz | zlib | none
  clevel: 3
  consolidated: true
  zarr_format: 2
```

所有字段都可被 CLI flag 覆盖：`--download-root` / `--output-dir` / `--bbox 70,140,15,55` / `--workers` / `--output-format {netcdf,zarr}` / `--zarr-chunks step=-1,latitude=64,longitude=64`。

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
climate_restore scan-once   [--source NAME] [--force] [common]
climate_restore list-sources
```

三种驱动方式的区别：

| 子命令 | 形态 | 幂等性来源 | 适用 |
|---|---|---|---|
| `run` | 处理单个 manifest，退出 | 无（每次都重处理） | 手动补一个起报 |
| `watch` | 常驻轮询，永不退出 | **内存** `seen` 集合 | 前台调试、单机长驻 |
| `scan-once` | 扫一遍全部 manifest，处理未完成的，退出 | **磁盘产物**（输出已完整就跳过） | **cron 定时调度** |

> `watch` 的 `seen` 在重启时会被"当前已存在的全部 manifest"重新播种，因此**进程宕机期间新增的 manifest 会被永久跳过**。`scan-once` 把幂等性建立在磁盘上的 `.zarr`/`.nc` 产物上：只要输出还不存在或不完整就会处理，所以可以由 cron 任意频率反复拉起，宕机不丢任务。判定"已完整"用的是**写到最后**的标记——consolidated zarr 看 `.zmetadata`、netcdf 看非空文件——所以半途崩溃留下的残缺 store 会被重新处理而非误跳过。

`scan-once` 专属 flag：

| flag | 含义 |
|---|---|
| `--source NAME` | 只扫某个源子树（如 `gfs-0p25`） |
| `--force` | 即使产物已存在也重新处理（改了规则/想重建时用） |

退出码：全部成功或跳过 `0`；只要有一个**已就绪**的 manifest 处理失败就返回 `1`（供调度器告警）。下载未完成（无 `completed_at` 或带 `failures`）的 manifest 计入 `not_ready`，留待下次扫描，不算失败。

### cron 定时处理（推荐用 `scan-once`）

```cron
# 每 10 分钟扫一次 download 输出，把新下载好的起报处理成 zarr。
# flock 防止上一轮没跑完时重叠启动（scan-once 本身幂等，flock 只是省资源）。
*/10 * * * * /usr/bin/flock -n /tmp/climate_restore.lock \
    sh -c 'cd /path/to/climate_restorage && uv run climate_restore scan-once \
        --config config/jobs/gfs_zarr_china.yaml --output-format zarr \
        >> logs/scan.log 2>&1'
```

公共 flag（`common`，三个处理子命令通用）：

| flag | 含义 |
|---|---|
| `--config FILE` | 加载作业 YAML（不传则用内置默认） |
| `--download-root DIR` | 覆盖 download 项目根 |
| `--output-dir DIR` | 覆盖输出目录 |
| `--bbox W,E,S,N` | 覆盖裁剪框 |
| `--source-type NAME` | 强制使用某个 adapter（默认看 `manifest.source.name`） |
| `--workers N` | 覆盖 YAML 的 `workers`（默认 `min(cpu_count, 4)`；`1` 关闭进程池）。85 文件实测：1→70s、4→23s、8→13s |
| `--output-format {netcdf,zarr}` | 覆盖 YAML 的 `output_format`（默认 `netcdf`） |
| `--zarr-chunks SPEC` | 覆盖 zarr chunks，格式 `dim=size,dim=size`，`-1` 表示整维一块 |
| `--log-level LVL` | 日志级别（默认 INFO，JSON 行格式输出到 stderr；TTY 下会显示 tqdm 进度条） |

---

## 5. 已支持的源

| 注册名 | adapter | 说明 |
|---|---|---|
| `gfs-0p25` | `GfsAdapter` | NOAA GFS 0.25° atmos forecast (wgrib2 idx) |
| `graphcast-sfc` / `graphcast-pres` / `graphcast` / `graphcast-history` | `GraphCastAdapter` | NOAA NWS GraphCastGFS (aigfs)，sfc + pres 两份独立 manifest |
| `aifs-single` / `ifs-hres` | `AifsAdapter` | ECMWF AIFS / IFS HRES (open-data) |
| `ifs-hres-0p1` | `IfsHres0p1Adapter` | ECMWF IFS 0.1°，独立于原 `ifs-hres` 0.25°；命名规则沿用 IFS |

`ifs_0p1_zarr_china_ser.yaml` 将新源裁到 70–140°E、15–55°N，保留原生
0.1° 网格（401×701），以 2 个解码进程写入独立 Zarr 目录。适配器按 GRIB
的微度精度规范化经纬度，避免浮点误差漏掉裁剪边界；不改变气象数值或插值。
每份 `fNNN.subset.grib2` 只保留编码时效等于文件时效的场，跳过 f000 中指向
f003 的零值最高/最低温预览，防止其覆盖真实 f003 数据。不存在的原生场保持 NaN。

GraphCast 每次起报会落两份 manifest（`-sfc` / `-pres`），各自处理成各自的 `.nc` / `.zarr`；下游若要"一次起报的完整状态"，可对同一 (date, cycle) 的两个文件 `xr.open_mfdataset` 合并。`-pres` 内部不同变量覆盖的压力层不一致（`t/u/v/gh` 在 [1000, 850, 500]，`q` 仅 [1000, 850]，`w` 在 [925, 850, 700, 500]），合并后 `pressure_level` 取并集 `[500, 700, 850, 925, 1000]`，未覆盖位置为 NaN。

变量命名约定（重命名后）：

- `u10` / `v10` / `t2m` / `d2m` / `sh2` / `r2` 等已自带数字的 shortName 保留原名
- 通用变量在 ≥ 80 m 高度自动加米后缀：`pres80m` / `q80m` / `t100m` / `u100m`
- 压力层变量保留原名并携带 `pressure_level` 维度（`t` / `u` / `v` / `gh` / `z` / `q` / `w` …）
- 非 instant 量加 stepType 后缀：`prate` → `prate_avg`，`tp` (accum) → `tp_accum`，cfgrib 的 `avg_hcc` → `hcc_avg`
- 地面瞬时 `t` 在 GFS 下改名为 `t_sfc`，避免与压力层 `t` 冲突
- 云量分层：高/中/低云 → `hcc` / `mcc` / `lcc`（各自 `_avg` 变体）。GFS 的总云量 `tcc` 实际同时出现在 **3 个 typeOfLevel**——`atmosphere`（整层，保留为 `tcc` / `tcc_avg`）、`convectiveCloudLayer`（对流云，重命名为 `tcc_conv`）、`boundaryLayerCloudLayer`（边界层云，`avg` → `tcc_bl_avg`）。三者 cfgrib shortName 都是 `tcc`，若不改名会在按 step 拼接时**塌缩成同一个 `tcc` 并互相静默覆盖**，因此后两层各起独立名字

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
| 落盘后磁盘损坏 | 文件在但内容损坏 | `verify()` 只确认文件存在（不再校验 size/sha256）；损坏在 cfgrib 解码该 GRIB 时报错暴露 ⚠️ |

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

- `tests/test_rules.py` — 五个 Rule 类的单元测试（in-memory `xr.Dataset`，不依赖 GRIB），含 `StepTypeSuffixRule` 的 `renames`（防 `tcc` 多层塌缩）回归用例
- `tests/test_registry.py` — 注册 / 查找 / 重名抛错 / 排序
- `tests/test_scan.py` — `scan-once` 幂等性 / 就绪判定
- `tests/test_gfs_golden.py` — 对 `../download/output/.../f001.subset.grib2` 跑 GFS adapter，断言产出的变量名集合（fixture 缺失时自动 skip）

## CAMS 全球大气成分预报

`ec_cams` 已注册独立adapter，任务配置为 `config/jobs/cams_zarr_china.yaml`（单worker）。
接收下载端完整52变量、5732消息的每起报manifest，输出完整0–120h逐小时联合坐标；7个3小时变量在非原生时效保持缺测。
9个辐射能量和2个降水变量只规范累计属性并保留源属性，原值不变；过去3小时阵风统一为max，AOD保持instant。
下载、恢复及生产运行约束见下载项目 [CAMS.md](../download/docs/CAMS.md)。


## 预报变量覆盖索引

正式 Zarr 写入并通过 chunk 校验后，自动增量更新同一共享根目录下的
`.catalog/<source>.sqlite3`。每次只读取当前起报的完整分片和小型气压坐标；
重复起报按元数据指纹跳过，回填较早起报和覆盖重写会修正覆盖范围与缺项标记。
普通 run/scan-once/batch 与概率历史回填都已接入，隐藏暂存库不发布。
索引写入失败会使任务失败；完整产品的幂等跳过/历史续跑会补发索引，无需重解 GRIB。

既有历史及新源需初始化一次，可显式执行或由后台巡检在产品稳定后完成；第一次单个起报更新不会宣称全部历史已登记：

```sh
uv run climate_restore coverage-index rebuild --root /path/to/zarr --all
uv run climate_restore coverage-index rebuild --root /path/to/zarr --source ifs-ens-probabilities
```

首次上线顺序：更新重整端 → 初始化全部源索引 → 更新 watcher。后续入库、历史回填
和重试自动更新。重建逐源加锁并原子替换；失败保留旧库，不影响正式 Zarr。
watcher 使用只读事务，不需要在数据卷创建 WAL/shm；不要改用 WAL journal。

外部工具发布后可以 `coverage-index update --store <正式路径>`；手工删除正式库前可
`coverage-index remove --store <原路径>` 撤回整个起报。remove 不删除数据。
外部手工改动/备份恢复可补发或重建，也可由 `deploy/coverage_index_maintain.py` 定时比对元数据指纹并修复。新源、缺失/未初始化/损坏索引自动重建；正常源只更新变化的起报，失败保留旧索引。生产宿主机每小时第 7、37 分钟执行，复用现有重整调度锁；安装和资源限制见 `deploy/README.md`。GRIB 原始冷归档无需变更索引。
`rebuild --index-dir /tmp/audit/.catalog` 允许只读审计时把结果写到独立临时目录。

持久协议 v1：state(version,source,ready,total,first_init,last_init)；
products(init,definitions,signature) 保存暴露名/base/level/attrs；
product_variables(init,name) 保存基础名和暴露名成员；
coverage(name,first_init,last_init,count) 保存物化聚合。按源 flock，SQLite DELETE
journal + FULL synchronous，更新一个事务、读端一个只读快照。coverage 表示包含字段的
已发布起报范围，不保证每个格点/每个概率窗口无缺失。详细上线、修复、备份和回滚
说明见同级 watcher 的 `docs/forecast-coverage-index.md`。

## ECMWF S2S 成员与原始窗口

接受 `schema_version=1, kind=s2s` 的组清单，`files[]` 保留 `group/request`，
不要求或伪造 `step_hours`；普通预报清单仍要求该字段。
`s2s-ecmwf` 用于控制预报；原同名清单若声明 `perturbed_forecast`，输出源自动
规范为 `s2s-ecmwf-perturbed`。实验组合清单可显式使用
`s2s-ecmwf-ensemble` 和 `source.forecast_type=ensemble`，每个文件的
`request.forecast_type` 分别声明控制或扰动预报，三种产物互不覆盖。

输出保留 `step/member/latitude/longitude`（压力层另有 `pressure_level`）；
控制成员也保留长度 1 的整数 `member=0`。变量沿用 `u10/v10/msl`、
`t2m_avg/d2m_avg/tcc_avg`、`ssrd_accum/strd_accum/tp_accum` 和压力层 `u/v/t/gh`。
每变量的 `s2s_sample_present_coord`、`s2s_window_start_coord`、
`s2s_window_end_coord` 属性指向按 GRIB 消息记录的非空间维坐标。
未发布组合保留 `presence=False`、窗口 `NaT`、数值 `NaN`；实际消息即使
全部格点为 NaN，仍标为存在。累计窗口从起报开始，平均保留真实区间，瞬时起止相同。
窗口标注 `zrread_time_boundary=1, timezone=UTC`，并显式使用带缺测值的
float64 CF 时间编码，避免稀疏 NaT 在 Zarr/NetCDF 读回时溢出。

处理前验证每文件起报、网格、成员类型、集合规模、请求变量/层/原生窗口；
缺成员、重复参数/层/成员/时效、混起报和窗口冲突均报错。历史组清单虽可能
统一请求 6h，海平面气压、辐射及压力层仅检查 ECMWF 的 24h 原生时效。
同变量控制/扰动按坐标合并，不使用确定性路径的 step 去重。
根属性 `s2s_message_coverage` 逐变量记录真实成员及样本数，`s2s_complete=1`
仅表示已声明请求的覆盖校验通过，不意味着每个变量都包含 101 个成员。

cfgrib 先保持惰性，再按单成员切块、区域裁剪后加载；由于 cfgrib 内部会先解
完整空间场，单成员切块防止全球全集合同时进入内存。查询常按成员读取时，
建议显式设置分块（不改变其他源默认值）：

```sh
uv run climate_restore run --manifest /path/to/s2s.manifest.json \
  --output-format zarr --workers 1 \
  --zarr-chunks member=1,step=28,latitude=32,longitude=32
```

本轮仅实现上述 ECMWF S2S 参数；不自动插值为 15 分钟，不将源累计值在重整阶段差分。

电力核心字段补充 `sp`（地面气压，0–1104h 每 24h）及
`mx2t6_max` / `mn2t6_min`（过去 6 小时最高/最低 2m 气温，6–1104h 每 6h）。
它们分别对应 ecCodes 参数 134、121、122；极值必须保持原始
`[end−6h,end]` 窗口，不能当瞬时温度或累计量处理，也不存在 0h 极值样本。
新增三字段当前通过本地合成 GRIB 解码/写盘验证，真实下载验收另行记录。

## GEFS 35 天指定成员预测

`gefs-ens` 是独立的 NOAA 0.5° 集合源，使用普通 `schema_version=1`
逐步清单；每文件必须有 `step_hours`，不使用 ECMWF 的 `kind=s2s`。
`variables` 按输出名声明，例如 `[{"name":"t2m"}, ...]`；完整目标为
`t2m/d2m/r2/u10/v10/u100/v100/sp/prmsl/tmax2m_max/tmin2m_min/tp_accum/sdswrf_avg/sdlwrf_avg/tcc_avg`。
原生时效为 0–240h 每 3h、246–840h 每 6h，合计 181 步。
每个成员的完整目标在 0h 仅有 9 个瞬时场，其余 180 步各 15 场，共 2709 条消息；
0h 未发布的统计场保持 NaN、presence=False 和窗口 NaT。

清单的 `source.members` 指定本次成员，如 `[0,1,2,3,4]` 为c00与p01–p04；
支持非空、无重复的0–30整数名单，也允许仅扰动成员如 `[1,7,30]`。
旧清单未声明 `members` 时，按原 `source.member`（默认0）处理，保留单控制成员兼容。
每步文件须包含所有声明成员×字段；逐消息校验成员身份：c00为cf/type1，p01–p30为pf/type3。
原始头中的30是扰动成员总数，不表示本次下载了30个成员；实际数量以`member`维和
`gefs_actual_member_count`为准。混合成员产物不保留误导性的单一GRIB成员类型属性，
各成员类型记录在`gefs_member_types`。

`gefs_complete_horizon=1`表示所选成员的15字段/181步完整；不表示全31成员。
`gefs_full_ensemble=1`仅表示成员轴恰为0–30；两个标记同时为1才表示全成员全时效。
切勿将不同成员配置写到同一产品目录后依赖scan-once覆盖：它按已存在产品跳过。
实验下载按日期×成员名单×时效生成独立目录，扩充成员需在独立目录重整后再验收。
合并时先按变量/成员拼接不重复时效，再对齐成员轴，避免每条消息都提前扩展成整轮网格；
成员时效缺口、重复消息和网格一致性校验保留。


复用 `s2s_present__*` / `s2s_window_start__*` / `s2s_window_end__*`
坐标及逐变量属性引用。保留原始 `GRIB_stepType`、单位和真实窗口：例如
降水及平均辐射可为 0–3、0–6、6–9、6–12h，不能当作从起报连续累计。
辐射原值是 W/m² 平均通量；重整不差分、不改窗口、不转换成 ECMWF 累计量。
读取侧按真实原窗口单独处理 GEFS；所选成员预测不自动生成校准集合概率产品。

### 从已有成员产品发布控制入口

```bash
uv run climate_restore export-gefs-control \
  --input /path/to/gefs-s2s/20261029_00z_gefs-s2s.zarr \
  --output-dir /path/to/zarr
```

只抽取已有产品的成员0，不重新下载或解码。输出为
`gefs-s2s-control/20261029_00z_gefs-s2s-control.zarr`，物理保留`member=[0]`及
全部变量窗口、presence和缺失值，并重算控制成员覆盖摘要与成员属性，记录父产品出处。
临时目录写入、重新读取逐值核验后才原子发布；重复运行逐值核验已有输出，完全相同即复用，
不同则报错且保留原成品，不隐式覆盖。成功发布更新覆盖索引。

生产 GEFS 使用 `deploy/gefs_publish.py`：完整原始清单校验后在隐藏暂存目录切片，
全量解码验证通过才原子发布成员与控制产品；由下载项目 `deploy/gefs_run.py` 调用。
