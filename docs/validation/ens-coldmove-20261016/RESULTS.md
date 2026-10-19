# ENS 概率源 30 天冷存验收

验证及部署日期：2026-10-16，北京时间。

## 结果

- 已安装 `/home/zhangmingyu/operation/forecast-coldmove.sh`，包含 IFS ENS / AIFS ENS
  两个概率源，保留 `KEEP_DAYS=30` 及原有北京时间每天 07:00 的 crontab。
- 脚本 SHA-256：`754c5970646bb857dcc04d486258ee20652eaacf8133b847400495ccca93a477`。
- 原脚本备份：`/home/zhangmingyu/operation/releases/probability-coldmove-20261016-135443/forecast-coldmove.sh.before`。
- 服务器 Linux 环境 `tests/test_probability_coldmove.py`：19 passed，0 failed，
  0 skipped，JUnit 见 `coldmove-tests.xml`；本地受影响范围测试为 18 passed、6 skipped，
  其中 6 个依赖 GNU date/flock 的流程场景已在服务器实际执行通过。
- 实际 Docker 内只读核验 `20261015/12z` 两源产品：均 `product_verified=true`。
- 已安装脚本的概率源只读演练：`cutoff=20260916`、`DONE rc=0`。

## 验收边界与失败保护

测试的转移场景在临时目录执行真实 shell、tar、文件校验和移除操作；仅替换 Docker
传输为本机解释器，所有删除均限制在测试临时原始目录。生产 Docker 路径通过两个真实
产品的只读校验另行验证。没有迁移或删除生产原始数据：当前仅有未满 30 天的起报。
因此上述结果不是“首批生产数据已转冷存”的证据。

覆盖完整产品放行、部分/全部分块缺失、压缩块损坏、窗口错误、概率越界、Zarr 落后于
manifest、完成标记缺失、下载失败、原始文件 SHA-256 不符、另一个 cycle 未完成、
冷端同大小不同内容、干运行、下载锁占用、未到保留期限等情况。

转移前校验产品；复制后再次校验，并逐文件比较冷、热目录的文件集合和 SHA-256。
发现问题保留热端；所有转移场景均核对热端 Zarr 内容未变化。概率下载与冷存共用锁。

部署回执见 `deployment.json`。回滚仅恢复备份宿主机脚本，无需重建查询服务。
