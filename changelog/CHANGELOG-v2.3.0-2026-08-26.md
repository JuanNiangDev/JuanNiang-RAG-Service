# Changelog

本项目的变更日志。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
每个版本一个文件：`CHANGELOG-<版本>-<日期>.md`。

## [2.3.0] - 2026-08-26

Grafana 数据面板（`deployment/grafana/rag-dashboard.json`）。

### Added

- **现成 Grafana 面板**（6 组 25 个面板，覆盖 v2.1.0 全部指标）：
  - 🩺 状态：嵌入模型就绪（值映射 就绪/未就绪）、总 tag/块数、四个分库 tag 数 stat
  - 🔍 检索：QPS / P95 耗时（含查询嵌入）/ 命中数 P95 / 错误率（均按分库）
  - ✍️ 写入：速率 / P95 耗时 / 块向量速率 / 失败率（按 op × 分库）
  - 🧠 嵌入：请求 QPS / 文本速率 / P95 耗时 / 错误率
  - 🌐 HTTP：QPS / P95 / 5xx 错误率（按路由模板）+ 状态码堆叠分布
  - 📦 存储规模趋势：分库 tag / 块向量数时间序列
- 导入方式：Grafana → Dashboards → Import → 上传 json → 选择 Prometheus 数据源（`${DS_PROMETHEUS}` 变量）

### Docs

- `README.md` / `docs/deployment.md`：补充面板导入说明
