# Changelog

本项目的变更日志。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
每个版本一个文件：`CHANGELOG-<版本>-<日期>.md`。

## [2.1.0] - 2026-08-26

暴露 Prometheus 指标（`/metrics`），接入 Grafana 监控。

### Added

- **`/metrics` 端点**：Prometheus 文本格式，前缀 `rag_`，标签全部低基数
- **HTTP 层指标**：`rag_http_requests_total{method,path,status}` +
  `rag_http_request_duration_seconds{method,path}`（axum 中间件，path 用路由模板避免高基数）
- **检索指标**：`rag_search_total{scoop}` / `rag_search_duration_seconds{scoop}`
  （含查询嵌入）/ `rag_search_hits_total{scoop}`（命中数分布）/ `rag_search_errors_total{scoop}`
- **写入指标**：`rag_write_total{op:upsert|batch|delete,scoop}` / `rag_write_duration_seconds{op,scoop}`
  / `rag_write_chunks_total{scoop}` / `rag_write_errors_total{op,scoop}`（写者任务内打点）
- **嵌入指标**：`rag_embed_requests_total` / `rag_embed_texts_total` /
  `rag_embed_duration_seconds` / `rag_embed_errors_total`
- **状态 gauge（scrape 时实时）**：`rag_tags{scoop}` / `rag_chunks{scoop}`（writer.health()）
  / `rag_embedder_ready`（模型就绪 0/1）

### Docs

- `README.md`：监控指标表 + Prometheus scrape 配置示例 + Grafana 面板建议
- `docs/deployment.md`：采集配置与指标语义

### 依赖

- 新增 `prometheus = "0.14"`
