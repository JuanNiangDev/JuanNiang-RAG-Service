# JuanNiang-RAG-Service

![banner](./docs/imgs/banner.png)

JuanNiang-Neo 的 RAG 检索服务：**分库（scoop）↔ tag（uuid）↔ 向量** 的存储与检索，供主 Agent 调用。

- 向量按功能块**物理分库**（knowledge / memory / groupmgr / plugin），检索只在
  目标分库内进行——不同集合互不挤占 top-k，精确度不再受无关数据干扰
- 原始文档与 uuid 由 Agent 保管，本服务只做向量化与检索
- 长文本**服务端透明分块**，Agent 契约始终是 tag ↔ 全文
- **零外部依赖**：无数据库、无独立推理服务（bge 模型进程内运行）
- 查询与写入**互不阻塞**（快照发布，读者无锁检索）；一次写入只影响本分库

## 构建

前置：Rust 工具链、CMake + make、C++ 编译器、OpenBLAS、libclang（`sudo apt install cmake make g++ libopenblas-dev libclang-dev`）。

```sh
# 1. 获取模型（自动下载 bge-small-zh-v1.5 Q8_0 GGUF，~25MB，魔搭社区）
make download

# 2. 编译
cargo build --release
```

首次构建会编译整个 llama.cpp（5–20 分钟），属正常现象。

## 运行

```sh
# 全部用默认配置（models/ 下模型、data/ 目录、127.0.0.1:3000）
cargo run --release

# 常用环境变量
RAG_MODEL_PATH=models/bge-small-zh-v1.5-q8_0.gguf
RAG_DATA_DIR=data          # 数据根目录（各分库在 data/scoops/<name>/）
RAG_PORT=3000
RAG_N_THREADS=4            # 阶段 0 实测最优
RAG_MAX_CHUNK_CHARS=260    # 单块 ≈ 256 token
RAG_OVERLAP_CHARS=50
RAG_STORE_RAW_VECTORS=true # 存原始向量，index.tvim 损坏可自愈
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 简易无鉴权 Web 控制台（基础信息 + 分页查看/增删向量） |
| PUT | `/scoops/{scoop}/tags/{tag}` | upsert：`{"text": "..."}`；长文自动分块 |
| POST | `/scoops/{scoop}/tags/batch` | 批量：`{"items": [{"tag": "...", "text": "..."}]}`，一次嵌入一次发布 |
| GET | `/scoops/{scoop}/tags?page=&page_size=` | 分页列出该分库全部 tag（UUID + 块数，上限 100 条/页） |
| POST | `/scoops/{scoop}/tags/batch-delete` | 批量删除：`{"tags": [...]}`（逐条尝试，上限 500 条） |
| GET | `/scoops/{scoop}/tags/search?q=&k=&min_score=` | 检索（**限定在 scoop 内**），返回 tag 列表 + 分数（0~1） |
| DELETE | `/scoops/{scoop}/tags/{tag}` | 删除 |
| GET | `/health` | 健康检查 + 各分库 tag/块数量 |
| GET | `/metrics` | Prometheus 指标（Grafana 采集，文本格式） |

`scoop` 白名单：`knowledge` / `memory` / `groupmgr` / `plugin`（非法值 → 400；
同一 tag 跨分库写入 → 409）。

```sh
# 示例：知识分库入库 + 检索
curl -X PUT localhost:3000/scoops/knowledge/tags/$(uuidgen) -H 'content-type: application/json' \
  -d '{"text": "这是一段用于入库的中文文本……"}'

curl 'localhost:3000/scoops/knowledge/tags/search?q=入库的中文文本&k=5'
# {"results":[{"tag":"...","score":0.87}]}
```

## 测试

```sh
cargo test                      # 单元测试（纯函数，不需要模型）
RAG_MODEL_PATH=models/bge-small-zh-v1.5-q8_0.gguf cargo test -- --ignored   # 需要模型
cargo run --release --example bench   # 基准三件套
```

## Web 控制台

`GET /` 提供简易**无鉴权**管理页面（仅限本机/内网，勿暴露公网）：

- 基础信息：模型状态 / 进程内存 / 各分库规模（5s 自动刷新）
- 分库切换（knowledge / memory / groupmgr / plugin），分页查看全部 tag（UUID + 块数）
- 添加/更新向量（tag 留空自动生成 UUID）、单条删除、勾选批量删除

```sh
# 浏览器打开
curl localhost:3000/ | head   # 或直接浏览器访问
```

## 监控指标（Prometheus / Grafana）

`GET /metrics` 输出 Prometheus 文本格式指标（前缀 `rag_`，无鉴权）：

| 指标 | 类型 | 标签 | 说明 |
|---|---|---|---|
| `rag_http_requests_total` | counter | method/path/status | HTTP 请求数（path 为路由模板，低基数） |
| `rag_http_request_duration_seconds` | histogram | method/path | HTTP 请求耗时 |
| `rag_search_total` / `rag_search_duration_seconds` / `rag_search_hits_total` / `rag_search_errors_total` | counter/histogram | scoop | 检索量/耗时/命中数分布/错误 |
| `rag_write_total` / `rag_write_duration_seconds` / `rag_write_errors_total` | counter/histogram | op(upsert/batch/delete)/scoop | 写入量/耗时/失败 |
| `rag_write_chunks_total` | counter | scoop | 写入块向量累计 |
| `rag_embed_requests_total` / `rag_embed_texts_total` / `rag_embed_duration_seconds` / `rag_embed_errors_total` | counter/histogram | — | 嵌入吞吐/耗时/失败 |
| `rag_tags` / `rag_chunks` | gauge | scoop | 各分库规模（scrape 时实时） |
| `rag_embedder_ready` | gauge | — | 嵌入模型就绪（0/1） |

```yaml
# Prometheus scrape_configs 示例
- job_name: juan-niang-rag
  scrape_interval: 15s
  static_configs:
    - targets: ['127.0.0.1:3000']
    metrics_path: /metrics
```

Grafana 建议面板：检索 P95（`histogram_quantile(0.95, sum(rate(rag_search_duration_seconds_bucket[5m])) by (le, scoop))`）、
写入失败率、各分库规模（`rag_tags`/`rag_chunks`）、嵌入模型就绪告警（`rag_embedder_ready == 0`）。

## 数据文件

```
data/scoops/<scoop>/
  index.tvim   # 压缩向量 + u64 块 id 映射（按分库独立）
  tags.bin     # tag→块id 映射 + next_id + 原始向量（重建用，按分库独立）
```

每个分库独立原子快照写（临时文件 + rename），一次写入只影响本分库；崩溃最多丢
最后一次写入；`.tvim` 损坏时从对应 `tags.bin` 自动重建。未知目录（不在白名单）
启动时告警跳过。

## 目录结构

```
src/
  main.rs          # 入口：配置 → 各分库存储加载 → 写者 → HTTP
  config.rs        # 环境变量配置
  embedding.rs     # 嵌入线程（llama.cpp，前缀/归一化/批量/截断/预热）
  chunker.rs       # 服务端内部分块
  vector_index.rs  # turbovec IdMapIndex 封装
  store.rs         # Scoop 白名单 + StoreSet（多分库）+ TagStore + tag 归属注册表
  writer.rs        # 写队列 + 按分库 COW 快照发布（读写互不阻塞）
  search.rs        # 块级检索 → 按 tag 聚合
  api.rs           # axum 路由（/scoops/{scoop}/...）
tests/e2e.rs       # 端到端（需模型）
```

详细设计见 [docs/architecture.md](docs/architecture.md)。

## 文档导航

| 文档 | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 项目架构：组件、核心机制、数据模型、决策记录、已知限制 |
| [docs/API.md](docs/API.md) | HTTP API 规范（对应 [api/openapi.yaml](api/openapi.yaml)） |
| [docs/development.md](docs/development.md) | 开发说明：环境、命令、代码规范、测试、协作约定 |
| [docs/deployment.md](docs/deployment.md) | 部署说明：本地/Docker、备份恢复、运维、故障排查 |
| [changelog/](changelog/) | 变更日志（每版本一个文件） |
