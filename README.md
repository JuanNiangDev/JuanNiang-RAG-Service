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
| PUT | `/scoops/{scoop}/tags/{tag}` | upsert：`{"text": "..."}`；长文自动分块 |
| POST | `/scoops/{scoop}/tags/batch` | 批量：`{"items": [{"tag": "...", "text": "..."}]}`，一次嵌入一次发布 |
| GET | `/scoops/{scoop}/tags/search?q=&k=&min_score=` | 检索（**限定在 scoop 内**），返回 tag 列表 + 分数（0~1） |
| DELETE | `/scoops/{scoop}/tags/{tag}` | 删除 |
| GET | `/health` | 健康检查 + 各分库 tag/块数量 |

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
