# 开发说明

面向本项目开发者的环境准备、常用命令、代码规范与协作约定。

## 1. 环境要求

| 依赖 | 版本 | 用途 |
|---|---|---|
| Rust 工具链 | 1.85+（edition 2024） | 编译 |
| CMake | 任意现代版本 | 编译 llama.cpp（构建脚本调用） |
| make | 任意 | cmake 默认 "Unix Makefiles" 生成器依赖（`make`） |
| C++ 编译器（g++） | 支持 C++17 | 编译 llama.cpp 与 onig |
| OpenBLAS 开发库 | 任意 | llama.cpp 矩阵乘法加速（`libopenblas-dev`） |
| libclang 开发库 | 任意 | llama-cpp-sys-2 构建时用 bindgen 生成 FFI 绑定（`libclang-dev`） |
| 网络（构建期） | 可达 cdn.pyke.io | ort 构建期拉取预编译 ONNX Runtime 静态库（一次性，~10MB） |
| 模型文件 | bge GGUF（~25MB）/ granite ONNX（~98MB） | 按后端二选一（`make download` / `make download-onnx`） |

Debian/Ubuntu 一键安装系统依赖：

```sh
sudo apt install cmake make g++ libopenblas-dev libclang-dev
```

获取模型（两种后端独立，按需下载）：

```sh
# bge GGUF（RAG_BACKEND=gguf，默认后端）：下载到 models/（魔搭 gubanjie/bge-small-zh-v1.5-q8_0.gguf，~25MB）
make download

# granite ONNX（RAG_BACKEND=onnx）：int8 量化 + tokenizer 下载到 models/granite/
# （onnx-community/granite-embedding-97m-multilingual-r2-ONNX，~98MB）
make download-onnx

# 服务器部署（根 docker-compose 把 data/rag-models 挂载为容器 /app/models）：
python3 scripts/download_model.py --preset granite --output ../../data/rag-models
```

模型路径可用 `RAG_MODEL_PATH` / `RAG_ONNX_MODEL_PATH` / `RAG_ONNX_TOKENIZER_PATH` 覆盖；
下载脚本支持换仓库/校验，见 `scripts/download_model.py --help`。

## 2. 常用命令

```sh
make help            # 全部命令清单（自文档化）
make download        # 下载 bge 模型到 models/（魔搭，~25MB）
make download-onnx   # 下载 granite ONNX 到 models/granite/（RAG_BACKEND=onnx 用，~98MB）
make build           # debug 构建
make build-release   # release 构建
make run             # release 模式启动（默认 127.0.0.1:3000）
make test            # 单元测试（无需模型）
make test-e2e        # 集成测试（需要模型）
make bench           # 基准三件套（需要模型，缺模型自动跳过嵌入部分）
make fmt / fmt-check # 格式化 / 检查
make clippy          # 静态检查（-D warnings，零警告目标）
make api-lint        # redocly 校验 api/openapi.yaml
```

**首次构建 5–20 分钟**（编译整个 llama.cpp 的 C++ 源码），之后增量构建秒级。

## 3. 目录结构

```
src/
  main.rs          # 入口：配置 → 存储加载 → 写者 → HTTP
  lib.rs           # 库入口（模块声明，集成测试复用）
  config.rs        # 环境变量配置（全部带默认值）
  embedding.rs     # 推理层：gguf（llama.cpp）/onnx（ort）双后端，专属线程 + channel；
  chunker.rs       # 服务端分块（纯函数，可独立测试）
  vector_index.rs  # IdMapIndex 封装：检索/删除/COW 拷贝
  store.rs         # 存储：tag↔块 1:N、双快照持久化、自愈重建
  writer.rs        # 写者任务：写队列 + 快照发布
  search.rs        # 聚合检索（纯函数）
  api.rs           # axum 路由、错误映射、查询缓存
tests/e2e.rs       # 端到端（#[ignore]，需模型）
examples/bench.rs  # 基准三件套
scripts/download_model.py  # 魔搭模型下载（bge GGUF / granite ONNX，make download[-onnx] 调用）
api/openapi.yaml   # OpenAPI 规范（API 变更必须同步）
```
changelog/         # 每版本一个 CHANGELOG-<版本>-<日期>.md
docs/              # 架构/API/开发/部署文档
```

## 4. 配置项（环境变量）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RAG_BACKEND` | `gguf` | 嵌入后端（`gguf`=llama.cpp+bge / `onnx`=ort CPU+granite） |
| `RAG_MODEL_PATH` | `models/bge-small-zh-v1.5-q8_0.gguf` | bge GGUF 路径（backend=gguf 时使用） |
| `RAG_ONNX_MODEL_PATH` | `models/granite/model.onnx` | granite ONNX 路径（backend=onnx 时使用） |
| `RAG_ONNX_TOKENIZER_PATH` | `models/granite/tokenizer.json` | granite 分词器路径（backend=onnx 时使用） |
| `RAG_DATA_DIR` | `data` | 数据目录（index.tvim + tags.bin） |
| `RAG_HOST` / `RAG_PORT` | `127.0.0.1` / `3000` | HTTP 监听 |
| `RAG_N_THREADS` | `4` | 推理线程数（实测 4 最优；两后端共用） |
| `RAG_N_CTX` | `512` | gguf 上下文长度（bge 上限）/ onnx 单条截断上限 |
| `RAG_BIT_WIDTH` | `4` | 量化位宽（嵌入维度随模型派生，不再可配：bge=512 / granite=384） |
| `RAG_MAX_CHUNK_CHARS` / `RAG_OVERLAP_CHARS` | `260` / `50` | 分块参数（≈256/50 token） |
| `RAG_STORE_RAW_VECTORS` | `true` | 存原始向量（自愈能力，内存 +2KB/块） |
| `RAG_LRU_CAPACITY` | `1024` | 查询 embedding 缓存条数 |

**后端切换注意**：向量维度随模型派生（bge=512 / granite=384）。切换 `RAG_BACKEND`
后，旧后端生成的索引维度与新模型不匹配，启动会明确报错
（`IndexDimMismatch`）——需切回原后端，或备份后清空 `data/scoops/` 重建。

## 5. 代码规范

- **格式**：`make fmt-check` 通过（cargo fmt）
- **静态检查**：`make clippy` 零警告（`-D warnings`）
- **错误链**：模块内错误（`EmbedError`/`VectorError`/`StoreError`/`WriterError`）→ 汇总为 `ServiceError`（thiserror 派生）→ API 层映射 HTTP 状态码
- **模块职责**：业务逻辑放纯函数（分块/聚合/归一化）便于单测；IO/并发放封装层
- **测试约定**：纯函数必须有单元测试；gguf 全流程测试标 `#[ignore]`（`make test-e2e` 运行）；ONNX 测试运行时探测模型文件，缺失自动跳过；测试数据用 `std::env::temp_dir()` 自清理

## 6. 测试与基准

```sh
make test            # 30 个单元测试（分块/聚合/存储/索引/嵌入纯函数；granite 模型存在时含 ONNX 路径测试）
make test-e2e        # 批量 vs 单条一致性（余弦 > 0.999）+ 端到端全流程（bge gguf）
make bench           # 嵌入延迟（含线程数扫描）/ 30 万向量检索 / COW 发布
```

改到推理/索引相关代码后，至少跑 `make test-e2e` 确认无回归。

## 7. 协作约定

- **提交**：小改动直接 `git add` + 描述性 commit message（`feat:` / `fix:` / `docs:` / `chore:` 前缀）
- **变更日志**：新功能/修复同步写入 `changelog/CHANGELOG-<版本>-<日期>.md`（Keep a Changelog 格式）
- **API 变更**：必须同步 `api/openapi.yaml`（`make api-lint` 校验）与 `docs/API.md`
- **架构变更**：同步 `docs/architecture.md`；若推翻设计稿决策，在架构文档"决策记录"表中更新

## 8. 调试

- 日志：启动日志见终端（tracing），含模型加载、预热、存储规模
- 运行态体检：`curl localhost:3000/info`（后端 / 模型状态 / 维度 / 内存 / 规模）
- 嵌入诊断：`curl -d '{"texts":["测试"]}' localhost:3000/embeddings`（维度/一致性/延迟）
- llama.cpp 内部日志会打到 stderr（模型加载细节、图构建），排查推理问题时可查看
- 数据自愈：启动时若索引与快照不一致，日志会打印"触发重建"
