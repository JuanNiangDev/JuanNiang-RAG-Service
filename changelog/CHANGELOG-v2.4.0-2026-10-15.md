# Changelog

本项目的变更日志。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
每个版本一个文件：`CHANGELOG-<版本>-<日期>.md`。

## [2.4.0] - 2026-10-15

嵌入后端可插拔：新增 granite ONNX CPU 后端（`RAG_BACKEND=onnx`），bge GGUF（llama.cpp）路径原样保留，运行时二选一。

### Added

- **ONNX 后端**（`RAG_BACKEND=onnx`）：
  - 模型 `granite-embedding-97m-multilingual-r2`（onnx-community r2 导出，int8 量化 ~98MB，ModelScope 下载）；384 维、32K 上下文、CLS 池化、无查询指令前缀
  - 推理栈：`ort` 2.0.0-rc.14（CPUExecutionProvider，预编译静态链接，运行镜像零额外依赖）+ HF `tokenizers` 加载模型仓 `tokenizer.json`
  - **逐条推理**：int8 动态激活量化的 scale 依赖整个批张量，批内打包会让同文本随批组成漂移（实测 cos≈0.984）；逐条推理保证同文本逐位一致（实测 diff=0），260 字 ≈ 33ms @4 线程
  - `POST /embeddings` 诊断端点：直接返回归一化向量（backend/dim/embeddings/truncated）
  - `scripts/download_model.py --preset granite` / `make download-onnx`
- **维度随模型派生**：`RAG_DIM` 移除，空库维度与旧索引校验都以嵌入线程上报的实际输出为准（bge=512 / granite=384）；维度不匹配启动即报 `IndexDimMismatch`，绝不静默混入
- `/info` 模型状态新增 `backend` 字段

### Changed

- `Embedder` 拆分为 `run_gguf` / `run_onnx` 双后端实现（对外句柄、channel 协议、指标不变）；启动期 `wait_loaded` 阻塞等首次加载结果以获取模型维度，预热仍后台进行
- bge 查询指令前缀仅 gguf 后端生效（granite 模型卡未定义查询指令）
- `RAG_N_CTX` 在 onnx 后端语义为单条文本 token 截断上限（granite 原生 32K，服务端分块已限制输入长度）

### Compatibility

- `RAG_BACKEND` 默认 `gguf`：bge 路径行为与 2.3.x 完全一致（无 granite 模型也可运行）
- 已有 512 维索引 + gguf 后端：不受影响；切换 onnx 后端时启动报维度不匹配（需切回或重建数据目录）

### Docs

- `docs/API.md` / `api/openapi.yaml`（2.1.0）：`POST /embeddings`、`model.backend`
- `docs/development.md` / `docs/architecture.md`：双后端配置表、决策记录（后端开关/维度派生/逐条推理）
