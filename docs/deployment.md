# 部署说明

本服务的部署、运维、备份恢复与故障排查指南。单机单进程架构，部署 = 一个二进制 + 模型文件 + 数据目录。

## 1. 前置条件

- Linux（内存读取依赖 `/proc`，其他平台 `/info` 的 memory 字段为 null）
- 模型文件：`bge-small-zh-v1.5-q8_0.gguf`（Q8_0，~25MB，`make download` 自动获取）
- 磁盘：数据目录随库增长（10 万 tag 约 400–650MB）

## 2. 本地部署

```sh
# 1. 获取模型（自动下载到 models/）
make download

# 2. 编译（首次 5–20 分钟，需 cmake/make/g++/libopenblas-dev/libclang-dev）
make build-release

# 3. 启动（默认 127.0.0.1:3000）
RAG_DATA_DIR=/var/lib/rag-service make run

# 4. 验证
curl localhost:3000/health   # {"status":"ok",...}
curl localhost:3000/info     # model.ready 应为 true
```

### 2.1 systemd 服务（推荐）

`/etc/systemd/system/rag-service.service`：

```ini
[Unit]
Description=JuanNiang RAG Service
After=network.target

[Service]
User=rag
Group=rag
ExecStart=/usr/local/bin/rag-service
Environment=RAG_MODEL_PATH=/opt/rag/models/bge-small-zh-v1.5-q8_0.gguf
Environment=RAG_DATA_DIR=/var/lib/rag-service
Environment=RAG_HOST=127.0.0.1
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```sh
sudo systemctl daemon-reload && sudo systemctl enable --now rag-service
```

## 3. Docker 部署

镜像内**已包含模型**（构建时从 `models/` 拷入，需先准备模型文件）；数据目录挂载卷持久化：

```sh
# 0. 准备模型（二选一）：
#    a. 已有模型文件：放进项目根 models/
#    b. 从魔搭社区下载：make download

# 1. 构建
# make build-release || true   # 本地可先验证编译
# 或直接：
docker build -t rag-service .

# 2. 运行（镜像内置模型；也可挂载卷覆盖 /app/models）
docker run -d --name rag-service --restart unless-stopped \
  -p 3000:3000 \
  -v rag-data:/app/data \
  rag-service

# 3. 验证
curl localhost:3000/health
curl localhost:3000/info   # model.ready 应为 true
```

### 3.1 国内网络加速（已内置）

Dockerfile 已针对国内网络做全链路加速，构建时无需额外配置：

| 环节 | 加速方式 |
|---|---|
| 基础镜像 | 加速域名 `docker.jiaxin.site/library/`（对应 Docker Hub 官方镜像） |
| apt 依赖 | 清华源 `mirrors.tuna.tsinghua.edu.cn`（builder 与 runtime 两个阶段） |
| cargo 依赖 | 清华 sparse 源（`sparse+https://mirrors.tuna.tsinghua.edu.cn/crates.io-index/`） |

如需切换/关闭加速（如加速站失效），修改 `Dockerfile` 中的 `FROM` 域名与源替换行即可。

- 镜像内默认 `RAG_HOST=0.0.0.0`（容器内必须监听所有接口）
- `HEALTHCHECK` 每 30s 探测 `/health`，失败自动重启（配合 `--restart`）
- 数据卷 `rag-data` 持久化，删除容器不丢数据

## 4. 数据管理

### 备份

数据 = `data/` 目录下两个文件，**停止服务后整体复制**最安全：

```sh
systemctl stop rag-service
tar czf rag-backup-$(date +%F).tar.gz /var/lib/rag-service/
systemctl start rag-service
```

不停止服务的在线备份亦可（快照写入是原子的，最多差最后一次写入），但需保证复制期间无写操作。

### 恢复

```sh
systemctl stop rag-service
tar xzf rag-backup-xxx.tar.gz -C /
systemctl start rag-service
# 启动日志出现 "存储就绪: N tag" 即恢复成功；若 index.tvim 损坏，
# 服务会用 tags.bin 原始向量自动重建（前提：RAG_STORE_RAW_VECTORS=true）
```

## 5. 运维监控

| 手段 | 内容 |
|---|---|
| `GET /health` | 存活 + tag/块规模 |
| `GET /info` | 模型就绪状态、进程内存（`memory.rss_kb`）、规模 |
| 日志 | 启动/预热/写入/删除事件；"触发重建"提示索引自愈 |

内存监控建议：`memory.rss_kb` 随库规模线性增长（10 万 tag ≈ 800MB–1GB），设定告警阈值；接近上限时评估 `RAG_STORE_RAW_VECTORS=false`（省 ~600MB，代价见下）。

## 6. 升级流程

```sh
# 1. 备份数据（见 §4）
# 2. 停服
systemctl stop rag-service
# 3. 替换二进制 / 重新构建镜像
# 4. 启动并验证
systemctl start rag-service
curl localhost:3000/info    # 确认 model.ready=true、tags/chunks 与备份一致
```

## 7. 故障排查

| 现象 | 排查 |
|---|---|
| 启动日志"模型加载失败" | 模型路径错误 / 文件损坏 / 非 GGUF；用 `RAG_MODEL_PATH` 指向正确文件；`/info` 的 `model.error` 会给出原因 |
| 写入返回 500"嵌入服务错误" | 模型未加载成功（降级模式）；查 `/info` 的 `model.ready` |
| 检索结果异常 | 分数为相似度（0~1）；`min_score` 阈值不当；确认查询文本非空 |
| 启动日志"触发重建" | 正常自愈：索引与快照不一致（崩溃残留/文件损坏）；若同时报"原始向量未持久化"，说明 `RAG_STORE_RAW_VECTORS=false` 且 `.tvim` 损坏，需 Agent 重推全文 |
| 端口占用 | 换 `RAG_PORT` 或停掉占用进程 |
| 内存持续增长 | 属预期（随库线性）；检查是否误开 `RAG_STORE_RAW_VECTORS` 或库规模超预期 |
| 服务无响应 | 查日志；`health` 与 `info` 均不可用时检查进程/容器状态（systemd/docker ps） |

## 8. 安全注意

- 服务**无认证**：默认监听 `127.0.0.1`（本地）；暴露公网前必须置于网关/反向代理之后（如 Caddy/Nginx + 鉴权）
- Docker 部署若需公网访问，同样经反向代理，不要直接映射端口到公网
