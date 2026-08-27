# Changelog

本项目的变更日志。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
每个版本一个文件：`CHANGELOG-<版本>-<日期>.md`。

## [2.0.0] - 2026-08-26

分库（scoop）机制：向量按功能块物理隔离存储与检索，解决此前知识/记忆/群管理
共库导致的 top-k 互相挤占、检索精确度下降问题。

> **Breaking**：旧路由 `/tags/...` 已替换为 `/scoops/{scoop}/tags/...`；
> 旧数据文件布局（`data/index.tvim` / `data/tags.bin`）不再读取，需清空或迁移到
> `data/scoops/<name>/` 布局。JuanNiang-Neo 客户端已同步升级，两端需一起发布。

### Added

- **Scoop 分库**：`knowledge` / `memory` / `groupmgr` / `plugin` 四个白名单分库，
  每库独立 `TagStore`、独立快照、独立双文件持久化（`data/scoops/<name>/`）
- **检索限定分库**：`GET /scoops/{scoop}/tags/search` 只在目标分库内 top-k，
  外来集合向量从物理上不参与检索
- **tag 全局归属注册表**：同一 tag 跨分库写入/删除返回 409（重启时从各库
  `tags.bin` 自动重建），杜绝"删除漏删出死数据"
- **per-scoop 状态**：`/health`、`/info` 返回 `scoops.<name>.{tags,chunks}` 统计
- **启动扫描加载**：`StoreSet::load` 枚举 `scoops/` 目录，只加载白名单内分库，
  未知目录告警跳过（防旧版本遗留脏数据混入）；白名单中缺失的分库自动初始化空库

### Changed

- **API 路由**：所有 tag 路由增加 scoop 路径段（upsert / batch / search / delete），
  非法 scoop 名 → 400
- **写者按 scoop 路由**：写命令携带 scoop，一次写入只持久化 + COW 发布本分库
  （其余分库快照与磁盘零参与）；批量 upsert 仍整批一次嵌入 + 一次发布
- **服务端恢复粒度**：自愈重建按分库独立生效，故障影响面收窄
- **客户端（JuanNiang-Neo）**：
  - `ragtag` 包新增 `Scoop*` 常量；`infrastructure/rag/handler` 全部方法带 scoop
  - 知识/记忆检索的候选集职责从"过滤外来 tag"收敛为"tag → 本地 ID 反查映射"
    （UUID v5 不可逆，映射表仍必需，但不再承担集合过滤）
  - 检索 k 值收窄：知识 15→10、记忆 20→10、群管理 30→15（召回域已限定分库内）
  - `jn.rag` 插件 API 统一写入独立 `plugin` 分库，与业务集合物理隔离

### Performance

| 维度 | 分库前 | 分库后 |
|---|---|---|
| 检索召回域 | 全库（两三千~数万向量） | 单分库（百~千级） |
| COW 快照成本 | 全库序列化往返（30 万 ~54ms） | 仅本分库（按集合规模，低一个量级） |
| 持久化写盘 | 全库双文件 | 本分库双文件 |
| 内存 | 一份大索引快照 | 多份小索引快照，总量 ≈ 原值 |

### Docs

- `docs/API.md`：v2 分库 API（路由、409 语义、per-scoop 状态）
- `docs/architecture.md`：分库架构（StoreSet、注册表、COW 分布）
- `docs/deployment.md`：`data/scoops/<name>/` 目录布局
- `README.md`：分库一览与数据文件说明

### 已知限制（沿用 v1.0.0）

1. **内存随库线性增长**：主要来自原始向量常驻（自愈能力换内存），10 万 tag 约 800 MB–1 GB；`RAG_STORE_RAW_VECTORS=false` 可省 ~600 MB，代价是 `.tvim` 损坏后需 Agent 重推全文
2. **tag 粒度 = 整篇文档**：长文命中返回整篇 tag，引用定位需 Agent 在自有文档中完成
3. **`n_ctx` 被 llama.cpp 向上取整至 4096**（`n_seq_max=16` 的副作用），多占 ~60 MB
4. **单机单进程**：数据在本机 `data/scoops/` 目录，不适合多实例横向扩展；备份直接复制目录