# JuanNiang-RAG-Service API 文档

> 版本：v2（分库 scoop 机制）
> 服务地址：`http://{host}:{port}`，默认 `http://127.0.0.1:3000`
> 数据格式：全部 JSON（`Content-Type: application/json`）
> 契约：Agent 保管原始文档与 uuid，本服务只做向量化、检索与删除

## 目录

- [通用约定](#通用约定)
  - [分库 scoop](#分库-scoop)
  - [分数语义](#分数语义)
  - [错误格式](#错误格式)
- [GET / —— Web 控制台](#get----web-控制台)
- [PUT /scoops/{scoop}/tags/{tag} —— 更新（upsert）](#put-scoopsscooptagstag--更新upsert)
- [POST /scoops/{scoop}/tags/batch —— 批量更新](#post-scoopsscooptagsbatch--批量更新)
- [GET /scoops/{scoop}/tags —— 分页列表](#get-scoopsscooptags--分页列表)
- [POST /scoops/{scoop}/tags/batch-delete —— 批量删除](#post-scoopsscooptagsbatch-delete--批量删除)
- [GET /scoops/{scoop}/tags/search —— 检索](#get-scoopsscooptagssearch--检索)
- [DELETE /scoops/{scoop}/tags/{tag} —— 删除](#delete-scoopsscooptagstag--删除)
- [GET /health —— 健康检查](#get-health--健康检查)
- [GET /info —— 服务信息](#get-info--服务信息)
- [示例：完整流程](#示例完整流程)
- [行为边界与注意事项](#行为边界与注意事项)

---

## 通用约定

### 分库 scoop

所有业务路由都以 **scoop（分库）** 为第一维：向量按功能块物理隔离存储，
**检索只在目标 scoop 内进行**——不同集合的向量互不挤占 top-k，检索精确度
不再受无关数据干扰。

| scoop | 内容 | 使用方（JuanNiang-Neo） |
|---|---|---|
| `knowledge` | 知识库条目 | 对话前知识召回（ragtag `k:` 前缀） |
| `memory` | 长期记忆条目 | 记忆语义召回（`m:`） |
| `groupmgr` | 群管理黑白语录/词条（黑白同库：一次检索取两边最优命中） | 违规语义核实（`s:` / `wt:`） |
| `plugin` | 插件 `jn.rag` 通用 API 的默认库 | Lua 插件（任意 tag） |

白名单由服务端枚举硬编码（`src/store.rs`），非法 scoop 名 → `400`。
新增 scoop = 服务端枚举加一项 + 客户端常量加一项，**不允许**任意字符串分库
（拼写错误会立即 400 拒绝，而不是静默建一个空库导致数据分散）。

添加/删除/检索某个 scoop 内的数据，对**其他 scoop 零影响**（快照、磁盘、
COW 拷贝均按 scoop 独立）。

### 分数语义

- 检索返回的 `score` 是**相似度**，范围 **[0, 1]**，越高越相似
- 完全相同的文本 ≈ 0.99+；语义相关通常 0.6–0.95；无关通常 < 0.6
- 由底层向量相似度映射而来（`(raw + 1) / 2`），阈值建议从 `0.5` 起步实测校准

### 错误格式

所有错误响应统一为：

```json
{ "error": "错误描述" }
```

| 状态码 | 含义 | 触发场景 |
|---|---|---|
| 400 | 参数错误 | `text`/`q` 为空、`items` 为空、**未知 scoop** |
| 404 | 资源不存在 | 删除不存在的 tag |
| 409 | 归属冲突 | 同一 tag 跨 scoop 重复写入/删除（tag 全局只允许归属一个 scoop） |
| 500 | 服务内部错误 | 模型未加载、存储损坏、嵌入失败等 |

---

## GET / —— Web 控制台

简易**无鉴权**管理页面（内嵌 HTML，仅限本机/内网使用，勿暴露公网）：

- 基础信息：模型状态/内存/各分库规模（5s 自动刷新）
- 分页查看各分库全部 tag（UUID + 块数，每页 20 条）
- 添加/更新向量（留空 tag 自动生成 UUID）、单条删除、勾选批量删除

```sh
# 浏览器打开
http://localhost:3000/
```

---

## PUT /scoops/{scoop}/tags/{tag} —— 更新（upsert）

`tag` 已存在 → **覆写**其全部向量；不存在 → 新建。

```
PUT /scoops/{scoop}/tags/{tag}
```

| 位置 | 参数 | 说明 |
|---|---|---|
| path | `scoop` | 分库名（见上文白名单，仅限 4 个枚举值） |
| path | `tag` | uuid，Agent 侧文档标识 |

**请求体**：

```json
{ "text": "要入库的文本内容" }
```

**响应 200**：

```json
{
  "scoop": "knowledge",
  "tag": "3af2b489-b13a-42e4-af98-fe89d0e6b00e",
  "chunk_count": 4,
  "truncated": false
}
```

| 字段 | 说明 |
|---|---|
| `scoop` | 入库的分库（原样返回） |
| `tag` | 入库的 tag（原样返回） |
| `chunk_count` | 实际入库的块数。≤512 token 为 1；长文自动分块（256 token/块）后 >1 |
| `truncated` | 是否存在被截断的块（单块超过模型上限 512 token 时，服务端截断处理） |

**curl 示例**：

```sh
curl -X PUT localhost:3000/scoops/knowledge/tags/3af2b489-b13a-42e4-af98-fe89d0e6b00e \
  -H 'content-type: application/json' \
  -d '{"text": "Rust 是一种系统编程语言，强调内存安全与零成本抽象。"}'
```

**语义**：响应返回时数据**已持久化并发布**（读己之写），之后检索立即可见。

**409 冲突示例**：同一 tag 已归属其他 scoop 时：

```sh
curl -X PUT localhost:3000/scoops/memory/tags/3af2b489-... \
  -H 'content-type: application/json' -d '{"text": "..."}'
# 409 { "error": "tag 3af2b489-... 已归属分库 knowledge，跨分库重复写入被拒绝" }
```

---

## POST /scoops/{scoop}/tags/batch —— 批量更新

多条文本一次批量 upsert：**一次嵌入推理 + 一次发布 + 一次持久化**（批量成本远低于逐条调用）。
全部条目必须隶属**同一个** scoop。

```
POST /scoops/{scoop}/tags/batch
```

**请求体**：

```json
{
  "items": [
    { "tag": "11111111-1111-1111-1111-111111111111", "text": "第一条文本" },
    { "tag": "22222222-2222-2222-2222-222222222222", "text": "第二条文本" }
  ]
}
```

**响应 200**（逐条结果，顺序与请求一致；单条失败不影响其他条目）：

```json
{
  "results": [
    {
      "tag": "11111111-1111-1111-1111-111111111111",
      "chunk_count": 1,
      "truncated": false,
      "error": null
    },
    {
      "tag": "22222222-2222-2222-2222-222222222222",
      "chunk_count": 2,
      "truncated": false,
      "error": null
    }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `error` | `null` 表示成功；否则为失败原因（如"文本为空"/"归属冲突"），该条目未入库 |
| 其余字段 | 同单条 upsert |

**curl 示例**：

```sh
curl -X POST localhost:3000/scoops/memory/tags/batch \
  -H 'content-type: application/json' \
  -d '{"items": [{"tag": "11111111-1111-1111-1111-111111111111", "text": "第一段"}, {"tag": "22222222-2222-2222-2222-222222222222", "text": "第二段"}]}'
```

**建议**：Agent 侧做首次全量同步或批量更新时使用本端点；已成功的条目即使部分失败也会落盘。

---

## GET /scoops/{scoop}/tags —— 分页列表

分页列出该分库全部 tag（UUID 字典序，稳定可复现）。

```
GET /scoops/{scoop}/tags?page=<页码>&page_size=<每页条数>
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `page` | `1` | 页码（从 1 起） |
| `page_size` | `20` | 每页条数，范围 1–100 |

**响应 200**：

```json
{
  "total": 128,
  "page": 1,
  "page_size": 20,
  "items": [
    { "tag": "3af2b489-b13a-42e4-af98-fe89d0e6b00e", "chunk_count": 4 },
    { "tag": "22222222-2222-2222-2222-222222222222", "chunk_count": 1 }
  ]
}
```

**说明**：读快照实现（无锁、不经过写者）；`chunk_count` 为该 tag 的块向量数。

---

## POST /scoops/{scoop}/tags/batch-delete —— 批量删除

逐条尝试删除（不属于本 scoop 的 tag 单条失败，不影响其他）；有成功条目时一次持久化+发布。

```
POST /scoops/{scoop}/tags/batch-delete
```

**请求体**：

```json
{ "tags": ["11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"] }
```

**响应 200**（顺序与请求一致）：

```json
{
  "results": [
    { "tag": "11111111-1111-1111-1111-111111111111", "deleted": true,  "error": null },
    { "tag": "22222222-2222-2222-2222-222222222222", "deleted": false, "error": "存储错误: tag 不存在: ..." }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `deleted` | 是否成功删除 |
| `error` | `null` 表示成功；否则为该条失败原因 |

**限制**：单次最多 500 条；`tags` 为空返回 400。

---

## GET /scoops/{scoop}/tags/search —— 检索

输入文本，返回按相似度排序的 **tag 列表**（去重聚合，一个 tag 只出现一次）。
**只在指定 scoop 内检索**，其他分库的向量不参与。

```
GET /scoops/{scoop}/tags/search?q=<文本>&k=<条数>&min_score=<阈值>
```

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| path `scoop` | ✅ | — | 分库名 |
| `q` | ✅ | — | 查询文本（空文本返回 400） |
| `k` | | `10` | 返回条数，范围 1–100（超出自动钳制） |
| `min_score` | | 无 | 相似度下限过滤（0~1），低于该值的 tag 不返回 |

**响应 200**：

```json
{
  "results": [
    { "tag": "3af2b489-b13a-42e4-af98-fe89d0e6b00e", "score": 0.888 },
    { "tag": "22222222-2222-2222-2222-222222222222", "score": 0.698 }
  ]
}
```

**curl 示例**：

```sh
curl 'localhost:3000/scoops/knowledge/tags/search?q=服务端透明分块与相似检索&k=5&min_score=0.5'
```

**说明**：

- 查询文本会自动加 bge 官方指令前缀（`为这个句子生成表示以用于检索相关文章：`），调用方无需处理
- 查询按 tag 聚合：一个 tag 有多个块命中时，取最高分块
- 命中缓存：相同的 `q` 短时间重复查询免推理（LRU，容量默认 1024，跨 scoop 共享——同文同向量）

---

## DELETE /scoops/{scoop}/tags/{tag} —— 删除

删除 tag 及其**全部**块向量（只能删除属于该 scoop 的 tag）。

```
DELETE /scoops/{scoop}/tags/{tag}
```

**响应 200**：

```json
{ "deleted": "3af2b489-b13a-42e4-af98-fe89d0e6b00e" }
```

**错误**：tag 不存在 → `404 { "error": "tag 不存在: <uuid>" }`；
tag 归属其他 scoop → `409 { "error": "..." }`

**curl 示例**：

```sh
curl -X DELETE localhost:3000/scoops/knowledge/tags/3af2b489-b13a-42e4-af98-fe89d0e6b00e
```

---

## GET /health —— 健康检查

```
GET /health
```

**响应 200**：

```json
{
  "status": "ok",
  "scoops": {
    "knowledge": { "tags": 128, "chunks": 340 },
    "memory":    { "tags": 56,  "chunks": 60 },
    "groupmgr":  { "tags": 2300, "chunks": 2300 },
    "plugin":    { "tags": 0,   "chunks": 0 }
  }
}
```

| 字段 | 说明 |
|---|---|
| `status` | `"ok"` 表示服务可用 |
| `scoops.<name>.tags` | 该分库 tag 总数 |
| `scoops.<name>.chunks` | 该分库块向量总数（长文分块后 chunks ≥ tags） |

**curl 示例**：

```sh
curl localhost:3000/health
```

---

## GET /info —— 服务信息

模型状态 + 进程内存 + per-scoop 规模，适合运维监控与集成前体检。

```
GET /info
```

**响应 200**：

```json
{
  "status": "ok",
  "model": {
    "ready": true,
    "model_name": "bge-small-zh-v1.5",
    "dim": 512,
    "n_params": 23691264,
    "n_threads": 4,
    "n_ctx": 4096,
    "error": null
  },
  "memory": {
    "rss_kb": 81244,
    "vsize_kb": 1915772
  },
  "scoops": {
    "knowledge": { "tags": 128, "chunks": 340 },
    "memory":    { "tags": 56,  "chunks": 60 },
    "groupmgr":  { "tags": 2300, "chunks": 2300 },
    "plugin":    { "tags": 0,   "chunks": 0 }
  }
}
```

| 字段 | 说明 |
|---|---|
| `model.ready` | 模型是否加载成功、可服务。`false` 时查看 `model.error` 获取失败原因 |
| `model.model_name` | GGUF 元数据里的模型名 |
| `model.dim` | 嵌入维度（bge-small-zh-v1.5 = 512） |
| `model.n_params` | 模型参数量 |
| `model.n_threads` | 推理线程数（配置 `RAG_N_THREADS`） |
| `model.n_ctx` | 实际上下文长度（llama.cpp 可能因多序列配置向上取整，如实上报） |
| `model.error` | `null` 表示正常；否则为加载/初始化失败原因 |
| `memory.rss_kb` | 进程常驻物理内存（kB）。仅 Linux（读 `/proc/self/status`），其他平台为 `null` |
| `memory.vsize_kb` | 进程虚拟内存（kB） |
| `scoops.<name>.*` | 各分库 tag 数 / 块向量数（新增分库自动出现，无需客户端感知） |

**curl 示例**：

```sh
curl localhost:3000/info
```

**典型用途**：

- 监控：`memory.rss_kb` 随库规模线性增长，可据此设定告警阈值
- 体检：入库前确认 `model.ready == true`，避免写入返回 500
- 对账：`scoops.<name>.chunks` 应与 Agent 侧该集合文档数 × 平均块数匹配

---

## 示例：完整流程

```sh
# 1. 生成一个 tag
TAG=$(uuidgen)

# 2. 入库一篇长文到 knowledge 分库（自动分块）
curl -X PUT localhost:3000/scoops/knowledge/tags/$TAG -H 'content-type: application/json' \
  -d '{"text": "这是第一段。这是第二段。……（长文）"}'

# 3. 在 knowledge 分库内检索
curl "localhost:3000/scoops/knowledge/tags/search?q=第二段的内容&k=3"

# 4. 不再需要时删除
curl -X DELETE localhost:3000/scoops/knowledge/tags/$TAG
```

---

## 行为边界与注意事项

1. **文本长度**：单文本无上限（超长自动分块），但单块超过 512 token 会被截断并置 `truncated: true`——截断会损失尾部信息，Agent 侧应避免喂入超长单段
2. **空文本**：`text` 或 `q` 为空字符串（或纯空白）返回 400
3. **幂等性**：upsert 是幂等的——同 scoop 内同 tag 重复提交相同文本，结果等价（内部先删旧块再写新块）
4. **tag 全局唯一归属**：同一 tag 只允许写入一个 scoop（跨 scoop 写入/删除 → 409）。此保证由服务端注册表在重启后从各库自动重建，无需客户端维护
5. **一致性**：写接口返回即持久化完成；服务崩溃最多丢失"最后一次未返回的写入"
6. **并发**：检索与写入互不阻塞；批量写期间检索延迟不受影响；一次写入只影响归属 scoop（其他 scoop 的快照/磁盘零参与）
7. **单机限制**：数据全部在本机 `data/scoops/<scoop>/` 目录（每 scoop 一对 `index.tvim` + `tags.bin`），备份直接复制整个 `data/` 目录