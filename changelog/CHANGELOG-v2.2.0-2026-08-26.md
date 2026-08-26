# Changelog

本项目的变更日志。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
每个版本一个文件：`CHANGELOG-<版本>-<日期>.md`。

## [2.2.0] - 2026-08-26

简易无鉴权 Web 控制台 + 分页列表/批量删除 API。

### Added

- **Web 控制台**（`GET /`，内嵌 HTML 单页）：基础信息（模型/内存/各分库规模，5s 自动刷新）、
  分库切换、分页查看全部 tag（UUID + 块数）、添加/更新向量（tag 留空自动生成 UUID）、
  单条删除、勾选批量删除
- **分页列表 API**：`GET /scoops/{scoop}/tags?page=&page_size=`——UUID 字典序分页
  （读快照无锁，page_size 上限 100），返回 `{total, page, page_size, items:[{tag, chunk_count}]}`
- **批量删除 API**：`POST /scoops/{scoop}/tags/batch-delete`——`{tags:[uuid...]}`，
  逐条尝试（不属于本 scoop 的 tag 单条失败不影响其他），有成功条目时一次持久化+发布；
  上限 500 条，返回逐条 `{tag, deleted, error}`
- **写者批量删除命令**：`Writer::batch_delete` + `WriteCmd::BatchDelete`（复用归属注册表
  校验，指标逐条按 `op=delete` 计数、耗时记整批一次）

### Docs

- `docs/API.md`：控制台 + 分页列表 + 批量删除三个端点
- `README.md`：API 表与 Web 控制台说明

### 安全提示

- 控制台与 `/metrics` 均无鉴权，仅限本机/内网使用，请勿暴露公网（页面顶部有警示横幅）。
