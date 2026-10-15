# JuanNiang-RAG-Service Makefile
# 常用入口：make help 查看全部命令
#
# 关键变量可覆盖：
#   make run MODEL=other/model.gguf

CARGO      ?= cargo
MODEL      ?= models/bge-small-zh-v1.5-q8_0.gguf
OPENAPI    ?= api/openapi.yaml
DEV_YAML   ?= dev.yaml
MODEL_ID   ?= gubanjie/bge-small-zh-v1.5-q8_0.gguf

# dev.yaml → 环境变量前缀（KEY=VALUE ...），供 make dev 注入。
# 文件缺失或解析失败时为空，回退代码内默认配置。
DEV_ENV := $(shell python3 -c 'import yaml; d = yaml.safe_load(open("$(DEV_YAML)")) or {}; print(" ".join(f"{k}={v}" for k, v in d.items()))' 2>/dev/null)

.PHONY: help build build-release run check fmt fmt-check clippy \
        test test-unit test-e2e bench download download-onnx api-lint api-preview clean

## 构建与运行
build: ## debug 构建
	$(CARGO) build

build-release: ## release 构建（首次编译 llama.cpp 需 5-20 分钟）
	$(CARGO) build --release

run: ## 以 release 模式启动服务（默认 127.0.0.1:3000）
	$(CARGO) run --release

dev: ## dev 模式：从 dev.yaml 读取环境变量启动（debug 构建，便于调试）
	@test -f $(DEV_YAML) || (echo "未找到 $(DEV_YAML)，已从模板创建，请修改后重试" && cp dev.yaml.tmp $(DEV_YAML))
	$(DEV_ENV) $(CARGO) run

check: ## 快速类型检查
	$(CARGO) check

## 代码质量
fmt: ## 格式化代码
	$(CARGO) fmt

fmt-check: ## 检查格式（CI 用）
	$(CARGO) fmt --check

clippy: ## 静态检查（零警告目标）
	$(CARGO) clippy --all-targets -- -D warnings

## 测试
test: test-unit ## 默认跑单元测试（不需要模型）

test-unit: ## 单元测试（纯函数，不需要模型）
	$(CARGO) test

test-e2e: ## 端到端 + 批量一致性测试（需要模型文件）
	RAG_MODEL_PATH=$(MODEL) $(CARGO) test -- --ignored

## 基准
bench: ## 基准三件套（需要模型文件，无模型时自动跳过嵌入部分）
	$(CARGO) run --release --example bench $(MODEL)

download: ## 下载 bge 模型到 models/（默认魔搭 gubanjie/bge-small-zh-v1.5-q8_0.gguf，~25MB）
	python3 scripts/download_model.py --model-id $(MODEL_ID) --output models

download-onnx: ## 下载 granite ONNX 到 models/granite/（RAG_BACKEND=onnx 用，int8 ~98MB）
	python3 scripts/download_model.py --preset granite --output models

## API 规范
api-lint: ## 校验 OpenAPI 规范（redocly）
	npx --yes @redocly/cli@latest lint $(OPENAPI)

api-preview: ## 交互式预览 API 文档
	npx --yes @redocly/cli@latest preview-docs $(OPENAPI)

## 其他
clean: ## 清理构建产物（保留数据与模型）
	$(CARGO) clean

## 帮助（自文档化：读取本文件中 ## 注释）
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
