#!/usr/bin/env python3
"""从魔搭社区（ModelScope）下载模型文件（默认找 .gguf）。

下载方式：优先用 modelscope SDK（pip install modelscope，支持断点/鉴权）；
SDK 未安装时自动回退到纯标准库 HTTP 实现（零依赖）。

用法：
    python3 scripts/download_model.py                          # 默认下载 bge q8 模型
    python3 scripts/download_model.py --preset granite         # 下载 granite ONNX（RAG_BACKEND=onnx 用）
    python3 scripts/download_model.py --model-id <模型ID>
    python3 scripts/download_model.py --model-id <模型ID> --file 指定文件名.gguf
    python3 scripts/download_model.py --output models/ --force

预设：
- bge（默认）：魔搭 gubanjie/bge-small-zh-v1.5-q8_0.gguf（bge-small-zh-v1.5 Q8_0，~25MB）
- granite：onnx-community/granite-embedding-97m-multilingual-r2-ONNX 的 int8 量化
  变体（~98MB）+ tokenizer.json，保存为 <output>/granite/{model.onnx,tokenizer.json,
  config.json}；granite 模型卡见仓库 config.json（hidden_size=384，CLS 池化）。

服务器部署（根 docker-compose 把 data/rag-models 挂到容器 /app/models）：
    python3 scripts/download_model.py --preset granite --output ../../data/rag-models
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

# 默认模型（魔搭社区 ID）
DEFAULT_MODEL_ID = "gubanjie/bge-small-zh-v1.5-q8_0.gguf"

# 魔搭 HTTP API 端点（SDK 不可用时的回退路径）
FILES_API = "https://modelscope.cn/api/v1/models/{model_id}/repo/files?Revision={rev}&Recursive=true"
RESOLVE_URL = "https://modelscope.cn/models/{model_id}/resolve/{rev}/{path}"


def have_sdk() -> bool:
    """是否安装了 modelscope SDK（导入链依赖 torch 等，任何导入异常都视为不可用并回退 HTTP）。"""
    try:
        import modelscope  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - SDK 依赖损坏（如 CUDA 缺失）时回退零依赖 HTTP
        return False


def sdk_download(model_id: str, output: str, revision: str) -> str:
    """用 modelscope SDK 下载整个仓库到 output。"""
    from modelscope import snapshot_download
    print(f"[SDK] snapshot_download({model_id}) → {output}")
    return snapshot_download(model_id, revision=revision, local_dir=output)


def api_get(url: str) -> dict:
    """GET 并解析 JSON，带基本错误处理。"""
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.load(resp)
    except Exception as e:  # noqa: BLE001 - 统一转为用户可读错误
        sys.exit(f"[错误] 请求魔搭 API 失败: {e}\n       URL: {url}")


def list_gguf_files(model_id: str, revision: str) -> list[dict]:
    """列出模型仓库里的 .gguf 文件（返回 [{path, size}]）。"""
    data = api_get(FILES_API.format(model_id=model_id, rev=revision))
    files = (data.get("Data") or {}).get("Files") or []
    gguf = [f for f in files if str(f.get("Path", "")).lower().endswith(".gguf")]
    if not gguf:
        # 打印响应结构便于排查（API 字段可能随版本变化）
        sys.exit(
            f"[错误] 在 {model_id} 中未找到 .gguf 文件。\n"
            f"       原始响应片段: {json.dumps(data, ensure_ascii=False)[:300]}"
        )
    return gguf


def download(url: str, dest: str, force: bool = False) -> None:
    """分块下载到 dest；文件已存在且非 force 时跳过。"""
    if os.path.exists(dest) and not force:
        print(f"[跳过] {dest} 已存在（--force 可覆盖）")
        return

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"[下载] {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1024 * 256):
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                print(f"\r[进度] {done / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB ({pct}%)", end="", flush=True)
    print(f"\n[完成] {dest}（{os.path.getsize(dest) / 1024 / 1024:.1f} MB）")


def verify_sha256(path: str, expected: str) -> None:
    """下载后校验 SHA-256（可选）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected.lower():
        sys.exit(f"[错误] SHA-256 校验失败: 期望 {expected}，实际 {actual}")
    print(f"[校验] SHA-256 一致: {actual}")


# granite ONNX 预设（RAG_BACKEND=onnx）：int8 量化变体 + tokenizer + 模型卡 config。
# 模型仓：onnx-community/granite-embedding-97m-multilingual-r2-ONNX（ModelScope 镜像）
GRANITE_MODEL_ID = "onnx-community/granite-embedding-97m-multilingual-r2-ONNX"
GRANITE_DIR = "granite"
# (仓库内路径, 保存文件名)；int8 为 CPU 部署优先变体（98MB，动态 int8 量化）
GRANITE_FILES = [
    ("onnx/model_int8.onnx", "model.onnx"),
    ("tokenizer.json", "tokenizer.json"),
    ("config.json", "config.json"),
]


def download_granite(output: str, revision: str, force: bool) -> None:
    """下载 granite ONNX 预设到 <output>/granite/。

    注意：不用 SDK snapshot_download——仓库含全部量化变体（~1.5GB），
    HTTP 逐文件下载只取所需三个文件。
    """
    dest_dir = os.path.join(output, GRANITE_DIR)
    print(f"[查找] {GRANITE_MODEL_ID} 文件清单…")
    data = api_get(FILES_API.format(model_id=GRANITE_MODEL_ID, rev=revision))
    repo_files = {f["Path"]: f for f in (data.get("Data") or {}).get("Files") or []}
    for repo_path, name in GRANITE_FILES:
        info = repo_files.get(repo_path)
        if info is None:
            sys.exit(
                f"[错误] {GRANITE_MODEL_ID} 中未找到 {repo_path}。\n"
                f"       仓库文件: {sorted(repo_files)[:20]}"
            )
        size_mb = info.get("Size", 0) / 1024 / 1024
        print(f"       - {repo_path}（{size_mb:.1f} MB）")

    for repo_path, name in GRANITE_FILES:
        url = RESOLVE_URL.format(model_id=GRANITE_MODEL_ID, rev=revision, path=repo_path)
        download(url, os.path.join(dest_dir, name), force)
    print(f"[完成] granite ONNX 已下载到 {dest_dir}/（RAG_ONNX_MODEL_PATH 指向 model.onnx）")


def main() -> None:
    parser = argparse.ArgumentParser(description="从魔搭社区下载模型（bge GGUF / granite ONNX 预设）")
    parser.add_argument(
        "--preset",
        choices=["bge", "granite"],
        default="bge",
        help="下载预设：bge=llama.cpp 用 GGUF（默认）；granite=ONNX 后端用（int8+tokenizer）",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"魔搭模型 ID（默认 {DEFAULT_MODEL_ID}；仅 bge 预设生效）",
    )
    parser.add_argument("--file", default=None, help="指定文件名；缺省自动找仓库内第一个 .gguf（仅 bge 预设）")
    parser.add_argument("--revision", default="master", help="版本（默认 master）")
    parser.add_argument("--output", default="models", help="保存目录（默认 models/）")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    parser.add_argument("--sha256", default=None, help="可选：下载后校验 SHA-256（仅单文件下载）")
    parser.add_argument("--sdk", action="store_true", help="强制使用 modelscope SDK（需 pip install modelscope）")
    args = parser.parse_args()

    if args.preset == "granite":
        download_granite(args.output, args.revision, args.force)
        return

    # SDK 优先；未安装且未强制时回退 HTTP
    if have_sdk() or args.sdk:
        sdk_download(args.model_id, args.output, args.revision)
        print(f"[完成] 模型已下载到 {args.output}/")
        return

    print("[提示] 未安装 modelscope SDK，使用零依赖 HTTP 方式下载（pip install modelscope 可启用 SDK 路径）")
    if args.file:
        files = [{"Path": args.file, "Size": 0}]
    else:
        print(f"[查找] {args.model_id} 中的 .gguf 文件…")
        files = list_gguf_files(args.model_id, args.revision)
        for f in files:
            print(f"       - {f['Path']}（{f.get('Size', 0) / 1024 / 1024:.1f} MB）")
        if len(files) > 1:
            print(f"[提示] 共 {len(files)} 个 .gguf，默认下载第一个；可用 --file 指定")

    target = files[0]["Path"]
    dest = os.path.join(args.output, os.path.basename(target))
    url = RESOLVE_URL.format(model_id=args.model_id, rev=args.revision, path=target)
    download(url, dest, args.force)

    if args.sha256:
        verify_sha256(dest, args.sha256)


if __name__ == "__main__":
    main()
