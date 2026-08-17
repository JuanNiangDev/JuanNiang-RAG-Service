# syntax=docker/dockerfile:1

# ========== 构建阶段 ==========
# 说明：
# - llama.cpp 由构建脚本编译 C++ 源码，需要 cmake + make + C++ 编译器 + OpenBLAS；
#   bindgen 生成 FFI 绑定还需 libclang（见下方 LIBCLANG_PATH）
# - 首次构建 5-20 分钟属正常；依赖层已独立缓存，改业务代码只增量编译
# - 基础镜像使用国内加速域名（docker.jiaxin.site/library/ 对应 Docker Hub 官方镜像）
FROM docker.jiaxin.site/library/rust:1-slim-bookworm AS builder

# 国内镜像加速：apt 换清华源，cargo 换清华 sparse 源
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' \
        /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources 2>/dev/null || true \
    && mkdir -p "$CARGO_HOME" \
    && printf '[source.crates-io]\nreplace-with = "tuna"\n\n[source.tuna]\nregistry = "sparse+https://mirrors.tuna.tsinghua.edu.cn/crates.io-index/"\n' > "$CARGO_HOME/config.toml" \
    && apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        make \
        g++ \
        pkg-config \
        libopenblas-dev \
        libclang-dev \
    && rm -rf /var/lib/apt/lists/*

# llama-cpp-sys-2 构建时用 bindgen 生成 FFI 绑定，必须能找到 libclang（bookworm 自带 LLVM 14）
ENV LIBCLANG_PATH=/usr/lib/llvm-14/lib

# 固定可移植 CPU 基线（x86-64-v2 = SSE2~SSE4.2/POPCNT，不含 AVX/AVX2/FMA/AVX512）。
# 原因：llama-cpp-sys-2 的 build.rs 只按 RUSTFLAGS 的 target-cpu 决定 GGML_NATIVE，
# 但 llama.cpp 的 cmake 在 GGML_NATIVE=OFF 时，GGML_AVX/GGML_AVX2/GGML_FMA/GGML_F16C/
# GGML_BMI2 仍默认全部 ON（会追加 -mavx2 等 flag），必须显式关掉。否则镜像在较新 CPU
# 的机器上构建出的二进制带 AVX2 指令，部署到旧 CPU（如 Xeon E5-2470 Sandy Bridge，
# 无 AVX2/FMA/F16C）时立即 SIGILL（exit 132）。GGML_* 环境变量会被 build.rs 透传给
# cmake 覆盖默认值。x86-64-v2 已被 2009 年后所有主流 x86-64 CPU 支持，bge 小模型
# 嵌入性能损失可忽略。
ENV RUSTFLAGS="-C target-cpu=x86-64-v2"

# 显式关闭 llama.cpp 默认开启的 AVX/AVX2/FMA/F16C/BMI2（保持 x86-64-v2 可移植基线）
ENV GGML_AVX=OFF \
    GGML_AVX2=OFF \
    GGML_BMI2=OFF \
    GGML_FMA=OFF \
    GGML_F16C=OFF \
    GGML_AVX512=OFF \
    GGML_AVX512_VBMI=OFF \
    GGML_AVX512_VNNI=OFF \
    GGML_AVX512_BF16=OFF \
    GGML_AVX_VNNI=OFF

WORKDIR /build

# 先拷贝清单 + 占位 lib：只编译依赖（含 llama.cpp C++）缓存成独立层。
# 注意必须用 --lib：若用 fn main(){} 占位去编可执行文件，会产出"空壳二进制"，
# 一旦被误拷进镜像，容器启动即 exit 0（零日志、无限重启）。
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src \
    && echo 'pub fn _placeholder() {}' > src/lib.rs \
    && cargo build --release --lib \
    && rm -rf src

# banner.txt 由 src/main.rs 的 include_str!("../banner.txt") 编译期内嵌，必须拷入
COPY banner.txt ./

# 拷贝真实源码。关键：必须 touch 源文件——Docker COPY 保留宿主机旧 mtime，
# cargo 的 mtime 指纹会误判"源码未变化"而复用占位产物（空壳进镜像，启动即 exit 0）
COPY src ./src
RUN find src -type f -exec touch {} + \
    && cargo build --release

# 防呆：校验最终二进制含服务日志字符串，杜绝空壳/旧产物被拷进运行镜像
RUN LC_ALL=C grep -a -q "RAG-Service 启动于" target/release/JuanNiang-RAG-Service \
    && echo "OK: 真实二进制确认" \
    || (echo "FATAL: 检测到空壳二进制(缺少 'RAG-Service 启动于')，构建中止" && exit 1)

# ========== 运行阶段 ==========
# 说明：
# - 模型随镜像分发：构建时从上下文 models/ 拷入（需先准备模型，
#   可用 scripts/download_model.py 从魔搭社区下载）；
#   运行时仍可挂载卷覆盖 /app/models
# - 数据目录 /app/data 建议挂载卷持久化
# - 基础镜像使用国内加速域名
FROM docker.jiaxin.site/library/debian:bookworm-slim AS runtime

# 国内镜像加速：apt 换清华源
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' \
        /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources 2>/dev/null || true \
    && apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 \
        libgomp1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/target/release/JuanNiang-RAG-Service /usr/local/bin/rag-service

# 模型随镜像分发：从构建上下文 models/ 拷入（约 25MB）
COPY models/ /app/models/

RUN mkdir -p /app/data
USER root

# 服务配置（可用环境变量覆盖，见 src/config.rs）
ENV RAG_MODEL_PATH=/app/models/bge-small-zh-v1.5-q8_0.gguf \
    RAG_DATA_DIR=/app/data \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=3000 \
    RAG_N_THREADS=4

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD wget -qO- http://127.0.0.1:3000/health >/dev/null 2>&1 || exit 1

VOLUME ["/app/data"]
ENTRYPOINT ["/usr/local/bin/rag-service"]
