//! 应用配置：全部来自环境变量，未设置时用默认值。

#[derive(Debug, Clone)]
pub struct Config {
    /// bge-small-zh-v1.5 Q8_0 GGUF 模型文件路径
    pub model_path: String,
    /// 数据根目录；各分库数据在 {data_dir}/scoops/{scoop}/ 下（index.tvim / tags.bin）
    pub data_dir: String,
    /// HTTP 监听地址
    pub host: String,
    pub port: u16,
    /// llama.cpp 推理线程数（阶段 0 实测：4 最优）
    pub n_threads: i32,
    /// 上下文长度（bge-small-zh 上限 512）
    pub n_ctx: u32,
    /// 嵌入维度（bge-small-zh-v1.5 = 512）
    pub dim: usize,
    /// turbovec 量化位宽（{2,3,4}，取 4）
    pub bit_width: usize,
    /// 服务端分块：单块最大字符数（中文 1 字 ≈ 1 token，260 ≈ 256 token）
    pub max_chunk_chars: usize,
    /// 服务端分块：相邻块重叠字符数（≈ 50 token）
    pub overlap_chars: usize,
    /// tags.bin 是否存原始向量（换取 .tvim 损坏时可自愈重建）
    pub store_raw_vectors: bool,
    /// 查询 embedding LRU 缓存容量
    pub lru_capacity: usize,
}

impl Config {
    /// 从环境变量构建配置；未设置的项回退到默认值
    pub fn from_env() -> Self {
        Self {
            model_path: env_or("RAG_MODEL_PATH", "models/bge-small-zh-v1.5-q8_0.gguf"),
            data_dir: env_or("RAG_DATA_DIR", "data"),
            host: env_or("RAG_HOST", "127.0.0.1"),
            port: num_or("RAG_PORT", 3000),
            n_threads: num_or("RAG_N_THREADS", 4),
            n_ctx: num_or("RAG_N_CTX", 512),
            dim: num_or("RAG_DIM", 512),
            bit_width: num_or("RAG_BIT_WIDTH", 4),
            max_chunk_chars: num_or("RAG_MAX_CHUNK_CHARS", 260),
            overlap_chars: num_or("RAG_OVERLAP_CHARS", 50),
            store_raw_vectors: bool_or("RAG_STORE_RAW_VECTORS", true),
            lru_capacity: num_or("RAG_LRU_CAPACITY", 1024),
        }
    }
}

/// 读环境变量，未设置返回默认值（字符串）
fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_owned())
}

/// 读环境变量并解析为数字
fn num_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// 读环境变量并解析为布尔
fn bool_or(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(default)
}
