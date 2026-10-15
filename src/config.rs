//! 应用配置：全部来自环境变量，未设置时用默认值。

/// 嵌入后端：运行时二选一（RAG_BACKEND=gguf|onnx）
///
/// - `Gguf`：llama.cpp 进程内推理，bge-small-zh-v1.5（512 维，查询加指令前缀）
/// - `Onnx`：ort CPU 推理，granite-embedding-97m-multilingual-r2（384 维，CLS 池化，无查询前缀）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// llama.cpp + bge GGUF（向后兼容的默认后端）
    #[default]
    Gguf,
    /// ort CPU + granite ONNX
    Onnx,
}

impl std::str::FromStr for Backend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "gguf" => Ok(Backend::Gguf),
            "onnx" => Ok(Backend::Onnx),
            other => Err(format!("RAG_BACKEND={other} 无效（可选 gguf|onnx）")),
        }
    }
}

impl Backend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Gguf => "gguf",
            Backend::Onnx => "onnx",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    /// 嵌入后端（gguf | onnx）
    pub backend: Backend,
    /// bge-small-zh-v1.5 Q8_0 GGUF 模型文件路径（backend=gguf 时使用）
    pub model_path: String,
    /// granite ONNX 模型文件路径（backend=onnx 时使用；int8 量化，~98MB）
    pub onnx_model_path: String,
    /// granite tokenizer.json 路径（backend=onnx 时使用）
    pub onnx_tokenizer_path: String,
    /// 数据根目录；各分库数据在 {data_dir}/scoops/{scoop}/ 下（index.tvim / tags.bin）
    pub data_dir: String,
    /// HTTP 监听地址
    pub host: String,
    pub port: u16,
    /// 推理线程数（阶段 0 实测：4 最优；两个后端共用）
    pub n_threads: i32,
    /// 上下文长度（gguf: bge 上限 512；onnx: 单条文本 token 截断上限，
    /// granite 本身支持 32K 但服务端分块已限制输入长度）
    pub n_ctx: u32,
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

impl Default for Config {
    /// 默认值：gguf 后端（向后兼容），向量索引维度不再配置——
    /// 由加载的模型实时派生（bge=512 / granite=384），见 main.rs 启动流程。
    fn default() -> Self {
        Self {
            backend: Backend::Gguf,
            model_path: "models/bge-small-zh-v1.5-q8_0.gguf".to_owned(),
            onnx_model_path: "models/granite/model.onnx".to_owned(),
            onnx_tokenizer_path: "models/granite/tokenizer.json".to_owned(),
            data_dir: "data".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 3000,
            n_threads: 4,
            n_ctx: 512,
            bit_width: 4,
            max_chunk_chars: 260,
            overlap_chars: 50,
            store_raw_vectors: true,
            lru_capacity: 1024,
        }
    }
}

impl Config {
    /// 从环境变量构建配置；未设置的项回退到默认值
    pub fn from_env() -> Self {
        let d = Self::default();
        Self {
            backend: backend_or(d.backend),
            model_path: env_or("RAG_MODEL_PATH", &d.model_path),
            onnx_model_path: env_or("RAG_ONNX_MODEL_PATH", &d.onnx_model_path),
            onnx_tokenizer_path: env_or("RAG_ONNX_TOKENIZER_PATH", &d.onnx_tokenizer_path),
            data_dir: env_or("RAG_DATA_DIR", &d.data_dir),
            host: env_or("RAG_HOST", &d.host),
            port: num_or("RAG_PORT", d.port),
            n_threads: num_or("RAG_N_THREADS", d.n_threads),
            n_ctx: num_or("RAG_N_CTX", d.n_ctx),
            bit_width: num_or("RAG_BIT_WIDTH", d.bit_width),
            max_chunk_chars: num_or("RAG_MAX_CHUNK_CHARS", d.max_chunk_chars),
            overlap_chars: num_or("RAG_OVERLAP_CHARS", d.overlap_chars),
            store_raw_vectors: bool_or("RAG_STORE_RAW_VECTORS", d.store_raw_vectors),
            lru_capacity: num_or("RAG_LRU_CAPACITY", d.lru_capacity),
        }
    }
}

/// 读 RAG_BACKEND；非法值告警并回退默认（不阻断启动）
fn backend_or(default: Backend) -> Backend {
    match std::env::var("RAG_BACKEND") {
        Ok(v) => match v.parse::<Backend>() {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("{e}，回退 {}", default.as_str());
                default
            }
        },
        Err(_) => default,
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
