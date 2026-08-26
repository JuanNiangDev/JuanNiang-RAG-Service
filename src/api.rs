//! HTTP API（阶段 6）。
//!
//! - PUT    /scoops/{scoop}/tags/{tag}       upsert（已存在则覆写）
//! - POST   /scoops/{scoop}/tags/batch       批量 upsert（一次嵌入 + 一次发布）
//! - GET    /scoops/{scoop}/tags/search?q=&k=&min_score=   检索（限制在 scoop 内）
//! - DELETE /scoops/{scoop}/tags/{tag}       删除
//! - GET    /health / /info                   健康检查 + per-scoop 规模
//!
//! scoop 是白名单枚举（knowledge / memory / groupmgr），非法值 → 400。

use crate::embedding::Embedder;
use crate::error::ServiceError;
use crate::search::{SearchHit, aggregate};
use crate::store::{Scoop, StoreError};
use crate::writer::Writer;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use tracing::error;
use uuid::Uuid;

// ---------- 应用状态 ----------

pub struct AppState {
    pub writer: Arc<Writer>,
    pub embedder: Embedder,
    /// 查询文本 → 归一化向量（命中免推理，§8.3）；scoop 无关（同文同向量）
    pub query_cache: Mutex<LruCache<String, Vec<f32>>>,
}

impl AppState {
    pub fn new(writer: Arc<Writer>, embedder: Embedder, lru_capacity: usize) -> Self {
        Self {
            writer,
            embedder,
            query_cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(lru_capacity).expect("LRU 容量必须 > 0"),
            )),
        }
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/info", get(info))
        .route("/scoops/{scoop}/tags/{tag}", put(upsert).delete(delete_tag))
        .route("/scoops/{scoop}/tags/batch", post(batch_upsert))
        .route("/scoops/{scoop}/tags/search", get(search))
        .with_state(state)
}

// ---------- 请求/响应类型 ----------

#[derive(Deserialize)]
pub struct UpsertRequest {
    pub text: String,
}

#[derive(Serialize)]
pub struct UpsertResponse {
    pub scoop: String,
    pub tag: Uuid,
    pub chunk_count: usize,
    pub truncated: bool,
}

#[derive(Deserialize)]
pub struct BatchUpsertRequest {
    pub items: Vec<BatchItem>,
}

#[derive(Deserialize)]
pub struct BatchItem {
    pub tag: Uuid,
    pub text: String,
}

#[derive(Serialize)]
pub struct BatchItemResponse {
    pub tag: Uuid,
    pub chunk_count: usize,
    pub truncated: bool,
    pub error: Option<String>,
}

#[derive(Serialize)]
pub struct BatchResponse {
    pub results: Vec<BatchItemResponse>,
}

#[derive(Deserialize)]
pub struct SearchParams {
    pub q: String,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default)]
    pub min_score: Option<f32>,
}

fn default_k() -> usize {
    10
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchHit>,
}

#[derive(Serialize, Default)]
pub struct ScoopStats {
    pub tags: usize,
    pub chunks: usize,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    /// 各 scoop 的 tag / 块数量
    pub scoops: HashMap<String, ScoopStats>,
}

#[derive(Serialize)]
pub struct MemoryInfo {
    /// 常驻物理内存（kB）
    pub rss_kb: u64,
    /// 虚拟内存（kB）
    pub vsize_kb: u64,
}

#[derive(Serialize)]
pub struct InfoResponse {
    pub status: &'static str,
    /// embedding 模型状态（由嵌入线程上报）
    pub model: crate::embedding::EmbedderInfo,
    /// 进程内存（仅 Linux，读 /proc/self/status）
    pub memory: Option<MemoryInfo>,
    /// 各 scoop 的 tag / 块数量
    pub scoops: HashMap<String, ScoopStats>,
}

fn scoop_stats_map(stats: Vec<(Scoop, usize, usize)>) -> HashMap<String, ScoopStats> {
    stats
        .into_iter()
        .map(|(scoop, tags, chunks)| (scoop.as_str().to_string(), ScoopStats { tags, chunks }))
        .collect()
}

// ---------- 处理器 ----------

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        scoops: scoop_stats_map(state.writer.health()),
    })
}

/// GET /info：模型状态 + 进程内存 + per-scoop 规模
async fn info(State(state): State<Arc<AppState>>) -> Json<InfoResponse> {
    let model = state.embedder.info().await;
    let memory = process_memory_kb().map(|(rss_kb, vsize_kb)| MemoryInfo { rss_kb, vsize_kb });
    Json(InfoResponse {
        status: "ok",
        model,
        memory,
        scoops: scoop_stats_map(state.writer.health()),
    })
}

/// 读 /proc/self/status 获取进程内存（非 Linux 返回 None）
fn process_memory_kb() -> Option<(u64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let mut rss = None;
    let mut vsize = None;
    for line in status.lines() {
        if let Some(v) = line.strip_prefix("VmRSS:") {
            rss = parse_kb(v);
        } else if let Some(v) = line.strip_prefix("VmSize:") {
            vsize = parse_kb(v);
        }
    }
    match (rss, vsize) {
        (Some(r), Some(v)) => Some((r, v)),
        _ => None,
    }
}

/// 解析 "12345 kB" 形式的数值
fn parse_kb(s: &str) -> Option<u64> {
    s.trim().trim_end_matches(" kB").parse().ok()
}

async fn upsert(
    State(state): State<Arc<AppState>>,
    Path((scoop, tag)): Path<(Scoop, Uuid)>,
    Json(body): Json<UpsertRequest>,
) -> Result<Json<UpsertResponse>, AppError> {
    if body.text.trim().is_empty() {
        return Err(AppError::bad_request("text 不能为空"));
    }
    let stats = state.writer.upsert(scoop, tag, body.text).await?;
    Ok(Json(UpsertResponse {
        scoop: scoop.as_str().to_string(),
        tag,
        chunk_count: stats.chunk_count,
        truncated: stats.truncated,
    }))
}

async fn batch_upsert(
    State(state): State<Arc<AppState>>,
    Path(scoop): Path<Scoop>,
    Json(body): Json<BatchUpsertRequest>,
) -> Result<Json<BatchResponse>, AppError> {
    if body.items.is_empty() {
        return Err(AppError::bad_request("items 不能为空"));
    }
    let tags: Vec<Uuid> = body.items.iter().map(|i| i.tag).collect();
    let items: Vec<(Uuid, String)> = body.items.into_iter().map(|i| (i.tag, i.text)).collect();
    let results = state.writer.batch_upsert(scoop, items).await?;
    let results = tags
        .into_iter()
        .zip(results)
        .map(|(tag, r)| match r {
            Ok(s) => BatchItemResponse {
                tag,
                chunk_count: s.chunk_count,
                truncated: s.truncated,
                error: None,
            },
            Err(e) => BatchItemResponse {
                tag,
                chunk_count: 0,
                truncated: false,
                error: Some(e.to_string()),
            },
        })
        .collect();
    Ok(Json(BatchResponse { results }))
}

/// 检索：q 必填，k 默认 10，min_score 可选；只在目标 scoop 内检索
async fn search(
    State(state): State<Arc<AppState>>,
    Path(scoop): Path<Scoop>,
    Query(params): Query<SearchParams>,
) -> Result<Json<SearchResponse>, AppError> {
    if params.q.trim().is_empty() {
        return Err(AppError::bad_request("q 不能为空"));
    }
    let k = params.k.clamp(1, 100);

    // 1. 查询向量（LRU 缓存命中免推理）
    // 注意：MutexGuard 不能跨 await，命中分支的守卫随语句结束释放，
    //       未命中时先 await 嵌入、再重新加锁写入缓存。
    let query_vec = {
        let cached = state
            .query_cache
            .lock()
            .expect("缓存锁中毒")
            .get(&params.q)
            .cloned();
        if let Some(v) = cached {
            v
        } else {
            let (vectors, _) = state.embedder.embed(vec![params.q.clone()], true).await?;
            let v = vectors
                .into_iter()
                .next()
                .ok_or_else(|| AppError(ServiceError::Internal("嵌入结果为空".into())))?;
            state
                .query_cache
                .lock()
                .expect("缓存锁中毒")
                .put(params.q.clone(), v.clone());
            v
        }
    };

    // 2. scoop 内块级检索 → 按 tag 聚合
    let snap = state.writer.snapshot(scoop);
    let k_chunks = (k * 3).max(30); // 多召回一些块再聚合，避免同一 tag 霸榜
    let hits = snap.index.search(&query_vec, k_chunks);
    let results = aggregate(&hits, &snap.chunk_owner, k, params.min_score);
    Ok(Json(SearchResponse { results }))
}

async fn delete_tag(
    State(state): State<Arc<AppState>>,
    Path((scoop, tag)): Path<(Scoop, Uuid)>,
) -> Result<Json<serde_json::Value>, AppError> {
    state.writer.delete(scoop, tag).await?;
    Ok(Json(serde_json::json!({ "deleted": tag })))
}

// ---------- 统一错误 → HTTP 响应 ----------

pub struct AppError(pub ServiceError);

impl AppError {
    fn bad_request(msg: impl Into<String>) -> Self {
        AppError(ServiceError::BadRequest(msg.into()))
    }
}

impl From<ServiceError> for AppError {
    fn from(e: ServiceError) -> Self {
        AppError(e)
    }
}

impl From<crate::embedding::EmbedError> for AppError {
    fn from(e: crate::embedding::EmbedError) -> Self {
        AppError(ServiceError::Embed(e))
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self.0 {
            ServiceError::BadRequest(m) => (StatusCode::BAD_REQUEST, m.clone()),
            ServiceError::Store(StoreError::TagNotFound(_)) => {
                (StatusCode::NOT_FOUND, self.0.to_string())
            }
            ServiceError::Store(StoreError::UnknownScoop(..)) => {
                (StatusCode::BAD_REQUEST, self.0.to_string())
            }
            ServiceError::Store(StoreError::TagInOtherScoop(..)) => {
                (StatusCode::CONFLICT, self.0.to_string())
            }
            _ => (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()),
        };
        error!(error = %self.0, "请求失败");
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}
