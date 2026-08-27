//! Prometheus 指标：暴露给 Grafana 采集（`/metrics` 端点，文本格式）。
//!
//! 命名前缀 `rag_`，标签全部低基数（scoop 枚举 / HTTP 路由模板 / 有限 op 集）。
//! gauge 在 scrape 时实时读取（writer.health() / embedder.info()），不打库。
//!
//! 指标一览：
//! - `rag_http_requests_total{method,path,status}` / `rag_http_request_duration_seconds{method,path}`
//! - `rag_search_total{scoop}` / `rag_search_duration_seconds{scoop}`
//!   / `rag_search_hits_total{scoop}`（命中数分布）/ `rag_search_errors_total{scoop}`
//! - `rag_write_total{op,scoop}`（op: upsert/batch/delete）/ `rag_write_chunks_total{scoop}`
//!   / `rag_write_duration_seconds{op,scoop}` / `rag_write_errors_total{op,scoop}`
//! - `rag_embed_requests_total` / `rag_embed_texts_total` / `rag_embed_duration_seconds` / `rag_embed_errors_total`
//! - `rag_tags{scoop}` / `rag_chunks{scoop}` / `rag_embedder_ready`（scrape 时实时）

use axum::extract::{MatchedPath, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use prometheus::{
    Encoder, Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge,
    IntGaugeVec, Opts, TextEncoder,
};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{Duration, Instant};

use crate::api::AppState;

// ---------- 注册辅助（非宏 API，显式注册到默认注册表） ----------

fn counter_vec(name: &str, help: &str, labels: &[&str]) -> IntCounterVec {
    let cv = IntCounterVec::new(Opts::new(name, help), labels)
        .unwrap_or_else(|e| panic!("构造 {name} 失败: {e}"));
    prometheus::default_registry()
        .register(Box::new(cv.clone()))
        .unwrap_or_else(|e| panic!("注册 {name} 失败: {e}"));
    cv
}

fn histogram_vec(name: &str, help: &str, buckets: Vec<f64>, labels: &[&str]) -> HistogramVec {
    let hv = HistogramVec::new(HistogramOpts::new(name, help).buckets(buckets), labels)
        .unwrap_or_else(|e| panic!("构造 {name} 失败: {e}"));
    prometheus::default_registry()
        .register(Box::new(hv.clone()))
        .unwrap_or_else(|e| panic!("注册 {name} 失败: {e}"));
    hv
}

fn gauge_vec(name: &str, help: &str, labels: &[&str]) -> IntGaugeVec {
    let gv = IntGaugeVec::new(Opts::new(name, help), labels)
        .unwrap_or_else(|e| panic!("构造 {name} 失败: {e}"));
    prometheus::default_registry()
        .register(Box::new(gv.clone()))
        .unwrap_or_else(|e| panic!("注册 {name} 失败: {e}"));
    gv
}

fn counter(name: &str, help: &str) -> IntCounter {
    let c = IntCounter::new(name, help).unwrap_or_else(|e| panic!("构造 {name} 失败: {e}"));
    prometheus::default_registry()
        .register(Box::new(c.clone()))
        .unwrap_or_else(|e| panic!("注册 {name} 失败: {e}"));
    c
}

fn histogram(name: &str, help: &str, buckets: Vec<f64>) -> Histogram {
    let h = Histogram::with_opts(HistogramOpts::new(name, help).buckets(buckets))
        .unwrap_or_else(|e| panic!("构造 {name} 失败: {e}"));
    prometheus::default_registry()
        .register(Box::new(h.clone()))
        .unwrap_or_else(|e| panic!("注册 {name} 失败: {e}"));
    h
}

fn gauge(name: &str, help: &str) -> IntGauge {
    let g = IntGauge::new(name, help).unwrap_or_else(|e| panic!("构造 {name} 失败: {e}"));
    prometheus::default_registry()
        .register(Box::new(g.clone()))
        .unwrap_or_else(|e| panic!("注册 {name} 失败: {e}"));
    g
}

// ---------- 指标访问（OnceLock 单例） ----------

pub fn http_requests() -> &'static IntCounterVec {
    static M: OnceLock<IntCounterVec> = OnceLock::new();
    M.get_or_init(|| {
        counter_vec(
            "rag_http_requests_total",
            "HTTP 请求总数（path 为路由模板，低基数）",
            &["method", "path", "status"],
        )
    })
}

pub fn http_duration() -> &'static HistogramVec {
    static M: OnceLock<HistogramVec> = OnceLock::new();
    M.get_or_init(|| {
        histogram_vec(
            "rag_http_request_duration_seconds",
            "HTTP 请求耗时（path 为路由模板）",
            vec![
                0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ],
            &["method", "path"],
        )
    })
}

pub fn search_total() -> &'static IntCounterVec {
    static M: OnceLock<IntCounterVec> = OnceLock::new();
    M.get_or_init(|| counter_vec("rag_search_total", "检索请求数", &["scoop"]))
}

pub fn search_duration() -> &'static HistogramVec {
    static M: OnceLock<HistogramVec> = OnceLock::new();
    M.get_or_init(|| {
        histogram_vec(
            "rag_search_duration_seconds",
            "检索耗时（含查询嵌入）",
            vec![
                0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ],
            &["scoop"],
        )
    })
}

pub fn search_hits() -> &'static HistogramVec {
    static M: OnceLock<HistogramVec> = OnceLock::new();
    M.get_or_init(|| {
        histogram_vec(
            "rag_search_hits_total",
            "检索返回命中数分布",
            vec![0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            &["scoop"],
        )
    })
}

pub fn search_errors() -> &'static IntCounterVec {
    static M: OnceLock<IntCounterVec> = OnceLock::new();
    M.get_or_init(|| {
        counter_vec(
            "rag_search_errors_total",
            "检索错误数（嵌入/内部失败）",
            &["scoop"],
        )
    })
}

pub fn write_total() -> &'static IntCounterVec {
    static M: OnceLock<IntCounterVec> = OnceLock::new();
    M.get_or_init(|| {
        counter_vec(
            "rag_write_total",
            "写入操作数（op: upsert/batch/delete）",
            &["op", "scoop"],
        )
    })
}

pub fn write_chunks() -> &'static IntCounterVec {
    static M: OnceLock<IntCounterVec> = OnceLock::new();
    M.get_or_init(|| counter_vec("rag_write_chunks_total", "写入块向量总数", &["scoop"]))
}

pub fn write_duration() -> &'static HistogramVec {
    static M: OnceLock<HistogramVec> = OnceLock::new();
    M.get_or_init(|| {
        histogram_vec(
            "rag_write_duration_seconds",
            "写入耗时（嵌入+应用+持久化+发布）",
            vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            &["op", "scoop"],
        )
    })
}

pub fn write_errors() -> &'static IntCounterVec {
    static M: OnceLock<IntCounterVec> = OnceLock::new();
    M.get_or_init(|| {
        counter_vec(
            "rag_write_errors_total",
            "写入失败数（op: upsert/batch/delete）",
            &["op", "scoop"],
        )
    })
}

pub fn embed_requests() -> &'static IntCounter {
    static M: OnceLock<IntCounter> = OnceLock::new();
    M.get_or_init(|| counter("rag_embed_requests_total", "嵌入批量请求数"))
}

pub fn embed_texts() -> &'static IntCounter {
    static M: OnceLock<IntCounter> = OnceLock::new();
    M.get_or_init(|| counter("rag_embed_texts_total", "嵌入文本条数"))
}

pub fn embed_duration() -> &'static Histogram {
    static M: OnceLock<Histogram> = OnceLock::new();
    M.get_or_init(|| {
        histogram(
            "rag_embed_duration_seconds",
            "嵌入耗时（含排队）",
            vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )
    })
}

pub fn embed_errors() -> &'static IntCounter {
    static M: OnceLock<IntCounter> = OnceLock::new();
    M.get_or_init(|| counter("rag_embed_errors_total", "嵌入失败数"))
}

fn tags_gauge() -> &'static IntGaugeVec {
    static M: OnceLock<IntGaugeVec> = OnceLock::new();
    M.get_or_init(|| gauge_vec("rag_tags", "各分库 tag 数（scrape 时实时）", &["scoop"]))
}

fn chunks_gauge() -> &'static IntGaugeVec {
    static M: OnceLock<IntGaugeVec> = OnceLock::new();
    M.get_or_init(|| gauge_vec("rag_chunks", "各分库块向量数（scrape 时实时）", &["scoop"]))
}

fn embedder_ready() -> &'static IntGauge {
    static M: OnceLock<IntGauge> = OnceLock::new();
    M.get_or_init(|| gauge("rag_embedder_ready", "嵌入模型是否就绪（0/1）"))
}

/// 上次成功读取的就绪状态：scrape 遇到嵌入线程繁忙超时时回退此值，
/// 保证 /metrics 永远不会被卡在推理后面。
static LAST_READY: AtomicI64 = AtomicI64::new(0);

/// scrape 时读取就绪状态；超时（嵌入线程正忙于长批量推理）则返回上次值。
async fn ready_with_fallback(state: &Arc<AppState>) -> i64 {
    match tokio::time::timeout(Duration::from_millis(500), state.embedder.info()).await {
        Ok(info) => {
            let v = info.ready as i64;
            LAST_READY.store(v, Ordering::Relaxed);
            v
        }
        Err(_) => LAST_READY.load(Ordering::Relaxed),
    }
}

// ---------- HTTP 中间件（请求数/耗时；path 用路由模板保持低基数） ----------

pub async fn http_metrics(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    // MatchedPath 是路由匹配后的模板路径（如 /scoops/{scoop}/tags/{tag}），
    // 避免具体 scoop/tag 值造成高基数标签
    let path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_string())
        .unwrap_or_else(|| "<unmatched>".to_string());
    let start = Instant::now();
    let resp = next.run(req).await;
    let status = resp.status().as_u16().to_string();
    http_requests()
        .with_label_values(&[method.as_str(), &path, &status])
        .inc();
    http_duration()
        .with_label_values(&[method.as_str(), &path])
        .observe(start.elapsed().as_secs_f64());
    resp
}

// ---------- /metrics 处理器 ----------

/// GET /metrics：文本格式输出全部指标；gauge 实时刷新（不打库）。
pub async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    // 存储规模 gauge：scrape 时实时读取
    for (scoop, tags, chunks) in state.writer.health() {
        tags_gauge()
            .with_label_values(&[scoop.as_str()])
            .set(tags as i64);
        chunks_gauge()
            .with_label_values(&[scoop.as_str()])
            .set(chunks as i64);
    }
    // 模型就绪状态 gauge：带超时，嵌入线程繁忙时不阻塞 scrape
    embedder_ready().set(ready_with_fallback(&state).await);

    let encoder = TextEncoder::new();
    let mut buf = Vec::new();
    if let Err(e) = encoder.encode(&prometheus::gather(), &mut buf) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("指标编码失败: {e}"),
        )
            .into_response();
    }
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        buf,
    )
        .into_response()
}
