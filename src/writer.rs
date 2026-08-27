//! 写者任务（阶段 5）：查询与写入互不阻塞的核心。
//!
//! - 写者线程独占 `StoreSet`（多 scoop，每个 scoop 独立 TagStore），通过 mpsc 收命令
//! - 每批命令：分块 → 批量嵌入 → 按 scoop 应用 → 本 scoop 持久化 → 本 scoop COW 发布
//!   （一次写入只影响归属 scoop，其余 scoop 的快照与磁盘文件零参与）
//! - 读者永远只读对应 scoop 的 `Arc<IndexSnapshot>`（瞬时读锁取引用，检索无锁）
//! - 批量命令只发布/持久化一次（摊销 COW 成本）
//!
//! 持久化先于发布：读者永远看不到未落盘的数据，崩溃最多丢最后一次应答前
//! 的写入（由调用方感知）。

use crate::chunker::{ChunkConfig, split};
use crate::config::Config;
use crate::embedding::Embedder;
use crate::error::ServiceError;
use crate::store::{IndexSnapshot, Scoop, StoreSet};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info};
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum WriterError {
    #[error("写者线程未响应（可能已退出）")]
    Closed,
}

/// 单次写入的统计
#[derive(Debug, Clone)]
pub struct WriteStats {
    pub chunk_count: usize,
    pub truncated: bool,
}

/// 写命令：通过 mpsc 发给写者线程（带 scoop 路由）
enum WriteCmd {
    Upsert {
        scoop: Scoop,
        tag: Uuid,
        text: String,
        reply: oneshot::Sender<Result<WriteStats, ServiceError>>,
    },
    BatchUpsert {
        scoop: Scoop,
        items: Vec<(Uuid, String)>,
        reply: oneshot::Sender<Vec<Result<WriteStats, ServiceError>>>,
    },
    Delete {
        scoop: Scoop,
        tag: Uuid,
        reply: oneshot::Sender<Result<(), ServiceError>>,
    },
    BatchDelete {
        scoop: Scoop,
        tags: Vec<Uuid>,
        reply: oneshot::Sender<Vec<Result<(), ServiceError>>>,
    },
}

/// 写者句柄：Clone 可分享，内部是 `Sender<WriteCmd>` + 各 scoop 最新快照
#[derive(Clone)]
pub struct Writer {
    tx: mpsc::Sender<WriteCmd>,
    snapshots: HashMap<Scoop, Arc<RwLock<Arc<IndexSnapshot>>>>,
}

impl Writer {
    /// 启动写者任务
    pub fn start(stores: StoreSet, embedder: Embedder, config: Config) -> Self {
        let mut snapshots = HashMap::new();
        for scoop in Scoop::ALL {
            let initial = stores.snapshot(scoop).expect("初始快照失败");
            snapshots.insert(scoop, Arc::new(RwLock::new(Arc::new(initial))));
        }
        let (tx, rx) = mpsc::channel(256);

        let task = WriterTask {
            stores,
            embedder,
            config,
            rx,
            snapshots: snapshots.clone(),
        };
        tokio::spawn(async move { task.run().await });

        Self { tx, snapshots }
    }

    /// 读者取指定 scoop 的最新快照：瞬时读锁拿 Arc，之后检索无锁
    pub fn snapshot(&self, scoop: Scoop) -> Arc<IndexSnapshot> {
        self.snapshots[&scoop].read().expect("快照锁中毒").clone()
    }

    /// 单条 upsert（读己之写：应答时数据已发布并落盘）
    pub async fn upsert(
        &self,
        scoop: Scoop,
        tag: Uuid,
        text: String,
    ) -> Result<WriteStats, ServiceError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(WriteCmd::Upsert {
                scoop,
                tag,
                text,
                reply,
            })
            .await
            .map_err(|_| ServiceError::Writer(WriterError::Closed))?;
        rx.await
            .map_err(|_| ServiceError::Writer(WriterError::Closed))?
    }

    /// 批量 upsert：一次嵌入 + 一次发布 + 一次持久化（同一 scoop 内）
    pub async fn batch_upsert(
        &self,
        scoop: Scoop,
        items: Vec<(Uuid, String)>,
    ) -> Result<Vec<Result<WriteStats, ServiceError>>, ServiceError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(WriteCmd::BatchUpsert {
                scoop,
                items,
                reply,
            })
            .await
            .map_err(|_| ServiceError::Writer(WriterError::Closed))?;
        let results = rx
            .await
            .map_err(|_| ServiceError::Writer(WriterError::Closed))?;
        Ok(results)
    }

    pub async fn delete(&self, scoop: Scoop, tag: Uuid) -> Result<(), ServiceError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(WriteCmd::Delete { scoop, tag, reply })
            .await
            .map_err(|_| ServiceError::Writer(WriterError::Closed))?;
        rx.await
            .map_err(|_| ServiceError::Writer(WriterError::Closed))?
    }

    /// 批量删除：逐条尝试（不属于本 scoop 的 tag 单条失败，不影响其他），
    /// 有成功条目时一次持久化+发布。返回逐条结果。
    pub async fn batch_delete(
        &self,
        scoop: Scoop,
        tags: Vec<Uuid>,
    ) -> Result<Vec<Result<(), ServiceError>>, ServiceError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(WriteCmd::BatchDelete { scoop, tags, reply })
            .await
            .map_err(|_| ServiceError::Writer(WriterError::Closed))?;
        // rx 内容本身即 Result<Vec<...>, ServiceError>，RecvError 需转换
        rx.await
            .map_err(|_| ServiceError::Writer(WriterError::Closed))
    }

    /// 各 scoop 规模：(scoop, tag 数, 块数)
    pub fn health(&self) -> Vec<(Scoop, usize, usize)> {
        Scoop::ALL
            .iter()
            .map(|&s| {
                let snap = self.snapshot(s);
                (s, snap.tag_count(), snap.chunk_count())
            })
            .collect()
    }
}

/// 写者任务本体（独占 StoreSet）
struct WriterTask {
    stores: StoreSet,
    embedder: Embedder,
    config: Config,
    rx: mpsc::Receiver<WriteCmd>,
    snapshots: HashMap<Scoop, Arc<RwLock<Arc<IndexSnapshot>>>>,
}

impl WriterTask {
    async fn run(mut self) {
        while let Some(cmd) = self.rx.recv().await {
            // 顺手把队列里已积压的命令一起取出（批量合并）
            let mut cmds = vec![cmd];
            while let Ok(next) = self.rx.try_recv() {
                cmds.push(next);
            }
            for c in cmds {
                self.handle(c).await;
            }
        }
        info!("写者任务退出");
    }

    async fn handle(&mut self, cmd: WriteCmd) {
        match cmd {
            WriteCmd::Upsert {
                scoop,
                tag,
                text,
                reply,
            } => {
                let result = self.apply_upsert(scoop, tag, text).await;
                let _ = reply.send(result);
            }
            WriteCmd::BatchUpsert {
                scoop,
                items,
                reply,
            } => {
                // 整批只发布/持久化一次
                let results = self.apply_batch_upsert(scoop, items).await;
                let _ = reply.send(results);
            }
            WriteCmd::Delete { scoop, tag, reply } => {
                let result = self.apply_delete(scoop, tag);
                let _ = reply.send(result);
            }
            WriteCmd::BatchDelete { scoop, tags, reply } => {
                let results = self.apply_batch_delete(scoop, tags);
                let _ = reply.send(results);
            }
        }
    }

    /// 单条 upsert：应用 → 本 scoop 持久化 → 本 scoop 发布
    async fn apply_upsert(
        &mut self,
        scoop: Scoop,
        tag: Uuid,
        text: String,
    ) -> Result<WriteStats, ServiceError> {
        let start = std::time::Instant::now();
        let result: Result<WriteStats, ServiceError> = async {
            let (chunks, vectors, truncated) =
                self.prepare_chunks(std::slice::from_ref(&text)).await?;
            let chunks = chunks.into_iter().next().unwrap_or_default();
            let vectors = vectors.into_iter().next().unwrap_or_default();
            let truncated = truncated.into_iter().next().unwrap_or(false);
            if vectors.is_empty() {
                return Err(ServiceError::BadRequest("文本为空或分块后无内容".into()));
            }
            let n = self.stores.upsert(scoop, tag, &chunks, &vectors)?;
            self.commit(scoop)?;
            debug!(scoop = %scoop, tag = %tag, chunks = n, "upsert 完成");
            Ok(WriteStats {
                chunk_count: n,
                truncated,
            })
        }
        .await;
        // 指标
        crate::metrics::write_total()
            .with_label_values(&["upsert", scoop.as_str()])
            .inc();
        crate::metrics::write_duration()
            .with_label_values(&["upsert", scoop.as_str()])
            .observe(start.elapsed().as_secs_f64());
        match &result {
            Ok(ws) => {
                crate::metrics::write_chunks()
                    .with_label_values(&[scoop.as_str()])
                    .inc_by(ws.chunk_count as u64);
            }
            Err(_) => {
                crate::metrics::write_errors()
                    .with_label_values(&["upsert", scoop.as_str()])
                    .inc();
            }
        }
        result
    }

    /// 批量 upsert：所有条目合并成一次嵌入，整批一次持久化+发布
    async fn apply_batch_upsert(
        &mut self,
        scoop: Scoop,
        items: Vec<(Uuid, String)>,
    ) -> Vec<Result<WriteStats, ServiceError>> {
        let start = std::time::Instant::now();
        let mut results = Vec::with_capacity(items.len());
        if items.is_empty() {
            return results;
        }

        // 1. 全部文本合并嵌入（一次批量推理）
        let texts: Vec<String> = items.iter().map(|(_, t)| t.clone()).collect();
        let prepared = self.prepare_chunks(&texts).await;
        let mut any_error = false;

        match prepared {
            Ok((all_chunks, all_vectors, all_truncated)) => {
                // 2. 逐条应用
                for (idx, (tag, _)) in items.iter().enumerate() {
                    let chunks = all_chunks[idx].clone();
                    let vectors = all_vectors[idx].clone();
                    let truncated = all_truncated[idx];
                    if vectors.is_empty() {
                        results.push(Err(ServiceError::BadRequest(
                            "文本为空或分块后无内容".into(),
                        )));
                        any_error = true;
                        continue;
                    }
                    match self.stores.upsert(scoop, *tag, &chunks, &vectors) {
                        Ok(n) => results.push(Ok(WriteStats {
                            chunk_count: n,
                            truncated,
                        })),
                        Err(e) => {
                            results.push(Err(ServiceError::Store(e)));
                            any_error = true;
                        }
                    }
                }
            }
            Err(e) => {
                for _ in &items {
                    results.push(Err(ServiceError::Internal(e.to_string())));
                }
                any_error = true;
            }
        }

        // 3. 即使有条目失败，已成功的部分也要持久化+发布；
        //    持久化/发布失败必须让调用方感知：成功条目标记为失败（未落盘）
        if (!any_error || results.iter().any(|r| r.is_ok()))
            && let Err(e) = self.commit(scoop)
        {
            let msg = format!("批量写入持久化失败: {e}");
            for r in results.iter_mut() {
                if r.is_ok() {
                    *r = Err(ServiceError::Internal(msg.clone()));
                }
            }
        }
        info!(scoop = %scoop, n = items.len(), "批量 upsert 完成");
        // 指标（op=batch：一次批量 = 一次写操作，逐条结果计入块数/错误）
        crate::metrics::write_total()
            .with_label_values(&["batch", scoop.as_str()])
            .inc();
        crate::metrics::write_duration()
            .with_label_values(&["batch", scoop.as_str()])
            .observe(start.elapsed().as_secs_f64());
        for r in &results {
            match r {
                Ok(ws) => {
                    crate::metrics::write_chunks()
                        .with_label_values(&[scoop.as_str()])
                        .inc_by(ws.chunk_count as u64);
                }
                Err(_) => {
                    crate::metrics::write_errors()
                        .with_label_values(&["batch", scoop.as_str()])
                        .inc();
                }
            }
        }
        results
    }

    fn apply_delete(&mut self, scoop: Scoop, tag: Uuid) -> Result<(), ServiceError> {
        let start = std::time::Instant::now();
        let result = (|| -> Result<(), ServiceError> {
            self.stores.delete(scoop, tag)?;
            self.commit(scoop)?;
            info!(scoop = %scoop, tag = %tag, "删除完成");
            Ok(())
        })();
        // 指标
        crate::metrics::write_total()
            .with_label_values(&["delete", scoop.as_str()])
            .inc();
        crate::metrics::write_duration()
            .with_label_values(&["delete", scoop.as_str()])
            .observe(start.elapsed().as_secs_f64());
        if result.is_err() {
            crate::metrics::write_errors()
                .with_label_values(&["delete", scoop.as_str()])
                .inc();
        }
        result
    }

    /// 批量删除：逐条尝试，有成功条目时一次持久化+发布。
    fn apply_batch_delete(
        &mut self,
        scoop: Scoop,
        tags: Vec<Uuid>,
    ) -> Vec<Result<(), ServiceError>> {
        let start = std::time::Instant::now();
        let mut results = Vec::with_capacity(tags.len());
        let mut any_ok = false;
        for tag in tags {
            match self.stores.delete(scoop, tag) {
                Ok(()) => {
                    any_ok = true;
                    results.push(Ok(()));
                }
                Err(e) => results.push(Err(ServiceError::Store(e))),
            }
        }
        // 有成功条目才持久化+发布（全失败时无变更，跳过 commit）；
        // 持久化失败时把成功条目标记为失败，避免调用方误以为已落盘
        if any_ok && let Err(e) = self.commit(scoop) {
            let msg = format!("批量删除持久化失败: {e}");
            for r in results.iter_mut() {
                if r.is_ok() {
                    *r = Err(ServiceError::Internal(msg.clone()));
                }
            }
        }
        info!(scoop = %scoop, n = results.len(), "批量删除完成");
        // 指标：逐条按 op=delete 计数（面板维度不变），耗时记整批一次
        crate::metrics::write_duration()
            .with_label_values(&["delete", scoop.as_str()])
            .observe(start.elapsed().as_secs_f64());
        for r in &results {
            crate::metrics::write_total()
                .with_label_values(&["delete", scoop.as_str()])
                .inc();
            if r.is_err() {
                crate::metrics::write_errors()
                    .with_label_values(&["delete", scoop.as_str()])
                    .inc();
            }
        }
        results
    }

    /// 分块 + 批量嵌入。返回 (每文本的块列表, 每文本的向量列表, 每文本是否有截断)
    async fn prepare_chunks(
        &mut self,
        texts: &[String],
    ) -> Result<(Vec<Vec<String>>, Vec<Vec<Vec<f32>>>, Vec<bool>), ServiceError> {
        let cfg = ChunkConfig::new(self.config.max_chunk_chars, self.config.overlap_chars);
        let all_chunks: Vec<Vec<String>> = texts.iter().map(|t| split(t, &cfg)).collect();
        let flat: Vec<String> = all_chunks.iter().flatten().cloned().collect();
        if flat.is_empty() {
            return Err(ServiceError::BadRequest("文本为空".into()));
        }

        let (vectors, truncated_flags) = self.embedder.embed(flat, false).await?;

        // 按文本还原向量分组
        let mut all_vectors = Vec::with_capacity(texts.len());
        let mut all_truncated = Vec::with_capacity(texts.len());
        let mut cursor = 0;
        for chunks in &all_chunks {
            let n = chunks.len();
            all_vectors.push(vectors[cursor..cursor + n].to_vec());
            let any_truncated = truncated_flags[cursor..cursor + n].iter().any(|&b| b);
            all_truncated.push(any_truncated);
            cursor += n;
        }
        Ok((all_chunks, all_vectors, all_truncated))
    }

    /// 本 scoop 持久化 → COW 发布 → 替换对应读者快照
    fn commit(&mut self, scoop: Scoop) -> Result<(), ServiceError> {
        self.stores.persist(scoop)?;
        let snap = self.stores.snapshot(scoop)?;
        snap.index.prepare();
        *self.snapshots[&scoop].write().expect("快照锁中毒") = Arc::new(snap);
        Ok(())
    }
}
