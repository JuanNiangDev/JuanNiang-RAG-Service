//! 端到端测试（阶段 8）：需要模型文件。
//!
//! 运行：RAG_MODEL_PATH=models/bge-small-zh-v1.5-q8_0.gguf \
//!       cargo test --test e2e -- --ignored

use juan_niang_rag_service::config::Config;
use juan_niang_rag_service::embedding::Embedder;
use juan_niang_rag_service::search::aggregate;
use juan_niang_rag_service::store::{Scoop, StoreSet};
use juan_niang_rag_service::writer::Writer;
use uuid::Uuid;

fn test_config(dir: &std::path::Path) -> Config {
    let model_path = std::env::var("RAG_MODEL_PATH").expect("需要 RAG_MODEL_PATH 环境变量");
    Config {
        model_path,
        data_dir: dir.to_string_lossy().into_owned(),
        host: "127.0.0.1".into(),
        port: 0,
        n_threads: 4,
        n_ctx: 512,
        dim: 512,
        bit_width: 4,
        max_chunk_chars: 260,
        overlap_chars: 50,
        store_raw_vectors: true,
        lru_capacity: 128,
    }
}

#[tokio::test]
#[ignore]
async fn full_cycle() {
    let dir = std::env::temp_dir().join(format!("rag_e2e_{}", Uuid::new_v4()));
    let config = test_config(&dir);
    std::fs::create_dir_all(&config.data_dir).unwrap();

    let embedder = Embedder::new(&config.model_path, config.n_threads, config.n_ctx);
    let stores = StoreSet::load(&config).unwrap();
    let writer = Writer::start(stores, embedder.clone(), config.clone());

    // 1. 长文 upsert（应触发服务端分块）；固定使用 knowledge 分库
    let scoop = Scoop::Knowledge;
    let tag = Uuid::new_v4();
    let long_text: String = (0..30)
        .map(|i| format!("这是第{i}段测试内容，用于验证服务端透明分块与相似检索。"))
        .collect();
    let stats = writer.upsert(scoop, tag, long_text).await.unwrap();
    assert!(
        stats.chunk_count >= 2,
        "长文应被分块，实际 {}",
        stats.chunk_count
    );

    // 2. 检索应命中该 tag
    let q = "服务端透明分块与相似检索".to_string();
    let (qv, _) = embedder.embed(vec![q], true).await.unwrap();
    let snap = writer.snapshot(scoop);
    let hits = snap.index.search(&qv[0], 30);
    assert!(!hits.is_empty(), "索引里应有块");
    let results = aggregate(&hits, &snap.chunk_owner, 10, None);
    assert!(results.iter().any(|h| h.tag == tag), "检索应命中 tag {tag}");

    // 3. 覆写为短文本：块数应为 1，旧块全部替换
    let stats = writer.upsert(scoop, tag, "短文本。".into()).await.unwrap();
    assert_eq!(stats.chunk_count, 1);
    assert_eq!(writer.health()[0].2, 1, "覆写后只剩 1 块");

    // 4. 删除
    writer.delete(scoop, tag).await.unwrap();
    assert_eq!(writer.health()[0].1, 0, "删除后 tag 应为 0");

    // 5. 重启模拟：从磁盘恢复，状态一致
    drop(writer);
    let stores2 = StoreSet::load(&config).unwrap();
    assert_eq!(stores2.scoop_stats()[0].1, 0);
    assert_eq!(stores2.scoop_stats()[0].2, 0);

    let _ = std::fs::remove_dir_all(&config.data_dir);
}
