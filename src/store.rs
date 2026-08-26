//! 存储层（阶段 4）：tag（uuid）↔ 块向量（1:N）的持久化与状态管理。
//!
//! 分库（scoop）机制：每个功能块一个独立向量库，检索只在目标 scoop 内进行，
//! 避免不同集合的向量互相污染 top-k（此前知识/记忆/群管理共库，外来向量
//! 挤占候选位导致精确度下降）。
//!
//! 每个 scoop 双文件快照（零 SQL）：
//! - `{data_dir}/scoops/{scoop}/index.tvim`：turbovec 压缩向量 + u64 块 id 映射（真源①）
//! - `{data_dir}/scoops/{scoop}/tags.bin`：tag_to_ids + next_id + 原始归一化向量（真源②，重建用）
//!
//! 原子写：临时文件 + rename。崩溃最多丢最后一次发布。
//! 自愈：启动时 `index.len() != 块数总和` 则用原始向量重建索引。
//! 一致性：同一 tag 只允许归属一个 scoop（写者维护 tag → scoop 注册表），
//! 跨 scoop 重复写入返回 409，杜绝"删除漏删出死数据"。

use crate::config::Config;
use crate::vector_index::VectorError;
use crate::vector_index::VectorIndex;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use thiserror::Error;
use tracing::{info, warn};
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("序列化失败: {0}")]
    Serialize(String),
    #[error("反序列化失败: {0}")]
    Deserialize(String),
    #[error("IO 错误: {0}")]
    Io(String),
    #[error("向量索引错误: {0}")]
    Vector(#[from] VectorError),
    #[error("tag 不存在: {0}")]
    TagNotFound(Uuid),
    #[error("tag {0} 同时归属分库 {1} 与 {2}（数据损坏），拒绝恢复：请人工核查两个分库的 tags.bin")]
    TagMultiScoop(Uuid, Scoop, Scoop),
    #[error("未知分库: {0}（白名单: {1}）")]
    UnknownScoop(String, &'static str),
    #[error("tag {0} 已归属分库 {1}，跨分库重复写入被拒绝")]
    TagInOtherScoop(Uuid, Scoop),
}

/// Scoop 分库白名单：业务语义强绑定（一个功能块一个库），硬编码枚举防止
/// 拼写错误导致数据分散（拼错会被立刻 400 拒绝，而不是静默开新空库）。
/// scoop 名直接拼接为文件路径，白名单天然免疫路径穿越。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Scoop {
    /// 知识库条目（Go 侧 ragtag 前缀 k:）
    Knowledge,
    /// 长期记忆条目（前缀 m:）
    Memory,
    /// 群管理黑白语录/词条（前缀 s: / wt:；黑白同库是因一次检索需同时取两边最优命中）
    GroupMgr,
    /// 插件 jn.rag 通用 API 的默认库（独立库避免与业务集合互相污染）
    Plugin,
}

impl Scoop {
    /// 全部白名单：加载时按此补齐缺失的空库
    pub const ALL: [Scoop; 4] = [
        Scoop::Knowledge,
        Scoop::Memory,
        Scoop::GroupMgr,
        Scoop::Plugin,
    ];

    pub const fn as_str(&self) -> &'static str {
        match self {
            Scoop::Knowledge => "knowledge",
            Scoop::Memory => "memory",
            Scoop::GroupMgr => "groupmgr",
            Scoop::Plugin => "plugin",
        }
    }
}

impl FromStr for Scoop {
    type Err = StoreError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "knowledge" => Ok(Scoop::Knowledge),
            "memory" => Ok(Scoop::Memory),
            "groupmgr" => Ok(Scoop::GroupMgr),
            "plugin" => Ok(Scoop::Plugin),
            _ => Err(StoreError::UnknownScoop(
                s.to_string(),
                "knowledge, memory, groupmgr, plugin",
            )),
        }
    }
}

impl std::fmt::Display for Scoop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// axum Path 提取：非法 scoop 名 → 400（错误信息带白名单）
impl<'de> Deserialize<'de> for Scoop {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// 发布给读者的不可变快照（写者 COW 拷贝后原子替换）
pub struct IndexSnapshot {
    pub index: VectorIndex,
    /// tag → 块 id 列表
    pub tag_to_ids: HashMap<Uuid, Vec<u64>>,
    /// 块 id → tag（聚合检索用）
    pub chunk_owner: HashMap<u64, Uuid>,
    pub next_id: u64,
}

impl IndexSnapshot {
    pub fn tag_count(&self) -> usize {
        self.tag_to_ids.len()
    }

    pub fn chunk_count(&self) -> usize {
        self.index.len()
    }
}

/// tags.bin 的磁盘格式
#[derive(Serialize, Deserialize, Default)]
struct TagsFile {
    tag_to_ids: HashMap<Uuid, Vec<u64>>,
    next_id: u64,
    /// 块 id → 原始归一化向量（重建索引用；可配置关闭）
    raw_vectors: HashMap<u64, Vec<f32>>,
}

/// 单个 scoop 的存储目录：{data_dir}/scoops/{name}/
fn scoop_dir(data_dir: &str, scoop: Scoop) -> PathBuf {
    PathBuf::from(data_dir).join("scoops").join(scoop.as_str())
}

/// 写者独占的可变状态（单个 scoop）
pub struct TagStore {
    index: VectorIndex,
    tag_to_ids: HashMap<Uuid, Vec<u64>>,
    next_id: u64,
    raw_vectors: HashMap<u64, Vec<f32>>,
    /// 是否持久化原始向量（配置）
    keep_raw: bool,
    data_dir: PathBuf,
}

impl TagStore {
    /// 从磁盘加载单个 scoop；无数据则返回空库
    pub fn load(config: &Config, scoop: Scoop) -> Result<Self, StoreError> {
        let data_dir = scoop_dir(&config.data_dir, scoop);
        let tags_path = data_dir.join("tags.bin");
        let index_path = data_dir.join("index.tvim");

        let mut store = if tags_path.exists() {
            Self::load_tags(&tags_path, &index_path, config, scoop)?
        } else {
            Self::empty(config, scoop)?
        };
        // 自愈：块数不一致 → 从原始向量重建索引
        let expected: usize = store.tag_to_ids.values().map(|v| v.len()).sum();
        if store.index.len() != expected {
            info!(
                "分库 {scoop}: 索引块数 {} 与 tags.bin 块数 {} 不一致，触发重建",
                store.index.len(),
                expected
            );
            store.rebuild_index()?;
            store.persist()?;
        }
        Ok(store)
    }

    fn load_tags(
        tags_path: &Path,
        index_path: &Path,
        config: &Config,
        scoop: Scoop,
    ) -> Result<Self, StoreError> {
        let bytes = std::fs::read(tags_path).map_err(|e| StoreError::Io(e.to_string()))?;
        let f: TagsFile =
            bincode::deserialize(&bytes).map_err(|e| StoreError::Deserialize(e.to_string()))?;

        let index = if index_path.exists() {
            VectorIndex::load(index_path).unwrap_or_else(|e| {
                info!("index.tvim 加载失败（{e}），稍后由一致性校验触发重建");
                VectorIndex::new(config.dim, config.bit_width).expect("构造空索引失败")
            })
        } else {
            VectorIndex::new(config.dim, config.bit_width).expect("构造空索引失败")
        };

        Ok(Self {
            index,
            tag_to_ids: f.tag_to_ids,
            next_id: f.next_id,
            raw_vectors: f.raw_vectors,
            keep_raw: config.store_raw_vectors,
            data_dir: scoop_dir(&config.data_dir, scoop),
        })
    }

    fn empty(config: &Config, scoop: Scoop) -> Result<Self, StoreError> {
        Ok(Self {
            index: VectorIndex::new(config.dim, config.bit_width).map_err(StoreError::Vector)?,
            tag_to_ids: HashMap::new(),
            next_id: 0,
            raw_vectors: HashMap::new(),
            keep_raw: config.store_raw_vectors,
            data_dir: scoop_dir(&config.data_dir, scoop),
        })
    }

    /// 从原始向量全量重建索引
    fn rebuild_index(&mut self) -> Result<(), StoreError> {
        let mut new_index = VectorIndex::new(self.index.dim(), 4).map_err(StoreError::Vector)?;
        // 原始向量缺失（keep_raw=false）时无法重建，保留原索引并告警
        if self.raw_vectors.is_empty() && !self.tag_to_ids.is_empty() {
            return Err(StoreError::Io(
                "原始向量未持久化，无法重建索引（需 Agent 重推全文）".to_string(),
            ));
        }
        let mut vectors = Vec::with_capacity(self.raw_vectors.len() * self.index.dim());
        let mut ids = Vec::with_capacity(self.raw_vectors.len());
        for (&id, v) in &self.raw_vectors {
            vectors.extend_from_slice(v);
            ids.push(id);
        }
        new_index
            .add_with_ids(&vectors, &ids)
            .map_err(StoreError::Vector)?;
        new_index.prepare();
        self.index = new_index;
        Ok(())
    }

    /// 原子双快照持久化：临时文件 + rename
    pub fn persist(&self) -> Result<(), StoreError> {
        std::fs::create_dir_all(&self.data_dir).map_err(|e| StoreError::Io(e.to_string()))?;

        let tags_path = self.data_dir.join("tags.bin");
        let index_path = self.data_dir.join("index.tvim");
        let tags_tmp = self.data_dir.join("tags.bin.tmp");
        let index_tmp = self.data_dir.join("index.tvim.tmp");

        let f = TagsFile {
            tag_to_ids: self.tag_to_ids.clone(),
            next_id: self.next_id,
            raw_vectors: if self.keep_raw {
                self.raw_vectors.clone()
            } else {
                HashMap::new()
            },
        };
        let bytes = bincode::serialize(&f).map_err(|e| StoreError::Serialize(e.to_string()))?;
        std::fs::write(&tags_tmp, bytes).map_err(|e| StoreError::Io(e.to_string()))?;
        self.index.write(&index_tmp)?;

        // rename 是原子操作；先 tags 后 index，崩溃时以 tags.bin 为准做一致性校验
        std::fs::rename(&tags_tmp, &tags_path).map_err(|e| StoreError::Io(e.to_string()))?;
        std::fs::rename(&index_tmp, &index_path).map_err(|e| StoreError::Io(e.to_string()))?;
        Ok(())
    }

    // ---------- 写操作（仅写者线程调用） ----------

    /// upsert：替换 tag 的全部旧块，写入新块
    pub fn upsert(
        &mut self,
        tag: Uuid,
        chunks: &[String],
        vectors: &[Vec<f32>],
    ) -> Result<usize, StoreError> {
        debug_assert_eq!(chunks.len(), vectors.len());
        let dim = self.index.dim();

        // 删除旧块
        if let Some(old_ids) = self.tag_to_ids.remove(&tag) {
            for id in old_ids {
                self.index.remove(id);
                self.raw_vectors.remove(&id);
            }
        }

        // 分配新块 id 并写入
        let mut ids = Vec::with_capacity(vectors.len());
        let mut flat = Vec::with_capacity(vectors.len() * dim);
        for v in vectors {
            if v.len() != dim {
                return Err(StoreError::Io(format!(
                    "向量维度 {}(实际) != {}(期望)",
                    v.len(),
                    dim
                )));
            }
            let id = self.next_id;
            self.next_id += 1;
            ids.push(id);
            flat.extend_from_slice(v);
            self.raw_vectors.insert(id, v.clone());
        }
        self.index.add_with_ids(&flat, &ids)?;
        self.tag_to_ids.insert(tag, ids);
        Ok(vectors.len())
    }

    /// 删除 tag 及其全部块
    pub fn delete(&mut self, tag: Uuid) -> Result<(), StoreError> {
        let ids = self
            .tag_to_ids
            .remove(&tag)
            .ok_or(StoreError::TagNotFound(tag))?;
        for id in ids {
            self.index.remove(id);
            self.raw_vectors.remove(&id);
        }
        Ok(())
    }

    // ---------- 快照发布 ----------

    /// COW 快照：索引走序列化往返拷贝，映射克隆
    pub fn snapshot(&self) -> Result<IndexSnapshot, StoreError> {
        std::fs::create_dir_all(&self.data_dir).map_err(|e| StoreError::Io(e.to_string()))?;
        let tmp = self.data_dir.join(".cow.tvim");
        let index = self.index.copy(&tmp).map_err(StoreError::Vector)?;
        let _ = std::fs::remove_file(&tmp);

        let mut chunk_owner = HashMap::with_capacity(self.index.len());
        for (&tag, ids) in &self.tag_to_ids {
            for &id in ids {
                chunk_owner.insert(id, tag);
            }
        }
        Ok(IndexSnapshot {
            index,
            tag_to_ids: self.tag_to_ids.clone(),
            chunk_owner,
            next_id: self.next_id,
        })
    }

    pub fn tag_count(&self) -> usize {
        self.tag_to_ids.len()
    }

    pub fn chunk_count(&self) -> usize {
        self.index.len()
    }
}

/// 全部分库的集合：写者独占。每个 scoop 独立的 TagStore + 独立的持久化/快照，
/// 一次写入只影响本 scoop（其余 scoop 的快照/磁盘文件不参与 COW 与写盘）。
pub struct StoreSet {
    stores: HashMap<Scoop, TagStore>,
    /// tag → scoop 归属注册表：防同一 tag 跨 scoop 重复写入（重启时从各库重建）
    tag_owner: HashMap<Uuid, Scoop>,
}

impl StoreSet {
    /// 从磁盘加载全部分库：扫描 scoops/ 目录只加载白名单内 scoop，
    /// 白名单中缺失的分库初始化为空库（全新部署 / 局部删除后自动补齐）。
    /// 未知目录告警跳过（防旧版本遗留脏数据混入）。
    pub fn load(config: &Config) -> Result<Self, StoreError> {
        let mut stores = HashMap::new();
        let base = PathBuf::from(&config.data_dir).join("scoops");
        match std::fs::read_dir(&base) {
            Ok(entries) => {
                for entry in entries {
                    let entry =
                        entry.map_err(|e| StoreError::Io(format!("读取分库目录项失败: {e}")))?;
                    let name = entry.file_name().to_string_lossy().into_owned();
                    match Scoop::from_str(&name) {
                        Ok(scoop) => {
                            let store = TagStore::load(config, scoop)?;
                            stores.entry(scoop).or_insert(store);
                        }
                        Err(_) if entry.path().is_dir() => {
                            warn!("分库目录 {name} 不在白名单内，跳过加载（如需使用请升级白名单）");
                        }
                        Err(_) => {}
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // 全新部署：scoops/ 尚不存在，走下方白名单初始化空库
            }
            Err(e) => return Err(StoreError::Io(format!("扫描分库目录失败: {e}"))),
        }
        for scoop in Scoop::ALL {
            stores
                .entry(scoop)
                .or_insert_with(|| TagStore::empty(config, scoop).expect("构造空分库失败"));
        }

        let mut set = Self {
            stores,
            tag_owner: HashMap::new(),
        };
        set.rebuild_tag_owner()?;
        Ok(set)
    }

    /// 从各库 tag_to_ids 重建 tag → scoop 注册表（启动/手动调用）。
    ///
    /// 同一 tag 出现在多个分库视为数据损坏：确定性报错（绝不静默覆盖，
    /// 且不依赖 HashMap 迭代顺序），需人工核查后处理两个分库的 tags.bin。
    fn rebuild_tag_owner(&mut self) -> Result<(), StoreError> {
        self.tag_owner.clear();
        let mut pairs: Vec<(Scoop, Uuid)> = Vec::new();
        for (&scoop, store) in &self.stores {
            for &tag in store.tag_to_ids.keys() {
                pairs.push((scoop, tag));
            }
        }
        pairs.sort(); // 确定性顺序：与 HashMap 迭代无关
        for (scoop, tag) in pairs {
            if let Some(prev) = self.tag_owner.insert(tag, scoop)
                && prev != scoop
            {
                return Err(StoreError::TagMultiScoop(tag, prev, scoop));
            }
        }
        Ok(())
    }

    fn scoop(&self, scoop: Scoop) -> &TagStore {
        &self.stores[&scoop]
    }

    fn scoop_mut(&mut self, scoop: Scoop) -> &mut TagStore {
        self.stores.get_mut(&scoop).unwrap_or_else(|| {
            panic!("分库 {scoop} 未初始化（白名单与存储不一致）");
        })
    }

    /// upsert（带归属校验）：tag 已归属其他 scoop 时拒绝（409）。
    pub fn upsert(
        &mut self,
        scoop: Scoop,
        tag: Uuid,
        chunks: &[String],
        vectors: &[Vec<f32>],
    ) -> Result<usize, StoreError> {
        if let Some(&owner) = self.tag_owner.get(&tag)
            && owner != scoop
        {
            return Err(StoreError::TagInOtherScoop(tag, owner));
        }
        let n = self.scoop_mut(scoop).upsert(tag, chunks, vectors)?;
        self.tag_owner.insert(tag, scoop);
        Ok(n)
    }

    /// 删除（带归属校验）：只能删除属于本 scoop 的 tag。
    pub fn delete(&mut self, scoop: Scoop, tag: Uuid) -> Result<(), StoreError> {
        if let Some(&owner) = self.tag_owner.get(&tag)
            && owner != scoop
        {
            return Err(StoreError::TagInOtherScoop(tag, owner));
        }
        self.scoop_mut(scoop).delete(tag)?;
        self.tag_owner.remove(&tag);
        Ok(())
    }

    pub fn snapshot(&self, scoop: Scoop) -> Result<IndexSnapshot, StoreError> {
        self.scoop(scoop).snapshot()
    }

    pub fn persist(&self, scoop: Scoop) -> Result<(), StoreError> {
        self.scoop(scoop).persist()
    }

    /// 各 scoop 规模：(scoop, tag 数, 块数)
    pub fn scoop_stats(&self) -> Vec<(Scoop, usize, usize)> {
        Scoop::ALL
            .iter()
            .map(|&s| {
                let st = self.scoop(s);
                (s, st.tag_count(), st.chunk_count())
            })
            .collect()
    }
}

// ---------- 单元测试（合成数据，不需要模型） ----------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(dir: &Path) -> Config {
        Config {
            model_path: "none".into(),
            data_dir: dir.to_string_lossy().into_owned(),
            host: "127.0.0.1".into(),
            port: 0,
            n_threads: 4,
            n_ctx: 512,
            dim: 8, // 测试用小维度（8 的倍数）
            bit_width: 4,
            max_chunk_chars: 260,
            overlap_chars: 50,
            store_raw_vectors: true,
            lru_capacity: 128,
        }
    }

    fn fake_vector(dim: usize, seed: f32) -> Vec<f32> {
        let mut v = vec![seed; dim];
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }

    #[test]
    fn upsert_delete_snapshot_cycle() {
        let dir = std::env::temp_dir().join(format!("rag_test_{}", Uuid::new_v4()));
        let cfg = test_config(&dir);
        let mut store = TagStore::empty(&cfg, Scoop::Knowledge).unwrap();
        let tag = Uuid::new_v4();

        let n = store
            .upsert(
                tag,
                &["块一".into(), "块二".into()],
                &[fake_vector(8, 0.5), fake_vector(8, 0.7)],
            )
            .unwrap();
        assert_eq!(n, 2);
        assert_eq!(store.tag_count(), 1);
        assert_eq!(store.chunk_count(), 2);

        // 覆写：同 tag 再写 1 块，旧 2 块应消失
        let n = store
            .upsert(tag, &["新块".into()], &[fake_vector(8, 0.9)])
            .unwrap();
        assert_eq!(n, 1);
        assert_eq!(store.chunk_count(), 1);

        // 快照
        let snap = store.snapshot().unwrap();
        assert_eq!(snap.tag_count(), 1);
        assert_eq!(snap.chunk_count(), 1);
        assert_eq!(snap.chunk_owner.len(), 1);

        // 持久化 + 重新加载
        store.persist().unwrap();
        drop(store);
        let mut reloaded = TagStore::load(&cfg, Scoop::Knowledge).unwrap();
        assert_eq!(reloaded.tag_count(), 1);
        assert_eq!(reloaded.chunk_count(), 1);
        assert_eq!(reloaded.next_id, 3, "两次 upsert 共分配了 3 个 id");

        // 删除
        reloaded.delete(tag).unwrap();
        assert_eq!(reloaded.tag_count(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_missing_data_returns_empty() {
        let dir = std::env::temp_dir().join(format!("rag_missing_{}", Uuid::new_v4()));
        let cfg = test_config(&dir);
        let store = TagStore::load(&cfg, Scoop::Memory).unwrap();
        assert_eq!(store.tag_count(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rebuild_when_index_missing() {
        let dir = std::env::temp_dir().join(format!("rag_rebuild_{}", Uuid::new_v4()));
        let cfg = test_config(&dir);
        let mut store = TagStore::empty(&cfg, Scoop::Knowledge).unwrap();
        store
            .upsert(Uuid::new_v4(), &["内容".into()], &[fake_vector(8, 0.3)])
            .unwrap();
        store.persist().unwrap();

        // 删掉 index.tvim，模拟损坏
        std::fs::remove_file(dir.join("scoops/knowledge/index.tvim")).unwrap();

        // load 应触发自愈重建
        let reloaded = TagStore::load(&cfg, Scoop::Knowledge).unwrap();
        assert_eq!(reloaded.chunk_count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scoop_parse_whitelist() {
        assert_eq!("knowledge".parse::<Scoop>().unwrap(), Scoop::Knowledge);
        assert_eq!("memory".parse::<Scoop>().unwrap(), Scoop::Memory);
        assert_eq!("groupmgr".parse::<Scoop>().unwrap(), Scoop::GroupMgr);
        assert_eq!("plugin".parse::<Scoop>().unwrap(), Scoop::Plugin);
        assert!(matches!(
            "随意名字".parse::<Scoop>(),
            Err(StoreError::UnknownScoop(..))
        ));
        assert!(matches!(
            "../etc".parse::<Scoop>(),
            Err(StoreError::UnknownScoop(..))
        ));
    }

    #[test]
    fn store_set_isolates_scoops() {
        let dir = std::env::temp_dir().join(format!("rag_set_{}", Uuid::new_v4()));
        let cfg = test_config(&dir);
        let mut set = StoreSet::load(&cfg).unwrap();

        // 四个 scoop 全部就绪（空库补齐）
        let stats = set.scoop_stats();
        assert_eq!(stats.len(), Scoop::ALL.len());
        assert!(stats.iter().all(|&(_, t, c)| t == 0 && c == 0));

        // 不同 tag 分别写入 knowledge 与 memory（内容不同），互不干扰
        let tag_k = Uuid::new_v4();
        let tag_m = Uuid::new_v4();
        set.upsert(
            Scoop::Knowledge,
            tag_k,
            &["知识内容".into()],
            &[fake_vector(8, 0.5)],
        )
        .unwrap();
        set.upsert(
            Scoop::Memory,
            tag_m,
            &["记忆内容".into()],
            &[fake_vector(8, 0.7)],
        )
        .unwrap();
        assert_eq!(set.scoop(Scoop::Knowledge).chunk_count(), 1);
        assert_eq!(set.scoop(Scoop::Memory).chunk_count(), 1);

        // 跨 scoop 覆写/删除被拒绝（409 语义；同 tag 只允许归属一个 scoop）
        assert!(matches!(
            set.upsert(
                Scoop::GroupMgr,
                tag_k,
                &["别处".into()],
                &[fake_vector(8, 0.1)],
            ),
            Err(StoreError::TagInOtherScoop(..))
        ));
        assert!(matches!(
            set.delete(Scoop::GroupMgr, tag_k),
            Err(StoreError::TagInOtherScoop(..))
        ));

        // 归属内删除成功，注册表同步移除；之后可重新写入任意 scoop
        set.delete(Scoop::Knowledge, tag_k).unwrap();
        assert!(matches!(
            set.upsert(
                Scoop::GroupMgr,
                tag_k,
                &["重生".into()],
                &[fake_vector(8, 0.2)],
            ),
            Ok(1)
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 同一 tag 的 tags.bin 出现在两个 scoop 目录（数据损坏）：
    /// 启动重建注册表必须确定性报错，而不是静默覆盖归属。
    #[test]
    fn rebuild_owner_detects_duplicate_tag() {
        let dir = std::env::temp_dir().join(format!("rag_dup_{}", Uuid::new_v4()));
        let cfg = test_config(&dir);

        // 手工向两个分库目录写入同一 tag 的 tags.bin
        let tag = Uuid::new_v4();
        for scoop in [Scoop::Knowledge, Scoop::Memory] {
            let mut store = TagStore::empty(&cfg, scoop).unwrap();
            store
                .upsert(tag, &["内容".into()], &[fake_vector(8, 0.5)])
                .unwrap();
            store.persist().unwrap();
        }

        // 加载必须失败，且错误携带两个冲突分库名
        let err = match StoreSet::load(&cfg) {
            Ok(_) => panic!("损坏数据不应加载成功"),
            Err(e) => e,
        };
        match err {
            StoreError::TagMultiScoop(t, a, b) => {
                assert_eq!(t, tag);
                let names = [a.as_str(), b.as_str()];
                assert!(names.contains(&"knowledge"));
                assert!(names.contains(&"memory"));
            }
            other => panic!("期望 TagMultiScoop，实际: {other}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn store_set_load_skips_unknown_dirs() {
        let dir = std::env::temp_dir().join(format!("rag_scan_{}", Uuid::new_v4()));
        let cfg = test_config(&dir);
        std::fs::create_dir_all(dir.join("scoops/legacy_old")).unwrap();

        let set = StoreSet::load(&cfg).unwrap();
        // 未知目录不加载、不报错；白名单全部分库就绪
        assert_eq!(set.scoop_stats().len(), Scoop::ALL.len());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
