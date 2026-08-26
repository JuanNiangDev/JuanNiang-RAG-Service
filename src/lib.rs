//! 库入口：模块全部走 lib，main 与集成测试共用。

pub mod api;
pub mod chunker;
pub mod config;
pub mod embedding;
pub mod error;
pub mod metrics;
pub mod search;
pub mod store;
pub mod vector_index;
pub mod writer;
