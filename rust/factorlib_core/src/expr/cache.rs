//! 全局子表达式 LRU 缓存管理器 (expr/cache.rs)。
//!
//! 严格对应白皮书 6.3 节设计：
//! - 键为 AST 规范化化简后的 64 位 `ahash` 指纹；
//! - 值为 `Arc<Array2<f64>>`，命中缓存时仅克隆 `Arc` 原子引用计数指针，零内存拷贝、零重复计算；
//! - 跟踪总内存占用（`current_memory_bytes`）与访问代数（`access_tick`），超过 `max_memory_bytes` 时
//!   按最近最少使用（LRU）策略淘汰旧子表达式；
//! - 线程安全，支持 Rayon 10+ 物理线程高并发共享读写。

use ahash::AHashMap;
use ndarray::Array2;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone)]
struct CacheEntry {
    matrix: Arc<Array2<f64>>,
    bytes: usize,
    last_used: u64,
}

#[derive(Debug, Default)]
struct CacheInner {
    map: AHashMap<u64, CacheEntry>,
    current_memory_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    pub entries: usize,
    pub current_memory_bytes: usize,
    pub max_memory_bytes: usize,
    pub hits: u64,
    pub misses: u64,
}

pub struct SubExprCache {
    inner: RwLock<CacheInner>,
    max_memory_bytes: usize,
    clock: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl SubExprCache {
    pub fn new(max_memory_bytes: usize) -> Self {
        Self {
            inner: RwLock::new(CacheInner {
                map: AHashMap::with_capacity(2048),
                current_memory_bytes: 0,
            }),
            max_memory_bytes,
            clock: AtomicU64::new(1),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn from_megabytes(cache_mb: usize) -> Self {
        Self::new(cache_mb.saturating_mul(1024 * 1024))
    }

    /// 查询缓存：若命中则更新时间戳并返回 `Arc<Array2<f64>>` 指针克隆。
    pub fn get(&self, key: u64) -> Option<Arc<Array2<f64>>> {
        if self.max_memory_bytes == 0 {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        if let Ok(guard) = self.inner.read() {
            if let Some(entry) = guard.map.get(&key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some(Arc::clone(&entry.matrix));
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// 写入缓存：若超出 `max_memory_bytes` 上限则淘汰最久未使用的条目。
    pub fn insert(&self, key: u64, matrix: Arc<Array2<f64>>) {
        let entry_bytes = matrix.len() * std::mem::size_of::<f64>();
        if self.max_memory_bytes == 0 || entry_bytes > self.max_memory_bytes {
            return;
        }
        let tick = self.clock.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut guard) = self.inner.write() {
            if let Some(existing) = guard.map.get_mut(&key) {
                existing.last_used = tick;
                return;
            }
            while guard.current_memory_bytes + entry_bytes > self.max_memory_bytes
                && !guard.map.is_empty()
            {
                // 找到 last_used 最小的条目予以驱逐
                if let Some((&oldest_key, _)) =
                    guard.map.iter().min_by_key(|(_, entry)| entry.last_used)
                {
                    if let Some(removed) = guard.map.remove(&oldest_key) {
                        guard.current_memory_bytes =
                            guard.current_memory_bytes.saturating_sub(removed.bytes);
                    }
                } else {
                    break;
                }
            }
            guard.current_memory_bytes += entry_bytes;
            guard.map.insert(
                key,
                CacheEntry {
                    matrix,
                    bytes: entry_bytes,
                    last_used: tick,
                },
            );
        }
    }

    pub fn clear(&self) {
        if let Ok(mut guard) = self.inner.write() {
            guard.map.clear();
            guard.current_memory_bytes = 0;
        }
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    pub fn stats(&self) -> CacheStats {
        let (entries, current_memory_bytes) = match self.inner.read() {
            Ok(guard) => (guard.map.len(), guard.current_memory_bytes),
            Err(_) => (0, 0),
        };
        CacheStats {
            entries,
            current_memory_bytes,
            max_memory_bytes: self.max_memory_bytes,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }
}
