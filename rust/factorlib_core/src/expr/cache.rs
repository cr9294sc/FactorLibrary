//! 全局分片 O(1) 子表达式真 LRU 缓存管理器 (expr/cache.rs)。
//!
//! 设计要点（P1-05 修复）：
//! - 32 路独立互斥分片（Sharded Mutex），消除跨线程全局写锁争用；
//! - 键由 `(u64 哈希选片, Arc<str> 规范表达式原文)` 组成，命中时严格比对表达式字符串，彻底消除 64 位哈希碰撞串值隐患；
//! - 每个分片内基于 Vec 槽位索引维护侵入式双向链表，实现 O(1) 命中置顶（真 LRU）与 O(1) 尾部驱逐；
//! - 当全局总内存接近 `max_memory_bytes` 时优先在当前分片驱逐，仅在必要时跨分片单锁探查 `NUM_SHARDS` 尾节点，杜绝 $O(M)$ 全表扫描与 AB-BA 死锁；
//! - 值为 `Arc<Array2<f64>>`，命中缓存时仅克隆引用计数指针，零矩阵内存拷贝。

use ahash::AHashMap;
use ndarray::Array2;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const NUM_SHARDS: usize = 32;
const NONE: usize = usize::MAX;

#[derive(Debug)]
struct Slot {
    key: Arc<str>,
    matrix: Arc<Array2<f64>>,
    bytes: usize,
    last_used: u64,
    prev: usize,
    next: usize,
}

#[derive(Debug)]
struct LruShard {
    map: AHashMap<Arc<str>, usize>,
    slots: Vec<Option<Slot>>,
    free: Vec<usize>,
    head: usize,
    tail: usize,
    bytes: usize,
}

impl Default for LruShard {
    fn default() -> Self {
        Self::new()
    }
}

impl LruShard {
    fn new() -> Self {
        Self {
            map: AHashMap::with_capacity(64),
            slots: Vec::with_capacity(64),
            free: Vec::new(),
            head: NONE,
            tail: NONE,
            bytes: 0,
        }
    }

    fn link_front(&mut self, idx: usize) {
        let old_head = self.head;
        if let Some(s) = self.slots[idx].as_mut() {
            s.prev = NONE;
            s.next = old_head;
        }
        if old_head != NONE {
            if let Some(h) = self.slots[old_head].as_mut() {
                h.prev = idx;
            }
        } else {
            self.tail = idx;
        }
        self.head = idx;
    }

    fn unlink(&mut self, idx: usize) {
        let (prev, next) = match self.slots[idx].as_ref() {
            Some(s) => (s.prev, s.next),
            None => return,
        };
        if prev != NONE {
            if let Some(p) = self.slots[prev].as_mut() {
                p.next = next;
            }
        } else {
            self.head = next;
        }
        if next != NONE {
            if let Some(n) = self.slots[next].as_mut() {
                n.prev = prev;
            }
        } else {
            self.tail = prev;
        }
    }

    fn touch(&mut self, idx: usize, tick: u64) {
        if let Some(s) = self.slots[idx].as_mut() {
            s.last_used = tick;
        }
        if self.head != idx {
            self.unlink(idx);
            self.link_front(idx);
        }
    }

    fn pop_lru(&mut self) -> Option<usize> {
        let idx = self.tail;
        if idx == NONE {
            return None;
        }
        self.unlink(idx);
        let slot = self.slots[idx].take()?;
        self.free.push(idx);
        self.map.remove(&slot.key);
        self.bytes = self.bytes.saturating_sub(slot.bytes);
        Some(slot.bytes)
    }

    fn tail_tick(&self) -> Option<u64> {
        if self.tail == NONE {
            None
        } else {
            self.slots[self.tail].as_ref().map(|s| s.last_used)
        }
    }

    fn upsert(
        &mut self,
        canon_expr: Arc<str>,
        matrix: Arc<Array2<f64>>,
        bytes: usize,
        tick: u64,
    ) -> isize {
        if let Some(&idx) = self.map.get(canon_expr.as_ref()) {
            let old_bytes = self.slots[idx].as_ref().map(|s| s.bytes).unwrap_or(0);
            if let Some(s) = self.slots[idx].as_mut() {
                s.matrix = matrix;
                s.bytes = bytes;
            }
            self.bytes = self.bytes.saturating_sub(old_bytes) + bytes;
            self.touch(idx, tick);
            return bytes as isize - old_bytes as isize;
        }

        let slot = Slot {
            key: Arc::clone(&canon_expr),
            matrix,
            bytes,
            last_used: tick,
            prev: NONE,
            next: NONE,
        };
        let idx = if let Some(free_idx) = self.free.pop() {
            self.slots[free_idx] = Some(slot);
            free_idx
        } else {
            let new_idx = self.slots.len();
            self.slots.push(Some(slot));
            new_idx
        };
        self.link_front(idx);
        self.map.insert(canon_expr, idx);
        self.bytes += bytes;
        bytes as isize
    }

    fn clear(&mut self) {
        self.map.clear();
        self.slots.clear();
        self.free.clear();
        self.head = NONE;
        self.tail = NONE;
        self.bytes = 0;
    }
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
    shards: [Mutex<LruShard>; NUM_SHARDS],
    max_memory_bytes: usize,
    current_memory_bytes: AtomicUsize,
    clock: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl SubExprCache {
    pub fn new(max_memory_bytes: usize) -> Self {
        Self {
            shards: std::array::from_fn(|_| Mutex::new(LruShard::new())),
            max_memory_bytes,
            current_memory_bytes: AtomicUsize::new(0),
            clock: AtomicU64::new(1),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn from_megabytes(cache_mb: usize) -> Self {
        Self::new(cache_mb.saturating_mul(1024 * 1024))
    }

    #[inline]
    fn shard_index(key_hash: u64) -> usize {
        (key_hash as usize) & (NUM_SHARDS - 1)
    }

    #[inline]
    fn try_reserve_bytes(&self, add: usize) -> bool {
        if add == 0 {
            return true;
        }
        let mut cur = self.current_memory_bytes.load(Ordering::Acquire);
        loop {
            let Some(next) = cur.checked_add(add) else {
                return false;
            };
            if next > self.max_memory_bytes {
                return false;
            }
            match self.current_memory_bytes.compare_exchange_weak(
                cur,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => cur = actual,
            }
        }
    }

    #[inline]
    fn release_global_bytes(&self, sub: usize) {
        if sub == 0 {
            return;
        }
        let mut cur = self.current_memory_bytes.load(Ordering::Acquire);
        while let Err(actual) = self.current_memory_bytes.compare_exchange_weak(
            cur,
            cur.saturating_sub(sub),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            cur = actual;
        }
    }

    /// 当快速预留失败时，逐片单锁扫描最旧 LRU 尾部并驱逐，直到腾出足够全局字节预算（杜绝死锁与 O(M) 扫描）。
    fn reserve_with_eviction(&self, needed: usize) -> bool {
        if self.try_reserve_bytes(needed) {
            return true;
        }
        let mut empty_scans = 0usize;
        while !self.try_reserve_bytes(needed) {
            let mut oldest: Option<(usize, u64)> = None;
            for i in 0..NUM_SHARDS {
                if let Ok(s) = self.shards[i].lock() {
                    if let Some(t) = s.tail_tick() {
                        if oldest.is_none_or(|(_, best_t)| t < best_t) {
                            oldest = Some((i, t));
                        }
                    }
                }
            }
            let Some((victim_idx, _)) = oldest else {
                empty_scans += 1;
                if empty_scans >= 8 {
                    return false;
                }
                std::thread::yield_now();
                continue;
            };
            empty_scans = 0;
            if let Ok(mut victim) = self.shards[victim_idx].lock() {
                if let Some(removed) = victim.pop_lru() {
                    self.release_global_bytes(removed);
                }
            }
        }
        true
    }

    /// 查询缓存：通过规范表达式原文精确比对防碰撞，若命中则在 O(1) 时间内刷新 LRU 链表并返回 `Arc` 克隆。
    pub fn get(&self, key_hash: u64, canon_expr: &str) -> Option<Arc<Array2<f64>>> {
        if self.max_memory_bytes == 0 {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let shard_idx = Self::shard_index(key_hash);
        if let Ok(mut shard) = self.shards[shard_idx].lock() {
            if let Some(&idx) = shard.map.get(canon_expr) {
                let tick = self.clock.fetch_add(1, Ordering::Relaxed);
                let mat = shard.slots[idx].as_ref().map(|s| Arc::clone(&s.matrix));
                if let Some(m) = mat {
                    shard.touch(idx, tick);
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    return Some(m);
                }
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// 写入缓存：通过原子预算预留严格保证全局内存上限，超出时按全局 LRU 顺序 O(1) 驱逐最久未使用条目。
    pub fn insert(&self, key_hash: u64, canon_expr: impl Into<Arc<str>>, matrix: Arc<Array2<f64>>) {
        let entry_bytes = matrix.len() * std::mem::size_of::<f64>();
        if self.max_memory_bytes == 0 || entry_bytes > self.max_memory_bytes {
            return;
        }
        let canon_arc: Arc<str> = canon_expr.into();
        let shard_idx = Self::shard_index(key_hash);

        // 单锁快路径：原子预留成功即持锁写入，确保 current_memory_bytes 永不超限
        if let Ok(mut shard) = self.shards[shard_idx].lock() {
            if let Some(&idx) = shard.map.get(canon_arc.as_ref()) {
                let old_bytes = shard.slots[idx].as_ref().map(|s| s.bytes).unwrap_or(0);
                if entry_bytes <= old_bytes {
                    let tick = self.clock.fetch_add(1, Ordering::Relaxed);
                    let _ = shard.upsert(canon_arc, matrix, entry_bytes, tick);
                    self.release_global_bytes(old_bytes - entry_bytes);
                    return;
                }
                if self.try_reserve_bytes(entry_bytes - old_bytes) {
                    let tick = self.clock.fetch_add(1, Ordering::Relaxed);
                    let _ = shard.upsert(canon_arc, matrix, entry_bytes, tick);
                    return;
                }
            } else if self.try_reserve_bytes(entry_bytes) {
                let tick = self.clock.fetch_add(1, Ordering::Relaxed);
                let _ = shard.upsert(canon_arc, matrix, entry_bytes, tick);
                return;
            }
        } else {
            return;
        }

        // 快路径预算不足：释放分片锁后统一预留 entry_bytes，保证写回时 actual_delta <= entry_bytes
        if !self.reserve_with_eviction(entry_bytes) {
            return;
        }

        if let Ok(mut shard) = self.shards[shard_idx].lock() {
            let tick = self.clock.fetch_add(1, Ordering::Relaxed);
            let actual_delta = shard.upsert(canon_arc, matrix, entry_bytes, tick);
            let over_reserved = entry_bytes as isize - actual_delta;
            if over_reserved > 0 {
                self.release_global_bytes(over_reserved as usize);
            }
        } else {
            self.release_global_bytes(entry_bytes);
        }
    }

    pub fn clear(&self) {
        for s in &self.shards {
            if let Ok(mut guard) = s.lock() {
                guard.clear();
            }
        }
        self.current_memory_bytes.store(0, Ordering::Relaxed);
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    pub fn stats(&self) -> CacheStats {
        let mut entries = 0usize;
        for s in &self.shards {
            if let Ok(guard) = s.lock() {
                entries += guard.map.len();
            }
        }
        CacheStats {
            entries,
            current_memory_bytes: self.current_memory_bytes.load(Ordering::Relaxed),
            max_memory_bytes: self.max_memory_bytes,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SubExprCache;
    use ndarray::Array2;
    use rayon::prelude::*;
    use std::sync::Arc;

    fn mat_with(val: f64) -> Arc<Array2<f64>> {
        Arc::new(Array2::from_elem((4, 4), val)) // 16 * 8 = 128 bytes
    }

    #[test]
    fn test_true_lru_evicts_cold_newer_entry_not_hot_older_entry() {
        // 容量恰好容纳 2 个 128B 矩阵
        let cache = SubExprCache::new(256);
        let a = mat_with(1.0);
        let b = mat_with(2.0);
        let c = mat_with(3.0);

        // 即使故意将 A 与 B 放入同一或不同分片，全局 LRU 都应正确生效
        cache.insert(1, "expr_a", Arc::clone(&a));
        cache.insert(2, "expr_b", Arc::clone(&b));

        // 命中 expr_a，刷新其 LRU 新鲜度，此时 expr_b 成为最久未使用项
        let hit_a = cache.get(1, "expr_a").expect("expr_a 应命中");
        assert_eq!(hit_a[[0, 0]], 1.0);

        // 插入 expr_c 触发驱逐：必须淘汰 expr_b 而非 expr_a
        cache.insert(3, "expr_c", Arc::clone(&c));

        assert!(
            cache.get(1, "expr_a").is_some(),
            "高频热键 expr_a 不得被 FIFO 误淘汰"
        );
        assert!(
            cache.get(2, "expr_b").is_none(),
            "冷键 expr_b 应当最先被 LRU 淘汰"
        );
        assert!(
            cache.get(3, "expr_c").is_some(),
            "新插入 expr_c 应当在缓存中"
        );
    }

    #[test]
    fn test_hash_collision_key_isolation() {
        let cache = SubExprCache::new(4096);
        let m1 = mat_with(10.0);
        let m2 = mat_with(20.0);

        // 人为构造完全相同的 u64 key_hash，但规范化表达式原文不同
        let colliding_hash = 0xdead_beef_cafe_babe_u64;
        cache.insert(colliding_hash, "cs_rank(close)", Arc::clone(&m1));
        cache.insert(colliding_hash, "cs_zscore(open)", Arc::clone(&m2));

        let g1 = cache.get(colliding_hash, "cs_rank(close)").unwrap();
        let g2 = cache.get(colliding_hash, "cs_zscore(open)").unwrap();
        assert_eq!(g1[[0, 0]], 10.0);
        assert_eq!(g2[[0, 0]], 20.0);
        assert!(cache.get(colliding_hash, "abs(volume)").is_none());
        assert_eq!(cache.stats().entries, 2);
    }

    #[test]
    fn test_sharded_concurrent_access_and_budget() {
        let cache = Arc::new(SubExprCache::new(128 * 16)); // 最多 16 个 128B 矩阵
        (0..200u64).into_par_iter().for_each(|i| {
            let key = format!("expr_{}", i % 40);
            if cache.get(i % 40, &key).is_none() {
                cache.insert(i % 40, key, mat_with(i as f64));
            }
        });
        let st = cache.stats();
        assert!(st.current_memory_bytes <= 128 * 16);
        assert!(st.entries <= 16);
    }
}
