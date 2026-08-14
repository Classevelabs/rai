//! KV cache for autoregressive generation with recurrent pondering support.
//!
//! During normal generation, each token's K,V are computed once and stored at its position.
//! During pondering (recurrent iteration), the current position's K,V are overwritten
//! each iteration — only the final converged values persist for future tokens.
//!
//! Layout per layer: `[num_kv_heads, max_ctx, head_dim]` stored flat.
//!
//! Each layer tracks a `filled` watermark: the number of leading positions that
//! hold real data. Stores must not leave gaps, reads must stay below the
//! watermark, and speculative decoding truncates it when drafts are rejected.
//! This turns the silent-garbage failure modes (attending over never-written
//! positions, reading a stale entry from a previous request) into panics or
//! `Err`s at the call site.

use anyhow::{anyhow, ensure, Result};

/// Positions allocated up front, before any conversation has happened.
///
/// The window a model *declares* and the window a conversation *uses* are
/// different numbers, and eagerly allocating the first is what used to force
/// the second to be small: a 40,960-token window on a 28-layer model reserves
/// 2.6 GB before a single token is generated, which pushes a laptop into
/// paging and makes decode several times slower than the same model at a
/// short window. Reserving the ceiling and allocating in steps means the long
/// window costs nothing until it is actually reached.
const INITIAL_POSITIONS: usize = 1024;

/// KV cache for a single transformer layer.
///
/// Allocated for `capacity` positions and grown towards `max_ctx` on demand.
/// `capacity` is the stride of both buffers, so it changes on every growth and
/// every index is computed from it rather than from `max_ctx`.
pub struct LayerKVCache {
    /// Key cache: `[num_kv_heads * capacity * head_dim]`
    k: Vec<f32>,
    /// Value cache: `[num_kv_heads * capacity * head_dim]`
    v: Vec<f32>,
    num_kv_heads: usize,
    /// Positions currently allocated. Also the row stride.
    capacity: usize,
    /// Ceiling this cache may grow to.
    max_ctx: usize,
    head_dim: usize,
    /// Number of leading positions that contain stored data.
    filled: usize,
}

impl LayerKVCache {
    /// Allocate a layer cache. Fails gracefully (instead of aborting the
    /// process) when the requested context would exceed available memory —
    /// a legitimately valid model file can still describe a cache far larger
    /// than the machine.
    pub fn new(num_kv_heads: usize, max_ctx: usize, head_dim: usize) -> Result<Self> {
        ensure!(
            num_kv_heads > 0 && max_ctx > 0 && head_dim > 0,
            "KV cache dimensions must be non-zero"
        );
        // The ceiling is validated even though it is not allocated: a model
        // declaring a window whose indices overflow is malformed regardless of
        // how much of it a conversation reaches.
        num_kv_heads
            .checked_mul(max_ctx)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| anyhow!("KV cache dimensions overflow"))?;
        let capacity = max_ctx.min(INITIAL_POSITIONS);
        let (k, v) = Self::allocate(num_kv_heads, capacity, head_dim)?;
        Ok(Self {
            k,
            v,
            num_kv_heads,
            capacity,
            max_ctx,
            head_dim,
            filled: 0,
        })
    }

    /// Two zeroed buffers for `positions` positions, or an error naming the size.
    fn allocate(
        num_kv_heads: usize,
        positions: usize,
        head_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let total = num_kv_heads
            .checked_mul(positions)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| anyhow!("KV cache dimensions overflow"))?;
        let mib = (total * 2 * std::mem::size_of::<f32>()) >> 20;
        let mut k = Vec::new();
        let mut v = Vec::new();
        k.try_reserve_exact(total)
            .and_then(|()| v.try_reserve_exact(total))
            .map_err(|_| anyhow!("cannot allocate {mib} MiB for the KV cache"))?;
        k.resize(total, 0.0);
        v.resize(total, 0.0);
        Ok((k, v))
    }

    /// Grow so that `pos` is addressable, doubling but never past `max_ctx`.
    ///
    /// Growth re-strides both buffers, so the filled positions of every head
    /// are copied to their new offsets. It is O(filled) and happens at most
    /// log2(max_ctx / INITIAL_POSITIONS) times per conversation.
    fn grow_to_hold(&mut self, pos: usize) -> Result<()> {
        if pos < self.capacity {
            return Ok(());
        }
        let target = self
            .capacity
            .saturating_mul(2)
            .max(pos + 1)
            .min(self.max_ctx);
        let (mut k, mut v) = Self::allocate(self.num_kv_heads, target, self.head_dim)?;
        for head in 0..self.num_kv_heads {
            let old = (head * self.capacity) * self.head_dim;
            let new = (head * target) * self.head_dim;
            let len = self.filled * self.head_dim;
            k[new..new + len].copy_from_slice(&self.k[old..old + len]);
            v[new..new + len].copy_from_slice(&self.v[old..old + len]);
        }
        self.k = k;
        self.v = v;
        self.capacity = target;
        Ok(())
    }

    /// Positions currently allocated, which is not the declared window.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Store K and V vectors at the given position.
    ///
    /// `k_vec` is `[num_kv_heads * head_dim]`, `v_vec` is `[num_kv_heads * head_dim]`.
    /// Overwriting an already-filled position is intentional (pondering
    /// iterations, speculative re-verification).
    ///
    /// # Panics
    /// Panics if `pos` is out of range, if storing at `pos` would leave a gap
    /// of unwritten positions below it, or on a vector length mismatch.
    pub fn store(&mut self, pos: usize, k_vec: &[f32], v_vec: &[f32]) {
        assert!(pos < self.max_ctx, "KV cache position is out of range");
        // Growth is fallible, but `store` is on the hot path and every caller
        // has already had the window checked against memory at load time. A
        // failure here means the machine lost memory mid-conversation, which
        // is not a recoverable state for a half-written layer.
        self.grow_to_hold(pos)
            .expect("KV cache growth failed mid-generation");
        assert!(
            pos <= self.filled,
            "KV store at position {pos} would leave a gap (filled {})",
            self.filled
        );
        let expected = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .expect("KV cache vector dimensions overflow");
        assert_eq!(k_vec.len(), expected, "KV key vector length mismatch");
        assert_eq!(v_vec.len(), expected, "KV value vector length mismatch");

        for h in 0..self.num_kv_heads {
            let src_start = h * self.head_dim;
            let dst_start = (h * self.capacity + pos) * self.head_dim;
            self.k[dst_start..dst_start + self.head_dim]
                .copy_from_slice(&k_vec[src_start..src_start + self.head_dim]);
            self.v[dst_start..dst_start + self.head_dim]
                .copy_from_slice(&v_vec[src_start..src_start + self.head_dim]);
        }
        self.filled = self.filled.max(pos + 1);
    }

    /// Get the cached K vector for a specific KV head at a specific position.
    ///
    /// # Panics
    /// Panics if `head` is out of range or `pos` is not a filled position.
    #[inline]
    pub fn get_k(&self, head: usize, pos: usize) -> &[f32] {
        assert!(head < self.num_kv_heads, "KV cache head is out of range");
        assert!(
            pos < self.filled,
            "KV read at unwritten position {pos} (filled {})",
            self.filled
        );
        let start = (head * self.capacity + pos) * self.head_dim;
        &self.k[start..start + self.head_dim]
    }

    /// Get the cached V vector for a specific KV head at a specific position.
    ///
    /// # Panics
    /// Panics if `head` is out of range or `pos` is not a filled position.
    #[inline]
    pub fn get_v(&self, head: usize, pos: usize) -> &[f32] {
        assert!(head < self.num_kv_heads, "KV cache head is out of range");
        assert!(
            pos < self.filled,
            "KV read at unwritten position {pos} (filled {})",
            self.filled
        );
        let start = (head * self.capacity + pos) * self.head_dim;
        &self.v[start..start + self.head_dim]
    }

    /// Number of leading positions that contain stored data.
    #[inline]
    pub fn filled(&self) -> usize {
        self.filled
    }

    /// Discard everything at and beyond `len` positions (no-op if already shorter).
    /// Used by speculative decoding to drop rejected draft entries.
    pub fn truncate(&mut self, len: usize) {
        self.filled = self.filled.min(len);
    }

    /// Reset the cache (for new generation).
    pub fn clear(&mut self) {
        self.k.iter_mut().for_each(|v| *v = 0.0);
        self.v.iter_mut().for_each(|v| *v = 0.0);
        self.filled = 0;
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * std::mem::size_of::<f32>()
    }
}

/// Full KV cache across all layers.
pub struct KVCache {
    layers: Vec<LayerKVCache>,
}

impl KVCache {
    /// Allocate cache for all layers.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        max_ctx: usize,
        head_dim: usize,
    ) -> Result<Self> {
        ensure!(num_layers > 0, "KV cache must contain at least one layer");
        let layers = (0..num_layers)
            .map(|_| LayerKVCache::new(num_kv_heads, max_ctx, head_dim))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    /// Store K,V at a position for a specific layer.
    ///
    /// # Panics
    /// Panics if `layer` is out of range, plus every [`LayerKVCache::store`]
    /// condition (position out of range, gap below the watermark, vector
    /// length mismatch).
    pub fn store(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        self.layers[layer].store(pos, k, v);
    }

    /// Get cached K for a head at a position in a layer.
    ///
    /// # Panics
    /// Panics if `layer` is out of range, `head_dim` does not match the
    /// cache, or the [`LayerKVCache::get_k`] conditions fail (head out of
    /// range, unfilled position).
    pub fn get_k(&self, layer: usize, head: usize, pos: usize, head_dim: usize) -> &[f32] {
        assert_eq!(
            self.layers[layer].head_dim, head_dim,
            "KV cache head_dim mismatch"
        );
        self.layers[layer].get_k(head, pos)
    }

    /// Get cached V for a head at a position in a layer.
    ///
    /// # Panics
    /// Panics if `layer` is out of range, `head_dim` does not match the
    /// cache, or the [`LayerKVCache::get_v`] conditions fail (head out of
    /// range, unfilled position).
    pub fn get_v(&self, layer: usize, head: usize, pos: usize, head_dim: usize) -> &[f32] {
        assert_eq!(
            self.layers[layer].head_dim, head_dim,
            "KV cache head_dim mismatch"
        );
        self.layers[layer].get_v(head, pos)
    }

    /// Number of leading positions with stored data in a layer.
    ///
    /// # Panics
    /// Panics if `layer` is out of range.
    pub fn filled(&self, layer: usize) -> usize {
        self.layers[layer].filled()
    }

    /// Truncate every layer to at most `len` filled positions.
    /// Speculative decoders call this after rejection so stale draft entries
    /// stop counting as valid context.
    pub fn truncate(&mut self, len: usize) {
        for layer in &mut self.layers {
            layer.truncate(len);
        }
    }

    /// Validate the dimensions used by an attention call before entering SIMD code.
    pub fn supports_attention(
        &self,
        layer: usize,
        num_kv_heads: usize,
        pos: usize,
        head_dim: usize,
    ) -> bool {
        self.layers.get(layer).is_some_and(|cache| {
            cache.num_kv_heads == num_kv_heads && pos < cache.max_ctx && cache.head_dim == head_dim
        })
    }

    /// Clear all layers.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Total memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of growing: a long declared window must not cost
    /// anything until a conversation actually reaches it.
    #[test]
    fn a_long_window_allocates_little_until_it_is_used() {
        let heads = 8;
        let head_dim = 128;
        let mut cache = LayerKVCache::new(heads, 40_960, head_dim).expect("allocate");
        let ceiling = heads * 40_960 * head_dim * 2 * std::mem::size_of::<f32>();
        assert!(
            cache.memory_bytes() * 8 < ceiling,
            "a 40,960 window should not allocate its ceiling up front: {} vs {ceiling}",
            cache.memory_bytes()
        );
        assert_eq!(cache.capacity(), INITIAL_POSITIONS);

        // Writing past the allocation grows it, and everything already stored
        // survives the re-stride. That is the part a wrong stride would break.
        let k: Vec<f32> = (0..heads * head_dim).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..heads * head_dim).map(|i| -(i as f32)).collect();
        for pos in 0..INITIAL_POSITIONS + 5 {
            cache.store(pos, &k, &v);
        }
        assert!(cache.capacity() > INITIAL_POSITIONS, "it must have grown");
        assert_eq!(cache.filled(), INITIAL_POSITIONS + 5);
        for head in 0..heads {
            let expected = &k[head * head_dim..(head + 1) * head_dim];
            assert_eq!(cache.get_k(head, 0), expected, "head {head} position 0");
            assert_eq!(
                cache.get_k(head, INITIAL_POSITIONS + 4),
                expected,
                "head {head} at the grown end"
            );
        }
    }

    /// Growth stops at the declared ceiling; it never quietly exceeds it.
    #[test]
    fn growth_never_passes_the_declared_window() {
        let mut cache = LayerKVCache::new(2, 4, 2).expect("allocate");
        assert_eq!(
            cache.capacity(),
            4,
            "a window below the step allocates once"
        );
        let k = vec![1.0f32; 4];
        for pos in 0..4 {
            cache.store(pos, &k, &k);
        }
        assert_eq!(cache.capacity(), 4);
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut cache = KVCache::new(2, 3, 16, 4).unwrap(); // 2 layers, 3 kv heads, 16 ctx, dim 4

        // Store at layer 0, position 0
        let k = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let v = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        cache.store(0, 0, &k, &v);

        // Retrieve head 0
        let k0 = cache.get_k(0, 0, 0, 4);
        assert_eq!(k0, &[1.0, 2.0, 3.0, 4.0]);

        // Retrieve head 2
        let k2 = cache.get_k(0, 2, 0, 4);
        assert_eq!(k2, &[9.0, 10.0, 11.0, 12.0]);

        let v1 = cache.get_v(0, 1, 0, 4);
        assert_eq!(v1, &[0.5, 0.6, 0.7, 0.8]);
    }

    #[test]
    fn test_overwrite_during_pondering() {
        let mut cache = KVCache::new(1, 1, 4, 2).unwrap();

        // First iteration: store [1.0, 2.0]
        cache.store(0, 0, &[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(cache.get_k(0, 0, 0, 2), &[1.0, 2.0]);

        // Pondering iteration: overwrite with [5.0, 6.0]
        cache.store(0, 0, &[5.0, 6.0], &[7.0, 8.0]);
        assert_eq!(cache.get_k(0, 0, 0, 2), &[5.0, 6.0]);
        assert_eq!(cache.get_v(0, 0, 0, 2), &[7.0, 8.0]);
    }

    #[test]
    fn test_memory_budget() {
        // SmolLM config: 30 layers, 3 KV heads, 512 ctx, dim 64
        let cache = KVCache::new(30, 3, 512, 64).unwrap();
        let bytes = cache.memory_bytes();
        let mb = bytes as f64 / (1024.0 * 1024.0);
        // Expected: 30 * 2 * 3 * 512 * 64 * 4 = 22,118,400 bytes ≈ 21.1 MB
        assert!(mb < 25.0, "KV cache too large: {mb:.1} MB");
        assert!(mb > 20.0, "KV cache too small: {mb:.1} MB");
    }

    #[test]
    fn test_filled_watermark_tracks_stores() {
        let mut cache = KVCache::new(1, 1, 8, 2).unwrap();
        assert_eq!(cache.filled(0), 0);
        cache.store(0, 0, &[1.0, 2.0], &[3.0, 4.0]);
        cache.store(0, 1, &[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(cache.filled(0), 2);
        // Overwrite below the watermark does not shrink it.
        cache.store(0, 0, &[9.0, 9.0], &[9.0, 9.0]);
        assert_eq!(cache.filled(0), 2);
    }

    #[test]
    #[should_panic(expected = "gap")]
    fn test_store_with_gap_panics() {
        let mut cache = KVCache::new(1, 1, 8, 2).unwrap();
        cache.store(0, 0, &[1.0, 2.0], &[3.0, 4.0]);
        cache.store(0, 5, &[1.0, 2.0], &[3.0, 4.0]); // positions 1..5 never written
    }

    #[test]
    #[should_panic(expected = "unwritten position")]
    fn test_read_beyond_watermark_panics() {
        let mut cache = KVCache::new(1, 1, 8, 2).unwrap();
        cache.store(0, 0, &[1.0, 2.0], &[3.0, 4.0]);
        let _ = cache.get_k(0, 0, 1, 2);
    }

    #[test]
    #[should_panic(expected = "head is out of range")]
    fn test_read_bad_head_panics() {
        let mut cache = KVCache::new(1, 2, 8, 2).unwrap();
        cache.store(0, 0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        let _ = cache.get_k(0, 2, 0, 2);
    }

    #[test]
    fn test_truncate_and_refill() {
        let mut cache = KVCache::new(1, 1, 8, 2).unwrap();
        for pos in 0..4 {
            cache.store(0, pos, &[pos as f32, 0.0], &[0.0, 0.0]);
        }
        // Reject drafts beyond position 1.
        cache.truncate(2);
        assert_eq!(cache.filled(0), 2);
        // Refill from the new frontier without a gap.
        cache.store(0, 2, &[7.0, 7.0], &[7.0, 7.0]);
        assert_eq!(cache.get_k(0, 0, 2, 2), &[7.0, 7.0]);
        assert_eq!(cache.filled(0), 3);
    }

    #[test]
    fn test_clear_resets_watermark() {
        let mut cache = KVCache::new(1, 1, 4, 2).unwrap();
        cache.store(0, 0, &[1.0, 2.0], &[3.0, 4.0]);
        cache.clear();
        assert_eq!(cache.filled(0), 0);
    }
}
