//! KV cache for autoregressive generation with recurrent pondering support.
//!
//! During normal generation, each token's K,V are computed once and stored at its position.
//! During pondering (recurrent iteration), the current position's K,V are overwritten
//! each iteration — only the final converged values persist for future tokens.
//!
//! Layout per layer: `[num_kv_heads, max_ctx, head_dim]` stored flat.

/// KV cache for a single transformer layer.
pub struct LayerKVCache {
    /// Key cache: `[num_kv_heads * max_ctx * head_dim]`
    k: Vec<f32>,
    /// Value cache: `[num_kv_heads * max_ctx * head_dim]`
    v: Vec<f32>,
    num_kv_heads: usize,
    max_ctx: usize,
    head_dim: usize,
}

impl LayerKVCache {
    pub fn new(num_kv_heads: usize, max_ctx: usize, head_dim: usize) -> Self {
        assert!(
            num_kv_heads > 0 && max_ctx > 0 && head_dim > 0,
            "KV cache dimensions must be non-zero"
        );
        let total = num_kv_heads
            .checked_mul(max_ctx)
            .and_then(|value| value.checked_mul(head_dim))
            .expect("KV cache dimensions overflow");
        Self {
            k: vec![0.0; total],
            v: vec![0.0; total],
            num_kv_heads,
            max_ctx,
            head_dim,
        }
    }

    /// Store K and V vectors at the given position.
    ///
    /// `k_vec` is `[num_kv_heads * head_dim]`, `v_vec` is `[num_kv_heads * head_dim]`.
    /// Overwrites any existing data at `pos` — this is intentional for pondering iterations.
    pub fn store(&mut self, pos: usize, k_vec: &[f32], v_vec: &[f32]) {
        assert!(pos < self.max_ctx, "KV cache position is out of range");
        let expected = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .expect("KV cache vector dimensions overflow");
        assert_eq!(k_vec.len(), expected, "KV key vector length mismatch");
        assert_eq!(v_vec.len(), expected, "KV value vector length mismatch");

        for h in 0..self.num_kv_heads {
            let src_start = h * self.head_dim;
            let dst_start = (h * self.max_ctx + pos) * self.head_dim;
            self.k[dst_start..dst_start + self.head_dim]
                .copy_from_slice(&k_vec[src_start..src_start + self.head_dim]);
            self.v[dst_start..dst_start + self.head_dim]
                .copy_from_slice(&v_vec[src_start..src_start + self.head_dim]);
        }
    }

    /// Get the cached K vector for a specific KV head at a specific position.
    #[inline]
    pub fn get_k(&self, head: usize, pos: usize) -> &[f32] {
        let start = (head * self.max_ctx + pos) * self.head_dim;
        &self.k[start..start + self.head_dim]
    }

    /// Get the cached V vector for a specific KV head at a specific position.
    #[inline]
    pub fn get_v(&self, head: usize, pos: usize) -> &[f32] {
        let start = (head * self.max_ctx + pos) * self.head_dim;
        &self.v[start..start + self.head_dim]
    }

    /// Reset the cache (for new generation).
    pub fn clear(&mut self) {
        self.k.iter_mut().for_each(|v| *v = 0.0);
        self.v.iter_mut().for_each(|v| *v = 0.0);
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
    pub fn new(num_layers: usize, num_kv_heads: usize, max_ctx: usize, head_dim: usize) -> Self {
        assert!(num_layers > 0, "KV cache must contain at least one layer");
        let layers = (0..num_layers)
            .map(|_| LayerKVCache::new(num_kv_heads, max_ctx, head_dim))
            .collect();
        Self { layers }
    }

    /// Store K,V at a position for a specific layer.
    pub fn store(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        self.layers[layer].store(pos, k, v);
    }

    /// Get cached K for a head at a position in a layer.
    pub fn get_k(&self, layer: usize, head: usize, pos: usize, _head_dim: usize) -> &[f32] {
        assert_eq!(
            self.layers[layer].head_dim, _head_dim,
            "KV cache head_dim mismatch"
        );
        self.layers[layer].get_k(head, pos)
    }

    /// Get cached V for a head at a position in a layer.
    pub fn get_v(&self, layer: usize, head: usize, pos: usize, _head_dim: usize) -> &[f32] {
        assert_eq!(
            self.layers[layer].head_dim, _head_dim,
            "KV cache head_dim mismatch"
        );
        self.layers[layer].get_v(head, pos)
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

    #[test]
    fn test_store_and_retrieve() {
        let mut cache = KVCache::new(2, 3, 16, 4); // 2 layers, 3 kv heads, 16 ctx, dim 4

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
        let mut cache = KVCache::new(1, 1, 4, 2);

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
        let cache = KVCache::new(30, 3, 512, 64);
        let bytes = cache.memory_bytes();
        let mb = bytes as f64 / (1024.0 * 1024.0);
        // Expected: 30 * 2 * 3 * 512 * 64 * 4 = 22,118,400 bytes ≈ 21.1 MB
        assert!(mb < 25.0, "KV cache too large: {mb:.1} MB");
        assert!(mb > 20.0, "KV cache too small: {mb:.1} MB");
    }
}
