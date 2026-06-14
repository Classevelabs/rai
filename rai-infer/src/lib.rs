#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    reason = "Inference kernels use index-heavy loops and explicit argument lists to keep SIMD hot paths allocation-free and easy to inspect."
)]

pub mod chat_template;
pub mod format;
pub mod gemm;
pub mod kv_cache;
pub mod layers;
pub mod model;
pub mod ponder;
pub mod sampler;
pub mod self_speculative;
pub mod speculative;
