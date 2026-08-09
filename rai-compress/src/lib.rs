//! Weight-matrix quantization and compression research toolkit.
//!
//! Two families live here:
//!
//! - **RC / HRC / SAC** (`compress`, `hrc`, `sac`, with the `prior`,
//!   `channel`, `sparse`, `quantize`, and `bitpack` building blocks) are
//!   **experimental research compressors**. They operate on in-memory
//!   `f64` matrices, report *modeled* byte sizes, and do **not** produce or
//!   consume the `.raimodel` container — there is no serialization format
//!   behind them.
//! - **GPTQ** (`gptq`) implements calibration-based quantization following
//!   Frantar et al., "GPTQ: Accurate Post-Training Quantization for
//!   Generative Pre-trained Transformers" (2022), using the Cholesky factor
//!   of the inverse Hessian. It is an independent Rust implementation and is
//!   not part of the Python `.raimodel` export pipeline.
//!
//! All public entry points validate caller input and return `Result` with
//! the crate's error types; they do not panic on invalid input.
#![forbid(unsafe_code)]

pub mod bitpack;
pub mod channel;
pub mod compress;
pub mod gptq;
pub mod hrc;
pub mod prior;
pub mod quantize;
pub mod sac;
pub mod sparse;

pub use bitpack::{BitPackError, BitPacker};
pub use channel::{ChannelError, ChannelNorm};
pub use compress::{
    compare, compress, compress_uniform_4bit, decompress_matrix, ComparisonReport,
    CompressedMatrix, CompressionError, CompressionStats, RCConfig,
};
pub use gptq::{
    gptq_decompress, gptq_quantize, hessian_weighted_mse, GptqError, GptqResult, GptqStats,
    GroupParams,
};
pub use hrc::{
    full_compare, hrc_compress, hrc_decompress, FullReport, HRCConfig, HRCStats, HRCompressed,
};
pub use prior::{PriorError, WeightPrior};
pub use quantize::{
    choose_bits, dequantize, quantize_uniform, BlockParams, QuantizeError, QuantizedBlock,
};
pub use sac::{sac_compress, sac_decompress, SACCompressed, SACConfig, SACStats};
pub use sparse::{SparseDenseDecomp, SparseError};
