use crate::bitpack::{BitPackError, BitPacker};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizeError {
    InvalidRepresentation(&'static str),
    BitPack(BitPackError),
    NumericalFailure,
}

impl std::fmt::Display for QuantizeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRepresentation(message) => {
                write!(formatter, "invalid quantized block: {message}")
            }
            Self::BitPack(error) => write!(formatter, "{error}"),
            Self::NumericalFailure => {
                formatter.write_str("dequantization produced a non-finite value")
            }
        }
    }
}

impl std::error::Error for QuantizeError {}

impl From<BitPackError> for QuantizeError {
    fn from(error: BitPackError) -> Self {
        Self::BitPack(error)
    }
}

/// Quantization parameters for a block of residual values.
#[derive(Debug, Clone)]
pub struct BlockParams {
    /// Scale factor: maps quantized values back to float.
    pub scale: f64,
    /// Zero point offset.
    pub zero_point: f64,
    /// Bits per value for this block.
    pub bits: u8,
}

/// A quantized block of residual values.
#[derive(Debug, Clone)]
pub struct QuantizedBlock {
    pub params: BlockParams,
    pub packed: BitPacker,
}

/// Uniform quantization of residual values.
///
/// Standard approach: map float range to integer grid.
/// Our advantage: because we operate on RESIDUALS (not raw weights),
/// the range is much smaller → same bits give better precision.
pub fn quantize_uniform(values: &[f64], bits: u8) -> QuantizedBlock {
    assert!(bits > 0 && bits <= 8);
    assert!(!values.is_empty(), "cannot quantize an empty block");
    assert!(
        values.iter().all(|value| value.is_finite()),
        "quantization values must be finite"
    );
    let levels = (1u32 << bits) as f64;

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    let scale = if range < 1e-15 {
        1.0
    } else {
        range / (levels - 1.0)
    };
    let zero_point = min;

    let quantized: Vec<u32> = values
        .iter()
        .map(|&v| {
            let q = ((v - zero_point) / scale).round() as i64;
            q.max(0).min(levels as i64 - 1) as u32
        })
        .collect();

    QuantizedBlock {
        params: BlockParams {
            scale,
            zero_point,
            bits,
        },
        packed: BitPacker::pack(&quantized, bits),
    }
}

/// Dequantize a block back to floats.
pub fn dequantize(block: &QuantizedBlock) -> Result<Vec<f64>, QuantizeError> {
    if block.params.bits != block.packed.bits_per_value {
        return Err(QuantizeError::InvalidRepresentation(
            "bit width metadata mismatch",
        ));
    }
    if !block.params.scale.is_finite() || block.params.scale <= 0.0 {
        return Err(QuantizeError::InvalidRepresentation(
            "scale must be finite and positive",
        ));
    }
    if !block.params.zero_point.is_finite() {
        return Err(QuantizeError::InvalidRepresentation(
            "zero point must be finite",
        ));
    }
    let values = block.packed.unpack()?;
    let decoded: Vec<f64> = values
        .iter()
        .map(|&q| q as f64 * block.params.scale + block.params.zero_point)
        .collect();
    if decoded.iter().any(|value| !value.is_finite()) {
        return Err(QuantizeError::NumericalFailure);
    }
    Ok(decoded)
}

/// Adaptive quantization: choose bits per block based on residual magnitude.
///
/// This is the key innovation over uniform bit allocation:
/// - Blocks with tiny residuals → 1-2 bits (the prior captured everything)
/// - Blocks with moderate residuals → 3-4 bits
/// - Blocks with large residuals → 5-6 bits (prior missed this part)
///
/// Uses absolute error tolerance: if the residual range is already small
/// relative to `abs_tol`, fewer bits are needed.
///
/// `abs_tol` is the maximum acceptable quantization error per element.
/// For LLM weights, typical values: 0.001 to 0.01.
pub fn choose_bits(residual_block: &[f64], abs_tol: f64) -> u8 {
    assert!(
        !residual_block.is_empty(),
        "cannot choose bits for an empty block"
    );
    assert!(
        residual_block.iter().all(|value| value.is_finite()),
        "residual values must be finite"
    );
    assert!(
        abs_tol.is_finite() && abs_tol > 0.0,
        "absolute tolerance must be finite and positive"
    );
    let min = residual_block.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = residual_block
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    // If range is within tolerance, 1 bit suffices (just encodes above/below midpoint)
    if range < abs_tol * 2.0 {
        return 1;
    }

    // bits needed = ceil(log2(range / abs_tol))
    // This gives enough quantization levels to cover the range with abs_tol precision
    let bits_needed = (range / abs_tol).log2().ceil() as i32;

    bits_needed.clamp(1, 8) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_roundtrip_4bit() {
        let values: Vec<f64> = (0..64).map(|i| (i as f64 - 32.0) * 0.1).collect();
        let block = quantize_uniform(&values, 4);
        let recovered = dequantize(&block).unwrap();

        let mse: f64 = values
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / values.len() as f64;

        // 4-bit over [-3.2, 3.1] range: step = 6.3/15 ≈ 0.42, MSE ≈ step²/12 ≈ 0.015
        assert!(mse < 0.02, "4-bit uniform should have low MSE, got {mse}");
    }

    #[test]
    fn adaptive_bits_small_residual() {
        // Range = 0.0063, with abs_tol = 0.005 → range < 2*tol → 1 bit
        let values: Vec<f64> = (0..64).map(|i| (i as f64) * 0.0001).collect();
        let bits = choose_bits(&values, 0.005);
        assert!(bits <= 2, "tiny residual should need few bits, got {bits}");
    }

    #[test]
    fn adaptive_bits_large_residual() {
        // Range = 63, with abs_tol = 0.005 → log2(63/0.005) = log2(12600) ≈ 13.6 → 8 (clamped)
        let values: Vec<f64> = (0..64).map(|i| (i as f64 - 32.0) * 1.0).collect();
        let bits = choose_bits(&values, 0.005);
        assert!(
            bits >= 4,
            "large residual should need more bits, got {bits}"
        );
    }
}
