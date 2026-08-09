/// Bit-level packing for variable-width integers.
///
/// Packs values into a byte stream using exactly `bits_per_value` bits each.
/// This is more efficient than standard byte-aligned storage:
/// - 2-bit values: 4x denser than u8
/// - 3-bit values: 2.67x denser than u8
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitPackError {
    InvalidBitWidth,
    SizeOverflow,
    DataLengthMismatch,
    AllocationFailed,
}

impl std::fmt::Display for BitPackError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBitWidth => formatter.write_str("bits_per_value must be 1-8"),
            Self::SizeOverflow => formatter.write_str("packed dimensions overflow"),
            Self::DataLengthMismatch => formatter.write_str("packed data length mismatch"),
            Self::AllocationFailed => formatter.write_str("unable to allocate unpacked values"),
        }
    }
}

impl std::error::Error for BitPackError {}

#[derive(Debug, Clone)]
pub struct BitPacker {
    pub data: Vec<u8>,
    pub bits_per_value: u8,
    pub num_values: usize,
}

impl BitPacker {
    /// Pack a slice of unsigned integers using `bits` bits each.
    pub fn pack(values: &[u32], bits: u8) -> Self {
        assert!(bits > 0 && bits <= 8, "bits must be 1-8");
        let mask = (1u32 << bits) - 1;
        assert!(
            values.iter().all(|&value| value <= mask),
            "value does not fit the requested bit width"
        );
        let total_bits = values
            .len()
            .checked_mul(bits as usize)
            .expect("packed bit length overflows");
        let num_bytes = total_bits
            .checked_add(7)
            .expect("packed byte length overflows")
            / 8;
        let mut data = vec![0u8; num_bytes];

        for (i, &val) in values.iter().enumerate() {
            let v = val & mask;
            let bit_offset = i * bits as usize;
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;

            // Write across byte boundary if needed
            data[byte_idx] |= (v as u8) << bit_idx;
            if bit_idx + bits as usize > 8 && byte_idx + 1 < num_bytes {
                data[byte_idx + 1] |= (v >> (8 - bit_idx)) as u8;
            }
        }

        Self {
            data,
            bits_per_value: bits,
            num_values: values.len(),
        }
    }

    /// Unpack all values.
    pub fn unpack(&self) -> Result<Vec<u32>, BitPackError> {
        let bits = self.bits_per_value as usize;
        if !(1..=8).contains(&bits) {
            return Err(BitPackError::InvalidBitWidth);
        }
        let expected_bytes = self
            .num_values
            .checked_mul(bits)
            .and_then(|value| value.checked_add(7))
            .ok_or(BitPackError::SizeOverflow)?
            / 8;
        if self.data.len() != expected_bytes {
            return Err(BitPackError::DataLengthMismatch);
        }
        let mask = (1u32 << bits) - 1;
        let mut values = Vec::new();
        values
            .try_reserve_exact(self.num_values)
            .map_err(|_| BitPackError::AllocationFailed)?;

        for i in 0..self.num_values {
            let bit_offset = i * bits;
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;

            let mut val = (self.data[byte_idx] >> bit_idx) as u32;
            if bit_idx + bits > 8 && byte_idx + 1 < self.data.len() {
                val |= (self.data[byte_idx + 1] as u32) << (8 - bit_idx);
            }
            values.push(val & mask);
        }

        Ok(values)
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Effective bits per value.
    pub fn effective_bpv(&self) -> f64 {
        if self.num_values == 0 {
            return 0.0;
        }
        self.data.len() as f64 * 8.0 / self.num_values as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_2bit() {
        let values: Vec<u32> = vec![0, 1, 2, 3, 1, 0, 3, 2];
        let packed = BitPacker::pack(&values, 2);
        let unpacked = packed.unpack().unwrap();
        assert_eq!(values, unpacked);
        assert_eq!(packed.size_bytes(), 2); // 8 values * 2 bits = 16 bits = 2 bytes
    }

    #[test]
    fn pack_unpack_3bit() {
        let values: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 3, 1];
        let packed = BitPacker::pack(&values, 3);
        let unpacked = packed.unpack().unwrap();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_unpack_4bit() {
        let values: Vec<u32> = (0..16).collect();
        let packed = BitPacker::pack(&values, 4);
        let unpacked = packed.unpack().unwrap();
        assert_eq!(values, unpacked);
        assert_eq!(packed.size_bytes(), 8); // 16 * 4 = 64 bits = 8 bytes
    }

    #[test]
    fn malformed_public_representation_is_rejected() {
        assert_eq!(
            BitPacker {
                data: vec![],
                bits_per_value: 4,
                num_values: 1,
            }
            .unpack(),
            Err(BitPackError::DataLengthMismatch)
        );
    }
}
