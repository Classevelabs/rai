/// Bit-level packing for variable-width integers.
///
/// Packs values into a byte stream using exactly `bits_per_value` bits each.
/// This is more efficient than standard byte-aligned storage:
/// - 2-bit values: 4x denser than u8
/// - 3-bit values: 2.67x denser than u8
#[derive(Debug, Clone)]
pub struct BitPacker {
    pub data: Vec<u8>,
    pub bits_per_value: u8,
    pub num_values: usize,
}

impl BitPacker {
    /// Pack a slice of unsigned integers using `bits` bits each.
    pub fn pack(values: &[u32], bits: u8) -> Self {
        assert!((1..=8).contains(&bits), "bits must be 1-8");
        let mask = (1u32 << bits) - 1;
        let total_bits = values.len() * bits as usize;
        let num_bytes = total_bits.div_ceil(8);
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
    pub fn unpack(&self) -> Vec<u32> {
        let bits = self.bits_per_value as usize;
        let mask = (1u32 << bits) - 1;
        let mut values = Vec::with_capacity(self.num_values);

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

        values
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
        (self.data.len() * 8) as f64 / self.num_values as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_2bit() {
        let values: Vec<u32> = vec![0, 1, 2, 3, 1, 0, 3, 2];
        let packed = BitPacker::pack(&values, 2);
        let unpacked = packed.unpack();
        assert_eq!(values, unpacked);
        assert_eq!(packed.size_bytes(), 2); // 8 values * 2 bits = 16 bits = 2 bytes
    }

    #[test]
    fn pack_unpack_3bit() {
        let values: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 3, 1];
        let packed = BitPacker::pack(&values, 3);
        let unpacked = packed.unpack();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_unpack_4bit() {
        let values: Vec<u32> = (0..16).collect();
        let packed = BitPacker::pack(&values, 4);
        let unpacked = packed.unpack();
        assert_eq!(values, unpacked);
        assert_eq!(packed.size_bytes(), 8); // 16 * 4 = 64 bits = 8 bytes
    }
}
