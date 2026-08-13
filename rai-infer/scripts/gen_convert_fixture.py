#!/usr/bin/env python3
"""Generate the golden fixture for the Rust converter's conformance test.

`rai-infer/tests/convert_matches_python.rs` builds a synthetic HuggingFace
checkpoint from the LCG below, runs `rai_infer::convert`, and requires the
output to be byte-identical to `tests/fixtures/tiny-convert.raimodel`.  This
script produces that file from the *same* weights using the reference
quantizer/writer (raimodel.py — the module export_rtn.py calls), so the test
compares Rust against Python rather than against itself.

numpy-only by design: no torch, no safetensors.  The weights are generated
directly here instead of being read back from the synthetic checkpoint, so a
bug in either side's safetensors handling shows up as differing bytes.

    python3 gen_convert_fixture.py

Every generated value is exactly representable in f16, so the checkpoint's f16
storage, the exporter's float32 view and this script's float64 maths all agree
bit for bit — the fixture pins the quantizer, not a float conversion.
"""

from pathlib import Path

import numpy as np

import raimodel

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "tiny-convert.raimodel"

MASK64 = (1 << 64) - 1
SEED = 0x20260813

# Mirrors TINY_CONFIG in test_raimodel.py (hidden 64, 2 layers, 4 heads,
# 2 kv heads, head_dim 16, intermediate 128, vocab 96), tied lm_head.
CONFIG = {
    "hidden_size": 64,
    "num_layers": 2,
    "num_heads": 4,
    "num_kv_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "vocab_size": 96,
    "max_context": 512,
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
    "bits": 4,
    "group_size": 64,
    "embed_bits": 8,
    "embed_group_size": 64,
}


class Lcg:
    """The same 64-bit LCG the Rust test uses (PCG-style multiplier)."""

    def __init__(self, seed):
        self.state = seed

    def next_u32(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & MASK64
        return (self.state >> 33) & 0xFFFFFFFF

    def weights(self, rows, cols):
        """Values in [-0.488, 0.488] on a 1/4096 grid: exact in f16."""
        flat = [(self.next_u32() % 4001 - 2000) / 4096.0 for _ in range(rows * cols)]
        return np.array(flat, dtype=np.float32).reshape(rows, cols)

    def norm(self, size):
        """Values near 1.0 on a 1/1024 grid: exact in f16."""
        flat = [1.0 + (self.next_u32() % 201 - 100) / 1024.0 for _ in range(size)]
        return np.array(flat, dtype=np.float32)


def main():
    rng = Lcg(SEED)
    hidden = CONFIG["hidden_size"]
    inter = CONFIG["intermediate_size"]
    vocab = CONFIG["vocab_size"]
    kv_dim = CONFIG["num_kv_heads"] * CONFIG["head_dim"]
    gs = CONFIG["group_size"]

    dims = [(hidden, hidden), (kv_dim, hidden), (kv_dim, hidden), (hidden, hidden),
            (inter, hidden), (inter, hidden), (hidden, inter)]

    sections = []

    embed = rng.weights(vocab, hidden)
    e_codes, e_scales, e_zeros, _ = raimodel.quantize_embedding_8bit(
        embed, group_size=CONFIG["embed_group_size"]
    )
    sections.append(raimodel.build_embedding_section(e_codes, e_scales, e_zeros))

    for layer in range(CONFIG["num_layers"]):
        packed = []
        for name, (rows, cols) in zip(raimodel.LAYER_LINEAR_NAMES, dims):
            weight = rng.weights(rows, cols)
            codes, scales, zeros, _ = raimodel.rtn_quantize(
                weight, bits=CONFIG["bits"], group_size=gs, label=f"L{layer}.{name}"
            )
            packed.append((codes, scales, zeros, rows, cols))
        input_ln = rng.norm(hidden)
        post_ln = rng.norm(hidden)
        sections.append(raimodel.build_layer_section(packed, input_ln, post_ln))

    sections.append(raimodel.pack_norm_section(rng.norm(hidden), "final norm"))

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    size = raimodel.write_raimodel(FIXTURE, CONFIG, sections)
    print(f"wrote {FIXTURE} ({size} bytes, {len(sections)} sections)")


if __name__ == "__main__":
    main()
