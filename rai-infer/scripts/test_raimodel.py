#!/usr/bin/env python3
"""Tests for raimodel.py — numpy-only, no pytest required.

Run with:
    python test_raimodel.py

Also (re)generates the golden fixtures used by the Rust conformance tests:
    rai-infer/tests/fixtures/tiny-tied.raimodel     (+ .expected.json)
    rai-infer/tests/fixtures/tiny-untied.raimodel   (+ .expected.json)
"""

import argparse
import io
import json
import struct
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

import raimodel

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# Configuration of the tiny golden-fixture model.
TINY_CONFIG = {
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


# =============================================================================
# Helpers
# =============================================================================

class _FakeParser:
    """argparse stand-in whose error() raises instead of exiting."""

    def error(self, message):
        raise ValueError(message)


def _unpack_nibbles(packed, cols):
    rows = packed.shape[0]
    out = np.empty((rows, cols), dtype=np.uint8)
    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = (packed >> 4) & 0x0F
    return out


def _correlated_hessian(cols, rng, rank=16, noise=0.05, samples=None):
    """H = X^T X / n for X with strongly correlated (near-low-rank) columns."""
    n = samples if samples is not None else 2 * cols
    base = rng.standard_normal((n, rank))
    mix = rng.standard_normal((rank, cols))
    X = base @ mix + noise * rng.standard_normal((n, cols))
    return X.T @ X / n


def _reference_gptq(weight, hessian, bits, block_size, group_size):
    """Straightforward GPTQ reference that records, for every column, the
    group params that were in effect when that column's codes were derived.

    Mirrors the exact numpy operations of raimodel.gptq_quantize so outputs
    are bit-identical when the shared implementation is correct.
    """
    W = np.asarray(weight, dtype=np.float64).copy()
    rows, cols = W.shape
    n_levels = 2 ** bits
    num_groups = -(-cols // group_size)

    H = np.asarray(hessian, dtype=np.float64).copy()
    damp = 0.01 * max(float(np.mean(np.diag(H))), 1e-6)
    H[np.diag_indices(cols)] += damp
    U = raimodel._cholesky_inverse_upper(H)

    codes = np.zeros((rows, cols), dtype=np.uint8)
    scales_f16 = np.zeros((rows, num_groups), dtype=np.float16)
    zeros_f16 = np.zeros((rows, num_groups), dtype=np.float16)
    used_scales = np.zeros((rows, cols), dtype=np.float16)
    used_zeros = np.zeros((rows, cols), dtype=np.float16)

    cur_gid = -1
    scale_f16 = zero_f16 = None
    group_scale = group_zero = None
    for block_start in range(0, cols, block_size):
        block_end = min(block_start + block_size, cols)
        err_block = np.zeros((rows, block_end - block_start), dtype=np.float64)
        for j in range(block_start, block_end):
            gid = j // group_size
            if gid != cur_gid:
                cur_gid = gid
                g_start = gid * group_size
                g_end = min(g_start + group_size, cols)
                scale_f16, zero_f16 = raimodel.compute_group_params(
                    W[:, g_start:g_end], n_levels, "reference group"
                )
                scales_f16[:, gid] = scale_f16
                zeros_f16[:, gid] = zero_f16
                group_scale = scale_f16.astype(np.float64)
                group_zero = zero_f16.astype(np.float64)
            used_scales[:, j] = scale_f16
            used_zeros[:, j] = zero_f16

            w_col = W[:, j]
            q_col = np.clip(
                np.round((w_col - group_zero) / group_scale), 0, n_levels - 1
            ).astype(np.uint8)
            codes[:, j] = q_col
            w_hat = q_col.astype(np.float64) * group_scale + group_zero
            err = (w_col - w_hat) / U[j, j]
            err_block[:, j - block_start] = err
            if j + 1 < block_end:
                W[:, j + 1:block_end] -= np.outer(err, U[j, j + 1:block_end])
        if block_end < cols:
            W[:, block_end:] -= err_block @ U[block_start:block_end, block_end:]

    return codes, scales_f16, zeros_f16, used_scales, used_zeros


def _dequant_scalar_f32(code, scale_f16, zero_f16):
    """One value dequantized exactly the way the Rust reader computes it."""
    return float(np.float32(code) * np.float32(scale_f16) + np.float32(zero_f16))


# =============================================================================
# Packing / layout tests
# =============================================================================

def test_pack_nibbles_roundtrip():
    rng = np.random.default_rng(11)
    codes = rng.integers(0, 16, size=(5, 12), dtype=np.uint8)
    packed = raimodel.pack_nibbles(codes)
    assert packed.shape == (5, 6), packed.shape
    assert packed.dtype == np.uint8
    # low nibble = even column, high nibble = odd column
    restored = _unpack_nibbles(packed, 12)
    assert np.array_equal(restored, codes)
    # hand-checked layout for one byte
    two = np.array([[0x3, 0xA]], dtype=np.uint8)
    assert raimodel.pack_nibbles(two)[0, 0] == 0xA3


def test_pack_nibbles_rejects_odd_columns():
    codes = np.zeros((3, 7), dtype=np.uint8)
    try:
        raimodel.pack_nibbles(codes)
    except ValueError as exc:
        assert "even" in str(exc)
    else:
        raise AssertionError("pack_nibbles accepted an odd column count")


def test_group_param_byte_layout():
    # Values chosen to be exactly representable in f16.
    scales = np.array([[1.5, 2.5], [0.375, 4.0]], dtype=np.float16)
    zeros = np.array([[-3.0, 0.25], [7.0, -0.5]], dtype=np.float16)
    packed = raimodel.pack_group_params(scales, zeros)
    rows, num_groups = scales.shape
    assert len(packed) == rows * num_groups * 4
    for r in range(rows):
        for g in range(num_groups):
            off = (r * num_groups + g) * 4
            # f16 scale at +0 (little-endian), f16 zero at +2
            assert packed[off:off + 2] == scales[r, g].astype("<f2").tobytes()
            assert packed[off + 2:off + 4] == zeros[r, g].astype("<f2").tobytes()
    # spot-check raw bytes: f16 1.5 == 0x3E00 little-endian
    assert packed[0:2] == b"\x00\x3e"


def test_header_field_offsets():
    config = dict(TINY_CONFIG)
    buf = io.BytesIO()
    raimodel.write_header(buf, config, 4)
    header = buf.getvalue()
    assert len(header) == 64
    assert header[0:4] == b"RAIM"
    assert struct.unpack_from("<I", header, 4)[0] == 1
    for offset, name in (
        (8, "hidden_size"), (12, "num_layers"), (16, "num_heads"),
        (20, "num_kv_heads"), (24, "head_dim"), (28, "intermediate_size"),
        (32, "vocab_size"), (36, "max_context"),
    ):
        assert struct.unpack_from("<I", header, offset)[0] == config[name], name
    assert struct.unpack_from("<f", header, 40)[0] == np.float32(config["rope_theta"])
    assert struct.unpack_from("<f", header, 44)[0] == np.float32(config["norm_eps"])
    assert header[48] == config["bits"]
    assert header[49] == config["group_size"]
    assert header[50] == config["embed_bits"]
    assert header[51] == config["embed_group_size"]
    assert struct.unpack_from("<I", header, 52)[0] == 4
    assert header[56:64] == b"\x00" * 8  # reserved


# =============================================================================
# Quantizer tests
# =============================================================================

def test_bug1_group_params_never_recomputed_mid_group():
    """Regression for BUG-1: with group_size=100 and block_size=128, group
    boundaries do not line up with block boundaries.  The buggy writer
    recomputed group params at every block start, overwriting the stored
    scale/zero AFTER codes had been derived against the originals."""
    rng = np.random.default_rng(42)
    rows, cols = 32, 256
    W = rng.standard_normal((rows, cols))
    H = _correlated_hessian(cols, rng)

    codes, scales, zeros, mse_100 = raimodel.gptq_quantize(
        W, H, bits=4, block_size=128, group_size=100, label="bug1"
    )
    ref_codes, ref_scales, ref_zeros, used_scales, used_zeros = _reference_gptq(
        W, H, bits=4, block_size=128, group_size=100
    )

    # The shared implementation must match the param-recording reference
    # bit for bit.
    assert np.array_equal(codes, ref_codes), "codes diverge from reference"
    assert np.array_equal(scales, ref_scales), "scales diverge from reference"
    assert np.array_equal(zeros, ref_zeros), "zeros diverge from reference"

    # Stored group params must equal the params the codes were computed
    # against, for every column of every group.
    for j in range(cols):
        gid = j // 100
        assert np.array_equal(used_scales[:, j], scales[:, gid]), \
            f"column {j}: stored scale differs from the scale used to encode it"
        assert np.array_equal(used_zeros[:, j], zeros[:, gid]), \
            f"column {j}: stored zero differs from the zero used to encode it"

    # Reconstruction with group_size=100 must stay within the same error
    # bound as the aligned group_size=128 run (the bug made it explode).
    _, _, _, mse_128 = raimodel.gptq_quantize(
        W, H, bits=4, block_size=128, group_size=128, label="bug1-aligned"
    )
    assert mse_100 <= mse_128 * 1.5, (
        f"group_size=100 mse {mse_100:.3e} vs group_size=128 mse {mse_128:.3e}"
    )

    # Every dequantized value must lie inside its group's representable
    # range [zero, zero + 15 * scale] — a corrupted scale/zero pairing puts
    # values outside it.
    deq = raimodel.dequantize(codes, scales, zeros, 100, np.float64)
    for gid in range((cols + 99) // 100):
        c0, c1 = gid * 100, min(gid * 100 + 100, cols)
        lo = zeros[:, gid].astype(np.float64)[:, None]
        hi = lo + 15.0 * scales[:, gid].astype(np.float64)[:, None]
        block = deq[:, c0:c1]
        assert (block >= lo - 1e-6).all() and (block <= hi + 1e-6).all()


def test_gptq_beats_rtn_on_correlated_hessian():
    """GPTQ with the Cholesky factor of H^-1 must beat plain round-to-nearest
    on Hessian-weighted MSE (the objective GPTQ minimizes)."""
    rng = np.random.default_rng(1234)
    rows, cols, gsize = 64, 128, 64
    W = rng.standard_normal((rows, cols))
    H = _correlated_hessian(cols, rng, rank=16, noise=0.05, samples=256)

    q_codes, q_s, q_z, _ = raimodel.gptq_quantize(W, H, group_size=gsize, label="gptq")
    r_codes, r_s, r_z, _ = raimodel.rtn_quantize(W, group_size=gsize, label="rtn")

    W_gptq = raimodel.dequantize(q_codes, q_s, q_z, gsize, np.float64)
    W_rtn = raimodel.dequantize(r_codes, r_s, r_z, gsize, np.float64)

    def hessian_weighted_mse(W_hat):
        D = W - W_hat
        return float(np.sum((D @ H) * D)) / W.size

    e_gptq = hessian_weighted_mse(W_gptq)
    e_rtn = hessian_weighted_mse(W_rtn)
    assert e_gptq < e_rtn, f"GPTQ {e_gptq:.4e} not better than RTN {e_rtn:.4e}"
    assert e_gptq < 0.9 * e_rtn, (
        f"GPTQ improvement too small: {e_gptq:.4e} vs RTN {e_rtn:.4e}"
    )


def test_zero_hessian_uses_damping_floor():
    """Regression for BUG-3: an all-zero Hessian used to make damp exactly 0,
    both Cholesky attempts raised, and the export died uncaught.  With the
    floor, H becomes a tiny multiple of I, and GPTQ with an isotropic Hessian
    degrades exactly to RTN (no cross-column error propagation)."""
    rng = np.random.default_rng(7)
    W = rng.standard_normal((16, 64))
    H = np.zeros((64, 64))
    codes, scales, zeros, mse = raimodel.gptq_quantize(W, H, group_size=32, label="zero-H")
    r_codes, r_scales, r_zeros, r_mse = raimodel.rtn_quantize(W, group_size=32)
    assert np.array_equal(codes, r_codes)
    assert np.array_equal(scales, r_scales)
    assert np.array_equal(zeros, r_zeros)
    assert mse == r_mse


def test_bad_hessian_raises_named_error():
    """A NaN Hessian must fail with a clear RuntimeError naming the tensor
    (either via the double-damping path or the U[j,j] guard), never by
    writing arbitrary codes."""
    rng = np.random.default_rng(8)
    W = rng.standard_normal((8, 32))
    H = np.full((32, 32), np.nan)
    try:
        raimodel.gptq_quantize(W, H, group_size=16, label="L3.down_proj")
    except RuntimeError as exc:
        assert "L3.down_proj" in str(exc)
    else:
        raise AssertionError("gptq_quantize accepted a NaN Hessian")


def test_rtn_roundtrip_error_bound():
    rng = np.random.default_rng(21)
    W = rng.standard_normal((24, 128)) * 0.2
    codes, scales, zeros, mse = raimodel.rtn_quantize(W, bits=4, group_size=64)
    deq = raimodel.dequantize(codes, scales, zeros, 64, np.float64)
    # Each value is within half a quantization step of the original
    # (plus f16 rounding slack on the params).
    step = np.repeat(scales.astype(np.float64), 64, axis=1)[:, :128]
    assert (np.abs(W - deq) <= step * 0.5 + 1e-3).all()
    assert mse < 1e-3


def test_embedding_8bit_roundtrip():
    rng = np.random.default_rng(22)
    W = rng.standard_normal((96, 64)) * 0.05
    codes, scales, zeros, mse = raimodel.quantize_embedding_8bit(W, group_size=64)
    assert codes.dtype == np.uint8
    assert codes.max() <= 255
    deq = raimodel.dequantize(codes, scales, zeros, 64, np.float64)
    assert np.abs(W - deq).max() < 1e-2
    assert mse < 1e-6


# =============================================================================
# Validation tests
# =============================================================================

def _valid_big_config():
    return {
        "hidden_size": 4096, "num_layers": 32, "num_heads": 32,
        "num_kv_heads": 8, "head_dim": 128, "intermediate_size": 14336,
        "vocab_size": 32768, "max_context": 2048, "rope_theta": 1e6,
        "norm_eps": 1e-5, "bits": 4, "group_size": 128,
        "embed_bits": 8, "embed_group_size": 64,
    }


def _expect_value_error(config, needle):
    try:
        raimodel.validate_model_config(config)
    except ValueError as exc:
        assert needle in str(exc), f"expected {needle!r} in {exc}"
    else:
        raise AssertionError(f"config accepted; expected error mentioning {needle!r}")


def test_validate_model_config():
    raimodel.validate_model_config(_valid_big_config())
    raimodel.validate_model_config(dict(TINY_CONFIG))

    cfg = _valid_big_config(); cfg["group_size"] = 100  # ceil(14336/100)=144 > 128
    _expect_value_error(cfg, "kernel maximum")

    cfg = _valid_big_config(); cfg["embed_group_size"] = 26  # ceil(4096/26)=158 > 128
    _expect_value_error(cfg, "kernel maximum")

    cfg = _valid_big_config(); cfg["head_dim"] = 100; cfg["num_heads"] = 41
    # 100 % 8 != 0 fires before the product check
    _expect_value_error(cfg, "multiple of 8")

    cfg = _valid_big_config(); cfg["num_kv_heads"] = 12
    _expect_value_error(cfg, "divisible")

    cfg = _valid_big_config(); cfg["head_dim"] = 64
    _expect_value_error(cfg, "num_heads * head_dim")

    cfg = _valid_big_config(); cfg["bits"] = 3
    _expect_value_error(cfg, "bit width")

    cfg = _valid_big_config(); cfg["norm_eps"] = float("nan")
    _expect_value_error(cfg, "finite")

    # Passes every earlier check but blows the 512MB RoPE table budget:
    # 5000 * (32512/2) * 2 * 4 = 650MB
    cfg = {
        "hidden_size": 32512, "num_layers": 1, "num_heads": 1,
        "num_kv_heads": 1, "head_dim": 32512, "intermediate_size": 1024,
        "vocab_size": 1000, "max_context": 5000, "rope_theta": 1e4,
        "norm_eps": 1e-5, "bits": 4, "group_size": 254,
        "embed_bits": 8, "embed_group_size": 254,
    }
    _expect_value_error(cfg, "RoPE")


def test_resolve_head_dim():
    assert raimodel.resolve_head_dim(None, 4096, 32) == 128
    assert raimodel.resolve_head_dim(128, 4096, 32) == 128
    try:
        raimodel.resolve_head_dim(96, 4096, 32)
    except ValueError as exc:
        assert "decoupled" in str(exc)
    else:
        raise AssertionError("decoupled head_dim was accepted")


def test_validate_export_options():
    parser = _FakeParser()
    good = argparse.Namespace(
        bits=4, embed_bits=8, group_size=128, embed_group_size=64,
        max_context=2048, cal_chunks=16, seq_len=2048,
    )
    raimodel.validate_export_options(parser, good)

    # rtn-style namespace without the calibration/embedding options
    rtn_style = argparse.Namespace(bits=4, group_size=128, max_context=2048)
    raimodel.validate_export_options(parser, rtn_style)

    for override in (
        {"bits": 3}, {"group_size": 127}, {"group_size": 256},
        {"embed_group_size": 0}, {"max_context": 0}, {"cal_chunks": 0},
        {"seq_len": 0}, {"embed_bits": 4},
    ):
        ns = argparse.Namespace(**{**vars(good), **override})
        try:
            raimodel.validate_export_options(parser, ns)
        except ValueError:
            pass
        else:
            raise AssertionError(f"options accepted with bad {override}")


def test_require_calibration_chunks():
    raimodel.require_calibration_chunks(3, 2048, 10000)
    try:
        raimodel.require_calibration_chunks(0, 4096, 1500)
    except RuntimeError as exc:
        msg = str(exc)
        assert "--seq-len" in msg and "1500" in msg
    else:
        raise AssertionError("empty calibration set was accepted")


def test_copy_tokenizer_json():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src" / "tokenizer.json"
        src.parent.mkdir()
        src.write_bytes(b'{"model": "a"}')
        dst = Path(td) / "out" / "tokenizer.json"
        dst.parent.mkdir()

        assert raimodel.copy_tokenizer_json(src, dst) is True
        assert dst.read_bytes() == b'{"model": "a"}'
        # identical file already present -> skip
        assert raimodel.copy_tokenizer_json(src, dst) is False
        # different file already present -> refuse
        dst.write_bytes(b'{"model": "b"}')
        try:
            raimodel.copy_tokenizer_json(src, dst)
        except RuntimeError as exc:
            assert "separate" in str(exc)
        else:
            raise AssertionError("differing tokenizer.json was overwritten")
        assert dst.read_bytes() == b'{"model": "b"}'  # untouched


# =============================================================================
# Golden fixture writer
# =============================================================================

def write_minimal_model(path, untied=False):
    """Emit a complete, valid, tiny .raimodel using the shared writer functions.

    2 layers, hidden=64, heads=4, kv_heads=2, head_dim=16, intermediate=128,
    vocab=96, group_size=64, embed_group_size=64; deterministic seeded
    weights.  `untied=True` adds a 4-bit lm_head as the final section.

    Returns (config, sections, expected, total_size) where `expected` holds
    exact dequantized f32 values computed from the same arrays the writer
    packed (for the Rust conformance test).
    """
    config = dict(TINY_CONFIG)
    rng = np.random.default_rng(202609 if untied else 202608)
    gs = config["group_size"]
    hs = config["hidden_size"]
    inter = config["intermediate_size"]
    vocab = config["vocab_size"]

    sections = []
    expected = {}

    # Section 0: embedding
    embed_w = rng.standard_normal((vocab, hs)) * 0.05
    e_codes, e_scales, e_zeros, _ = raimodel.quantize_embedding_8bit(
        embed_w, group_size=config["embed_group_size"]
    )
    sections.append(raimodel.build_embedding_section(e_codes, e_scales, e_zeros))
    embed_deq = raimodel.dequantize(
        e_codes, e_scales, e_zeros, config["embed_group_size"], np.float32
    )
    expected["embedding_row0"] = [float(x) for x in embed_deq[0, :]]

    # Sections 1..=L: layers
    kv_dim = config["num_kv_heads"] * config["head_dim"]
    dims = [(hs, hs), (kv_dim, hs), (kv_dim, hs), (hs, hs),
            (inter, hs), (inter, hs), (hs, inter)]
    for li in range(config["num_layers"]):
        linears = []
        for name, (rows, cols) in zip(raimodel.LAYER_LINEAR_NAMES, dims):
            w = rng.standard_normal((rows, cols)) * 0.1
            codes, scales, zeros, _ = raimodel.rtn_quantize(
                w, bits=config["bits"], group_size=gs, label=f"L{li}.{name}"
            )
            linears.append((codes, scales, zeros, rows, cols))
            if li == 0 and name == "q_proj":
                deq = raimodel.dequantize(codes, scales, zeros, gs, np.float32)
                expected["layer0_q_proj_row0_first8"] = [float(x) for x in deq[0, :8]]
        input_ln = (1.0 + 0.01 * rng.standard_normal(hs)).astype(np.float32)
        post_ln = (1.0 + 0.01 * rng.standard_normal(hs)).astype(np.float32)
        sections.append(raimodel.build_layer_section(linears, input_ln, post_ln))

    # Section L+1: final norm
    final_norm = (1.0 + 0.01 * rng.standard_normal(hs)).astype(np.float32)
    sections.append(raimodel.pack_norm_section(final_norm, "final norm"))

    # Section L+2 (untied only): lm_head
    if untied:
        lm_w = rng.standard_normal((vocab, hs)) * 0.05
        lm_codes, lm_scales, lm_zeros, _ = raimodel.rtn_quantize(
            lm_w, bits=config["bits"], group_size=gs, label="lm_head"
        )
        sections.append(
            raimodel.pack_linear_section(lm_codes, lm_scales, lm_zeros, vocab, hs)
        )
        lm_deq = raimodel.dequantize(lm_codes, lm_scales, lm_zeros, gs, np.float32)
        expected["lm_head_row0_first8"] = [float(x) for x in lm_deq[0, :8]]

    total_size = raimodel.write_raimodel(path, config, sections)
    return config, sections, expected, total_size


def _dequant_linear_row_prefix(section, gs, count):
    """Dequantize the first `count` columns of row 0 of a packed linear
    sub-section, straight from its raw bytes."""
    rows, cols = struct.unpack_from("<II", section, 0)
    num_groups = -(-cols // gs)
    params_size = rows * num_groups * 4
    params = np.frombuffer(section[8:8 + params_size], dtype="<f2").reshape(rows, num_groups, 2)
    nibbles = np.frombuffer(
        section[8 + params_size:8 + params_size + (rows * cols) // 2], dtype=np.uint8
    ).reshape(rows, cols // 2)
    values = []
    for col in range(count):
        byte = int(nibbles[0, col // 2])
        code = (byte & 0x0F) if col % 2 == 0 else (byte >> 4)
        g = col // gs
        values.append(_dequant_scalar_f32(code, params[0, g, 0], params[0, g, 1]))
    return (rows, cols), values


def _verify_container(data, config, sections, expected):
    """Parse the emitted bytes independently and check them against the
    config, the section blobs, and the expected dequantized values."""
    # Header
    assert data[0:4] == b"RAIM"
    assert struct.unpack_from("<I", data, 4)[0] == 1
    for offset, name in (
        (8, "hidden_size"), (12, "num_layers"), (16, "num_heads"),
        (20, "num_kv_heads"), (24, "head_dim"), (28, "intermediate_size"),
        (32, "vocab_size"), (36, "max_context"),
    ):
        assert struct.unpack_from("<I", data, offset)[0] == config[name], name
    assert struct.unpack_from("<f", data, 40)[0] == np.float32(config["rope_theta"])
    assert struct.unpack_from("<f", data, 44)[0] == np.float32(config["norm_eps"])
    assert (data[48], data[49], data[50], data[51]) == (
        config["bits"], config["group_size"], config["embed_bits"], config["embed_group_size"]
    )
    num_sections = struct.unpack_from("<I", data, 52)[0]
    assert num_sections == len(sections)

    # Section table: contiguous, byte-exact sections, ends exactly at EOF
    position = 64 + num_sections * 16
    for i in range(num_sections):
        offset, size = struct.unpack_from("<QQ", data, 64 + i * 16)
        assert offset == position, f"section {i} not contiguous"
        assert data[offset:offset + size] == sections[i], f"section {i} bytes differ"
        position = offset + size
    assert position == len(data), "file has trailing bytes past the last section"

    # Embedding row 0, dequantized from the raw file bytes
    vocab, hidden = config["vocab_size"], config["hidden_size"]
    egs = config["embed_group_size"]
    num_groups = -(-hidden // egs)
    off0, size0 = struct.unpack_from("<QQ", data, 64)
    sec0 = data[off0:off0 + size0]
    params_size = vocab * num_groups * 4
    e_params = np.frombuffer(sec0[:params_size], dtype="<f2").reshape(vocab, num_groups, 2)
    e_codes = np.frombuffer(sec0[params_size:], dtype=np.uint8).reshape(vocab, hidden)
    row0 = [
        _dequant_scalar_f32(e_codes[0, c], e_params[0, c // egs, 0], e_params[0, c // egs, 1])
        for c in range(hidden)
    ]
    assert row0 == expected["embedding_row0"]

    # Layer 0 q_proj row 0, first 8 columns (q_proj is the first sub-section)
    gs = config["group_size"]
    off1, size1 = struct.unpack_from("<QQ", data, 64 + 16)
    (rows, cols), values = _dequant_linear_row_prefix(data[off1:off1 + size1], gs, 8)
    assert (rows, cols) == (hidden, hidden)
    assert values == expected["layer0_q_proj_row0_first8"]

    # lm_head row 0, first 8 columns (untied only; last section)
    if "lm_head_row0_first8" in expected:
        off_l, size_l = struct.unpack_from("<QQ", data, 64 + (num_sections - 1) * 16)
        (rows, cols), values = _dequant_linear_row_prefix(data[off_l:off_l + size_l], gs, 8)
        assert (rows, cols) == (vocab, hidden)
        assert values == expected["lm_head_row0_first8"]


def test_write_minimal_model():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tiny.raimodel"
        config, sections, expected, total = write_minimal_model(path)
        data = path.read_bytes()
        assert len(data) == total
        # file size must equal header + table + sections exactly
        expected_size = (
            raimodel.HEADER_SIZE
            + len(sections) * raimodel.SECTION_ENTRY_SIZE
            + sum(len(s) for s in sections)
        )
        assert total == expected_size, f"{total} != {expected_size}"
        assert len(sections) == config["num_layers"] + 2  # tied: no lm_head
        _verify_container(data, config, sections, expected)


def test_write_minimal_model_untied():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tiny-untied.raimodel"
        config, sections, expected, total = write_minimal_model(path, untied=True)
        data = path.read_bytes()
        assert len(data) == total
        assert len(sections) == config["num_layers"] + 3  # untied: + lm_head
        assert "lm_head_row0_first8" in expected
        _verify_container(data, config, sections, expected)


def test_generate_golden_fixtures():
    """Write the fixtures the Rust conformance test reads, plus JSON sidecars
    with the exact expected dequantized values."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for untied in (False, True):
        name = "tiny-untied" if untied else "tiny-tied"
        model_path = FIXTURE_DIR / f"{name}.raimodel"
        config, sections, expected, total = write_minimal_model(model_path, untied=untied)
        assert model_path.stat().st_size == total
        _verify_container(model_path.read_bytes(), config, sections, expected)

        sidecar = {
            "config": config,
            "tied": not untied,
            "num_sections": len(sections),
            "file_size": total,
            "expected": expected,
        }
        sidecar_path = FIXTURE_DIR / f"{name}.expected.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="ascii")

        # The JSON floats must round-trip to the exact same f32 values.
        loaded = json.loads(sidecar_path.read_text(encoding="ascii"))
        for key, values in expected.items():
            round_tripped = [float(np.float32(v)) for v in loaded["expected"][key]]
            assert round_tripped == values, f"{name}: {key} did not round-trip"
        print(f"    fixture: {model_path} ({total} bytes)")


# =============================================================================
# Runner
# =============================================================================

TESTS = [
    test_pack_nibbles_roundtrip,
    test_pack_nibbles_rejects_odd_columns,
    test_group_param_byte_layout,
    test_header_field_offsets,
    test_bug1_group_params_never_recomputed_mid_group,
    test_gptq_beats_rtn_on_correlated_hessian,
    test_zero_hessian_uses_damping_floor,
    test_bad_hessian_raises_named_error,
    test_rtn_roundtrip_error_bound,
    test_embedding_8bit_roundtrip,
    test_validate_model_config,
    test_resolve_head_dim,
    test_validate_export_options,
    test_require_calibration_chunks,
    test_copy_tokenizer_json,
    test_write_minimal_model,
    test_write_minimal_model_untied,
    test_generate_golden_fixtures,
]


def main():
    failures = 0
    for test in TESTS:
        try:
            test()
        except Exception:
            failures += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS {test.__name__}")
    if failures:
        print(f"\n{failures}/{len(TESTS)} tests FAILED")
        sys.exit(1)
    print(f"\nAll {len(TESTS)} tests passed")


if __name__ == "__main__":
    main()
