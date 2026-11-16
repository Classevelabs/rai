#!/usr/bin/env python3
"""Shared quantizer/writer for the .raimodel container format (numpy-only).

Single source of truth used by export_raimodel.py (GPTQ, full calibration),
export_fast.py (GPTQ, fewer calibration chunks) and export_rtn.py
(round-to-nearest).  This module MUST NOT import torch or transformers:
the test suite (test_raimodel.py) and the golden-fixture generator run on
plain numpy.

Binary layout (all little-endian), mirrored by the Rust reader
(rai-infer/src/format.rs):

  [64 bytes]              Header
  [num_sections * 16]     Section table: (u64 offset, u64 size) per section.
                          Sections are contiguous: the first starts right
                          after the table, the last ends exactly at EOF.
  [variable]              Section 0:            embedding (8-bit)
  [variable]              Sections 1..=L:       transformer layers
  [variable]              Section L+1:          final RMSNorm (f32)
  [variable]              Section L+2 (untied): lm_head (4-bit linear)

Header field offsets:
   0  u8[4]  magic b"RAIM"
   4  u32    version (1)
   8  u32    hidden_size
  12  u32    num_layers
  16  u32    num_heads
  20  u32    num_kv_heads
  24  u32    head_dim
  28  u32    intermediate_size
  32  u32    vocab_size
  36  u32    max_context
  40  f32    rope_theta
  44  f32    norm_eps
  48  u8     bits (must be 4)
  49  u8     group_size
  50  u8     embed_bits (must be 8)
  51  u8     embed_group_size
  52  u32    num_sections (L+2 tied, L+3 untied)
  56  u8[8]  reserved (zero)

Section encodings:
  Embedding section:  [vocab * ceil(hidden/embed_gs) * 4 bytes group params]
                      [vocab * hidden bytes of u8 codes]
  Linear sub-section: [u32 rows][u32 cols]
                      [rows * ceil(cols/gs) * 4 bytes group params]
                      [rows * cols / 2 bytes packed nibbles]
  Layer section:      7 linears (q,k,v,o,gate,up,down) + 2 f32[hidden] norms
  Group params:       per row, per group: f16 scale at byte +0, f16 zero at
                      byte +2 (little-endian)
  Nibble packing:     low nibble = even column, high nibble = odd column
  Dequantization:     weight = code * scale + zero
"""

import json
import math
import os
import re
import struct
import tempfile
import time
from pathlib import Path

import numpy as np

# Smallest positive (subnormal) float16; scales are clamped to at least this
# so that stored f16 scales are always strictly positive.
MIN_F16_SCALE = float(np.nextafter(np.float16(0), np.float16(1)))

HEADER_SIZE = 64
SECTION_ENTRY_SIZE = 16

# Limits enforced by the Rust reader (rai-infer/src/format.rs).  Exports that
# violate these would only fail when the model is first loaded, hours after
# quantization started, so we mirror them here and fail fast.
MAX_HIDDEN_SIZE = 65_536
MAX_INTERMEDIATE_SIZE = 1_048_576
MAX_LAYERS = 1_024
MAX_HEADS = 1_024
MAX_VOCAB_SIZE = 10_000_000
MAX_CONTEXT = 1_000_000
MAX_GEMM_GROUPS = 1024
MAX_ROPE_TABLE_BYTES = 512 * 1024 * 1024

# Fixed order of the seven quantized linears inside every layer section.
LAYER_LINEAR_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


# =============================================================================
# Validation
# =============================================================================

def validate_export_options(parser, args):
    """CLI-level validation shared by all exporters.

    Only checks values known at argument-parse time; model-dependent
    constraints are checked by validate_model_config() as soon as the config
    is known (before any calibration work starts).
    """
    if args.bits != 4:
        parser.error("--bits must be 4; the .raimodel reader supports only 4-bit weights")
    if getattr(args, "embed_bits", 8) != 8:
        parser.error("--embed-bits must be 8; the .raimodel reader supports only 8-bit embeddings")
    for name, value in (
        ("--group-size", args.group_size),
        ("--embed-group-size", getattr(args, "embed_group_size", 64)),
    ):
        if value < 2 or value > 254 or value % 2:
            parser.error(f"{name} must be an even integer in 2..=254")
    if not 1 <= args.max_context <= MAX_CONTEXT:
        parser.error(f"--max-context must be in 1..={MAX_CONTEXT}")
    cal_chunks = getattr(args, "cal_chunks", None)
    seq_len = getattr(args, "seq_len", None)
    if (cal_chunks is not None and cal_chunks < 1) or (seq_len is not None and seq_len < 1):
        parser.error("--cal-chunks and --seq-len must be greater than zero")


def default_output_name(model, bits):
    """`<directory or repo name>-q<bits>.raimodel`, as `rai convert` names it.

    Mirrors `default_output_name` in rai-infer/src/convert.rs:

    * A directory on this machine is resolved first, so `--model .` is named
      after the directory it is rather than `.`.
    * Anything else is a HuggingFace repo id (`Qwen/Qwen2.5-0.5B-Instruct`), and
      both separators are split on: `str.split('/')` alone left `C:\\models\\Foo`
      intact and derived an output name containing a drive letter.
    * The name is kept as it is, case and spaces included. Lowercasing is
      invisible on Windows and macOS and a trap on Linux, where the file the
      exporter wrote and the name the user types are then two different files.
    """
    local = Path(model)
    if local.is_dir():
        tail = local.resolve().name
    else:
        tail = re.split(r"[\\/]+", str(model).rstrip("\\/"))[-1]
    if not tail:
        raise ValueError(f"cannot derive an output name from {model}; pass --output")
    return f"{tail}-q{bits}.raimodel"


def validate_model_config(config):
    """Mirror the Rust reader's validate_config() so bad exports fail fast.

    `config` is the dict later passed to write_header()/write_raimodel().
    Raises ValueError with a clear message on the first violated constraint.
    """
    bounded_nonzero = (
        ("hidden_size", MAX_HIDDEN_SIZE),
        ("num_layers", MAX_LAYERS),
        ("num_heads", MAX_HEADS),
        ("num_kv_heads", MAX_HEADS),
        ("head_dim", MAX_HIDDEN_SIZE),
        ("intermediate_size", MAX_INTERMEDIATE_SIZE),
        ("vocab_size", MAX_VOCAB_SIZE),
        ("max_context", MAX_CONTEXT),
    )
    for name, maximum in bounded_nonzero:
        value = config[name]
        if not 1 <= value <= maximum:
            raise ValueError(f"invalid {name}: {value}; the .raimodel reader requires 1..={maximum}")

    if config["bits"] != 4:
        raise ValueError(f"unsupported weight bit width {config['bits']}; the reader expects 4")
    if config["embed_bits"] != 8:
        raise ValueError(f"unsupported embedding bit width {config['embed_bits']}; the reader expects 8")
    for name in ("group_size", "embed_group_size"):
        value = config[name]
        if value < 2 or value > 254 or value % 2:
            raise ValueError(f"{name} must be an even integer in 2..=254, got {value}")
    if config["hidden_size"] % 2 or config["intermediate_size"] % 2:
        raise ValueError("hidden and intermediate dimensions must be even for packed 4-bit kernels")
    if config["head_dim"] % 8:
        raise ValueError(f"head_dim must be a multiple of 8 for SIMD attention kernels, got {config['head_dim']}")
    if config["num_heads"] % config["num_kv_heads"]:
        raise ValueError(
            f"num_heads ({config['num_heads']}) must be divisible by num_kv_heads ({config['num_kv_heads']})"
        )
    projected_hidden = config["num_heads"] * config["head_dim"]
    if projected_hidden != config["hidden_size"]:
        raise ValueError(
            f"hidden_size {config['hidden_size']} does not equal "
            f"num_heads * head_dim ({projected_hidden})"
        )

    group_size = config["group_size"]
    max_linear_groups = max(
        -(-config["hidden_size"] // group_size),
        -(-config["intermediate_size"] // group_size),
    )
    if max_linear_groups > MAX_GEMM_GROUPS:
        raise ValueError(
            f"group_size {group_size} needs {max_linear_groups} quantization groups for "
            f"hidden={config['hidden_size']}/intermediate={config['intermediate_size']}; "
            f"the reader's kernel maximum is {MAX_GEMM_GROUPS}. Use a larger --group-size."
        )
    embedding_groups = -(-config["hidden_size"] // config["embed_group_size"])
    if embedding_groups > MAX_GEMM_GROUPS:
        raise ValueError(
            f"embed_group_size {config['embed_group_size']} needs {embedding_groups} embedding "
            f"groups for hidden={config['hidden_size']}; the reader's kernel maximum is "
            f"{MAX_GEMM_GROUPS}. Use a larger --embed-group-size."
        )

    for name in ("rope_theta", "norm_eps"):
        value = config[name]
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}")

    rope_bytes = config["max_context"] * (config["head_dim"] // 2) * 2 * 4
    if rope_bytes > MAX_ROPE_TABLE_BYTES:
        raise ValueError(
            f"RoPE table would need {rope_bytes} bytes (max_context={config['max_context']}, "
            f"head_dim={config['head_dim']}); the reader's maximum is {MAX_ROPE_TABLE_BYTES}. "
            f"Lower --max-context."
        )


def resolve_head_dim(config_head_dim, hidden_size, num_heads):
    """Resolve head_dim from a HuggingFace config.

    Pass getattr(cfg, "head_dim", None) as `config_head_dim`.  Some configs
    declare an explicit head_dim that differs from hidden_size // num_heads
    (decoupled head dimension); the .raimodel format cannot represent that,
    so we refuse instead of silently exporting garbage attention shapes.
    """
    if config_head_dim:
        head_dim = int(config_head_dim)
        if num_heads * head_dim != hidden_size:
            raise ValueError(
                f"model config declares head_dim={head_dim} with num_heads={num_heads}, "
                f"so num_heads * head_dim = {num_heads * head_dim} != hidden_size {hidden_size}. "
                f"The .raimodel format cannot represent a decoupled head_dim yet; "
                f"this model cannot be exported."
            )
        return head_dim
    return hidden_size // num_heads


def read_rope_theta(cfg):
    """Read the RoPE base from a HuggingFace config, on any transformers version.

    transformers 5 moved this value: on 4.x it is `cfg.rope_theta`, on 5.x it
    lives in `cfg.rope_parameters["rope_theta"]` and the old attribute is gone.
    Every exporter used to read it as `getattr(cfg, "rope_theta", 10000.0)`, so
    on 5.x — which is what `requirements-lock.txt` pins and what a fresh
    `pip install -r requirements.txt` resolves to — the fallback silently won
    and the exported header carried the Llama constant instead of the model's
    own base.

    That is the worst shape a bug can take here. The export succeeds, the file
    passes every validation, `rai run` loads it, and it generates text with the
    wrong positional encoding. SmolLM2-1.7B (130000) exported as 10000;
    Mistral v0.2+ (1000000) and Llama-3 (500000) are affected the same way.
    Models that happen to use 10000 — TinyLlama, Phi-3, Gemma — were
    accidentally correct, which is why this survived.

    So: no default. A config this cannot read is a refusal, exactly as an
    unrepresentable architecture is a refusal in
    `assert_exportable_architecture`. A wrong RoPE base is not recoverable by
    the reader and not visible to the user.
    """
    parameters = getattr(cfg, "rope_parameters", None)
    if isinstance(parameters, dict) and parameters.get("rope_theta") is not None:
        return float(parameters["rope_theta"])

    theta = getattr(cfg, "rope_theta", None)
    if theta is not None:
        return float(theta)

    raise ValueError(
        "this checkpoint's config exposes no rope_theta, in neither the "
        "transformers 4 location (cfg.rope_theta) nor the transformers 5 one "
        "(cfg.rope_parameters['rope_theta']). Exporting with a guessed RoPE "
        "base would produce a model that loads cleanly and generates with the "
        "wrong positional encoding, so this is refused instead. If the "
        "architecture genuinely has no RoPE base, it cannot be represented by "
        "the .raimodel format."
    )


def require_calibration_chunks(num_chunks, seq_len, total_tokens):
    """Fail fast when the calibration text yields zero full-length chunks.

    Without this, every Hessian stays None and the export crashes much later
    with an opaque AttributeError.
    """
    if num_chunks == 0:
        raise RuntimeError(
            f"no calibration chunks: the calibration text tokenized to {total_tokens} tokens, "
            f"fewer than one --seq-len window of {seq_len}. "
            f"Lower --seq-len or provide more calibration text."
        )


def validate_f16_params(scale, zero_point, label):
    if (not np.isfinite(scale).all() or not np.isfinite(zero_point).all()
            or np.any(scale <= 0)):
        raise ValueError(f"{label} produced non-finite or non-positive FP16 quantization parameters")


# =============================================================================
# Group quantization primitives
# =============================================================================

def compute_group_params(group_slice, n_levels, label):
    """Per-row min/max affine params for one column group, rounded to f16.

    The f16 round-trip happens HERE, before any codes are derived, so the
    stored params are bit-identical to the params the codes are computed
    against and the Rust dequant (code * scale + zero) is exact.
    """
    group64 = np.asarray(group_slice, dtype=np.float64)
    row_min = group64.min(axis=1)
    row_max = group64.max(axis=1)
    scale = np.maximum((row_max - row_min) / (n_levels - 1), MIN_F16_SCALE)
    scale_f16 = scale.astype(np.float16)
    zero_f16 = row_min.astype(np.float16)
    validate_f16_params(scale_f16, zero_f16, label)
    return scale_f16, zero_f16


def rtn_quantize(weight_np, bits=4, group_size=128, label="tensor"):
    """Round-to-nearest quantization per group.

    Returns (codes u8 [rows, cols], scales f16 [rows, groups],
    zeros f16 [rows, groups], mse float).
    """
    weight64 = np.asarray(weight_np, dtype=np.float64)
    rows, cols = weight64.shape
    n_levels = 2 ** bits
    num_groups = -(-cols // group_size)

    codes = np.zeros((rows, cols), dtype=np.uint8)
    scales_f16 = np.zeros((rows, num_groups), dtype=np.float16)
    zeros_f16 = np.zeros((rows, num_groups), dtype=np.float16)

    for gid in range(num_groups):
        c_start = gid * group_size
        c_end = min(c_start + group_size, cols)
        group = weight64[:, c_start:c_end]
        scale_f16, zero_f16 = compute_group_params(group, n_levels, f"{label} group {gid}")
        scales_f16[:, gid] = scale_f16
        zeros_f16[:, gid] = zero_f16
        s64 = scale_f16.astype(np.float64)
        z64 = zero_f16.astype(np.float64)
        codes[:, c_start:c_end] = np.clip(
            np.round((group - z64[:, None]) / s64[:, None]), 0, n_levels - 1
        ).astype(np.uint8)

    mse = quantization_mse(weight64, codes, scales_f16, zeros_f16, group_size)
    return codes, scales_f16, zeros_f16, mse


def quantize_embedding_8bit(weight_np, group_size=64, label="embedding"):
    """8-bit uniform quantization of the embedding matrix.

    Returns (codes u8 [vocab, hidden], scales f16 [vocab, groups],
    zeros f16 [vocab, groups], mse float).
    """
    return rtn_quantize(weight_np, bits=8, group_size=group_size, label=label)


def quantization_mse(weight_np, codes, scales_f16, zeros_f16, group_size):
    """Weight-space MSE using the stored f16 params (matches Rust dequant)."""
    weight64 = np.asarray(weight_np, dtype=np.float64)
    rows, cols = weight64.shape
    total_sq_err = 0.0
    for gid in range(scales_f16.shape[1]):
        c_start = gid * group_size
        c_end = min(c_start + group_size, cols)
        s = scales_f16[:, gid].astype(np.float64)[:, None]
        z = zeros_f16[:, gid].astype(np.float64)[:, None]
        recon = codes[:, c_start:c_end].astype(np.float64) * s + z
        total_sq_err += float(np.sum((weight64[:, c_start:c_end] - recon) ** 2))
    return total_sq_err / (rows * cols)


def dequantize(codes, scales_f16, zeros_f16, group_size, dtype=np.float32):
    """Dequantize codes back to weights: code * scale + zero, per group.

    With dtype=float32 this reproduces the Rust reader's arithmetic exactly
    (f16 params convert to f32 losslessly; u8 codes convert losslessly).
    """
    rows, cols = codes.shape
    out = np.empty((rows, cols), dtype=dtype)
    for gid in range(scales_f16.shape[1]):
        c_start = gid * group_size
        c_end = min(c_start + group_size, cols)
        s = scales_f16[:, gid].astype(dtype)[:, None]
        z = zeros_f16[:, gid].astype(dtype)[:, None]
        out[:, c_start:c_end] = codes[:, c_start:c_end].astype(dtype) * s + z
    return out


# =============================================================================
# GPTQ (Frantar et al., 2023 — "GPTQ: Accurate Post-Training Quantization
# for Generative Pre-trained Transformers", arXiv:2210.17323)
# =============================================================================

def _cholesky_inverse_upper(H):
    """Upper-triangular Cholesky factor U of H^-1, with H^-1 = U^T @ U.

    This is the factor the GPTQ reference implementation uses (it computes
    torch.linalg.cholesky(cholesky_inverse(cholesky(H)), upper=True)):
    the per-column divisor is d_j = U[j, j] and the error-propagation row is
    U[j, j+1:].  Note this is NOT the plain symmetric inverse H^-1 itself.
    Raises np.linalg.LinAlgError when H is not positive definite.
    """
    L = np.linalg.cholesky(H)
    identity = np.eye(H.shape[0], dtype=np.float64)
    H_inv = np.linalg.solve(L.T, np.linalg.solve(L, identity))
    # Symmetrize before factorizing: the two triangular solves can leave
    # tiny asymmetries that break the second Cholesky.
    H_inv = (H_inv + H_inv.T) * 0.5
    return np.linalg.cholesky(H_inv).T


def gptq_quantize(weight_np, hessian_np, bits=4, block_size=128, group_size=128,
                  label="tensor"):
    """GPTQ quantization returning integer codes and f16-rounded group params.

    Algorithm (Frantar et al., 2023, arXiv:2210.17323, Algorithm 1 with
    lazy batch updates): let U be the upper-triangular Cholesky factor of
    H^-1 (H^-1 = U^T U).  For each column j in order:
        q_j   = quantize(W[:, j]) against the current group's params
        err_j = (W[:, j] - dequant(q_j)) / U[j, j]
        W[:, j+1:] -= outer(err_j, U[j, j+1:])
    Updates within a block of `block_size` columns are applied eagerly; the
    columns after the block receive the accumulated update in one matmul.

    Group params are computed ONCE per group, at the first column of the
    group, from the current (error-compensated) weights — never recomputed
    mid-group, regardless of how group boundaries fall relative to block
    boundaries — and are round-tripped through f16 BEFORE any code is
    derived, so the stored params are exactly the params the codes were
    computed against.

    Returns:
        codes:  np.uint8 [rows, cols] — integer codes (0 .. 2^bits - 1)
        scales: np.float16 [rows, num_groups]
        zeros:  np.float16 [rows, num_groups]
        mse:    float — weight-space MSE vs the original weights
    """
    weight64 = np.asarray(weight_np, dtype=np.float64)
    rows, cols = weight64.shape
    if hessian_np.shape != (cols, cols):
        raise ValueError(
            f"{label}: Hessian shape {hessian_np.shape} does not match weight columns {cols}"
        )
    n_levels = 2 ** bits
    num_groups = -(-cols // group_size)

    W = weight64.copy()
    H = np.asarray(hessian_np, dtype=np.float64).copy()

    # Damp the Hessian.  The floor keeps damp > 0 even for an all-zero
    # Hessian (e.g. a dead input feature set), where damp would otherwise be
    # 0 and both Cholesky attempts would fail after hours of calibration.
    damp = 0.01 * max(float(np.mean(np.diag(H))), 1e-6)
    H[np.diag_indices(cols)] += damp

    try:
        U = _cholesky_inverse_upper(H)
    except np.linalg.LinAlgError:
        H[np.diag_indices(cols)] += 0.1 * max(float(np.mean(np.diag(H))), 1e-6)
        try:
            U = _cholesky_inverse_upper(H)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"{label}: Hessian is not positive definite even after damping twice; "
                f"cannot run GPTQ on this tensor"
            ) from exc

    codes = np.zeros((rows, cols), dtype=np.uint8)
    scales_f16 = np.zeros((rows, num_groups), dtype=np.float16)
    zeros_f16 = np.zeros((rows, num_groups), dtype=np.float16)

    group_scale = None
    group_zero = None
    cur_gid = -1

    for block_start in range(0, cols, block_size):
        block_end = min(block_start + block_size, cols)
        err_block = np.zeros((rows, block_end - block_start), dtype=np.float64)

        for j in range(block_start, block_end):
            gid = j // group_size
            if gid != cur_gid:
                # First column of a new group: freeze this group's params.
                cur_gid = gid
                g_start = gid * group_size
                g_end = min(g_start + group_size, cols)
                scale_f16, zero_f16 = compute_group_params(
                    W[:, g_start:g_end], n_levels, f"{label} group {gid}"
                )
                scales_f16[:, gid] = scale_f16
                zeros_f16[:, gid] = zero_f16
                group_scale = scale_f16.astype(np.float64)
                group_zero = zero_f16.astype(np.float64)

            d = U[j, j]
            if not np.isfinite(d) or d <= 0:
                raise RuntimeError(
                    f"{label}: Cholesky factor diagonal U[{j},{j}] = {d!r} is not a "
                    f"positive finite number; aborting instead of writing corrupt codes"
                )

            w_col = W[:, j]
            q_col = np.clip(
                np.round((w_col - group_zero) / group_scale), 0, n_levels - 1
            ).astype(np.uint8)
            codes[:, j] = q_col
            w_hat = q_col.astype(np.float64) * group_scale + group_zero

            err = (w_col - w_hat) / d
            err_block[:, j - block_start] = err
            if j + 1 < block_end:
                W[:, j + 1:block_end] -= np.outer(err, U[j, j + 1:block_end])

        if block_end < cols:
            W[:, block_end:] -= err_block @ U[block_start:block_end, block_end:]
        if not np.isfinite(W[:, block_start:]).all():
            raise RuntimeError(
                f"{label}: non-finite weights after the GPTQ block update for columns "
                f"{block_start}..{block_end - 1}; the Hessian is likely ill-conditioned"
            )

    mse = quantization_mse(weight64, codes, scales_f16, zeros_f16, group_size)
    return codes, scales_f16, zeros_f16, mse


# =============================================================================
# Binary format packing
# =============================================================================

def pack_nibbles(codes):
    """Pack u8 codes (0-15) into nibble pairs: low nibble = even column,
    high nibble = odd column.

    Uses an explicit check rather than `assert` so the guard survives
    `python -O`.
    """
    rows, cols = codes.shape
    if cols % 2 != 0:
        raise ValueError(f"cannot pack nibbles: column count must be even, got {cols}")
    even = codes[:, 0::2]
    odd = codes[:, 1::2]
    return ((even & 0x0F) | ((odd & 0x0F) << 4)).astype(np.uint8)


def pack_group_params(scales, zeros):
    """Pack per-row-per-group f16 scales and zeros into bytes.

    Layout: for each row r, for each group g:
    [f16 scale (LE) at +0, f16 zero (LE) at +2] = 4 bytes.
    """
    if scales.shape != zeros.shape:
        raise ValueError(f"scales shape {scales.shape} != zeros shape {zeros.shape}")
    rows, num_groups = scales.shape
    params = np.empty((rows, num_groups, 2), dtype=np.float16)
    params[:, :, 0] = scales
    params[:, :, 1] = zeros
    return params.astype("<f2").tobytes()


def pack_linear_section(codes, scales, zeros, rows, cols):
    """Pack a quantized linear layer: [u32 rows][u32 cols][params][nibbles]."""
    if codes.shape != (rows, cols):
        raise ValueError(f"codes shape {codes.shape} does not match declared ({rows}, {cols})")
    data = bytearray()
    data.extend(struct.pack("<II", rows, cols))
    data.extend(pack_group_params(scales, zeros))
    data.extend(pack_nibbles(codes).tobytes())
    return bytes(data)


def pack_norm_section(weights_np, label="norm"):
    """Pack RMSNorm weights as little-endian f32, rejecting non-finite values
    (the Rust reader would refuse to load them)."""
    weights = np.asarray(weights_np, dtype=np.float32)
    if not np.isfinite(weights).all():
        raise ValueError(f"{label} weights contain non-finite values")
    return weights.astype("<f4").tobytes()


def build_embedding_section(codes, scales, zeros):
    """Section 0: [group params][u8 codes] (no sub-header)."""
    return pack_group_params(scales, zeros) + np.ascontiguousarray(codes, dtype=np.uint8).tobytes()


def build_layer_section(linears_packed, input_ln, post_attn_ln):
    """One transformer layer: 7 packed linears in LAYER_LINEAR_NAMES order,
    then the two f32 RMSNorm vectors."""
    if len(linears_packed) != len(LAYER_LINEAR_NAMES):
        raise ValueError(
            f"layer section needs exactly {len(LAYER_LINEAR_NAMES)} linears, got {len(linears_packed)}"
        )
    section = bytearray()
    for codes, scales, zeros, rows, cols in linears_packed:
        section.extend(pack_linear_section(codes, scales, zeros, rows, cols))
    section.extend(pack_norm_section(input_ln, "input_layernorm"))
    section.extend(pack_norm_section(post_attn_ln, "post_attention_layernorm"))
    return bytes(section)


def write_header(f, config, num_sections):
    """Write the 64-byte header (field offsets documented in the module docstring)."""
    header = bytearray(HEADER_SIZE)
    header[0:4] = b"RAIM"
    struct.pack_into("<I", header, 4, 1)
    struct.pack_into("<I", header, 8, config["hidden_size"])
    struct.pack_into("<I", header, 12, config["num_layers"])
    struct.pack_into("<I", header, 16, config["num_heads"])
    struct.pack_into("<I", header, 20, config["num_kv_heads"])
    struct.pack_into("<I", header, 24, config["head_dim"])
    struct.pack_into("<I", header, 28, config["intermediate_size"])
    struct.pack_into("<I", header, 32, config["vocab_size"])
    struct.pack_into("<I", header, 36, config["max_context"])
    struct.pack_into("<f", header, 40, config["rope_theta"])
    struct.pack_into("<f", header, 44, config["norm_eps"])
    header[48] = config["bits"]
    header[49] = config["group_size"]
    header[50] = config["embed_bits"]
    header[51] = config["embed_group_size"]
    struct.pack_into("<I", header, 52, num_sections)
    f.write(header)


def write_raimodel(path, config, sections_data):
    """Assemble and write the container: header, section table, sections.

    `sections_data` must be the complete ordered list of section byte strings
    (embedding, layers, final norm, optional lm_head).  Validates the config
    against the Rust reader's constraints and returns the total file size.
    """
    validate_model_config(config)
    num_sections = len(sections_data)
    tied_sections = config["num_layers"] + 2
    untied_sections = config["num_layers"] + 3
    if num_sections not in (tied_sections, untied_sections):
        raise ValueError(
            f"invalid section count {num_sections}; expected {tied_sections} (tied) "
            f"or {untied_sections} (untied) for {config['num_layers']} layers"
        )
    for i, data in enumerate(sections_data):
        if len(data) == 0:
            raise ValueError(f"section {i} is empty")

    data_start = HEADER_SIZE + num_sections * SECTION_ENTRY_SIZE
    offsets = []
    current_offset = data_start
    for data in sections_data:
        offsets.append(current_offset)
        current_offset += len(data)

    def write(f):
        write_header(f, config, num_sections)
        for offset, data in zip(offsets, sections_data):
            f.write(struct.pack("<QQ", offset, len(data)))
        for data in sections_data:
            f.write(data)

    _write_atomically(path, write)
    return current_offset


def _write_atomically(path, write):
    """Write through a sibling `.partial` file and rename it into place.

    An export that dies mid-write (killed, disk full) then leaves no truncated
    file under the real name, and never a half-overwritten earlier export. The
    name is this process's own and created exclusively, as in convert.rs, so
    two exports to one target cannot write into each other's bytes.
    """
    path = Path(path)
    partial = path.with_name(f"{path.name}.{os.getpid()}-{time.time_ns()}.partial")
    f = open(partial, "xb")
    try:
        with f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    # A finished file is kept when only the rename fails (the target held open
    # by a scanner, say): deleting hours of export would be worse.
    try:
        os.replace(partial, path)
    except OSError as error:
        raise RuntimeError(
            f"moving {partial} into place as {path} failed ({error}); the finished file "
            f"is intact at the first path — move or rename it by hand"
        ) from error


# =============================================================================
# Tokenizer copy
# =============================================================================

def checkpoint_tokenizer_json(model, tokenizer):
    """The checkpoint's own tokenizer.json bytes, which `rai convert` copies.

    Both tools then write the same file, so the check beside a model compares
    like with like: the copy a tokenizer saves is re-serialized (newer
    `tokenizers` write BPE merges as pairs) and was refused as another model's
    when `rai convert` had already put the original there. A checkpoint with no
    tokenizer.json of its own gets the one its fast tokenizer saves.
    """
    if Path(model).is_dir():
        path = Path(model) / "tokenizer.json"
        path = path if path.is_file() else None
    else:
        # A repo id: the file from_pretrained has just put in the cache.
        from transformers.utils import cached_file

        path = cached_file(model, "tokenizer.json", _raise_exceptions_for_missing_entries=False)
    if path is not None:
        return Path(path).read_bytes()
    return saved_tokenizer_json(tokenizer)


def saved_tokenizer_json(tokenizer):
    """The tokenizer.json bytes a fast tokenizer saves: the file the runtime reads."""
    with tempfile.TemporaryDirectory(prefix="rai_tokenizer_") as tmp_dir:
        tokenizer.save_pretrained(tmp_dir)
        saved = Path(tmp_dir) / "tokenizer.json"
        if not saved.exists():
            raise RuntimeError("tokenizer export did not produce the required tokenizer.json")
        return saved.read_bytes()


def prepare_output_folder(output_path):
    """Create the folder the model and its tokenizer are written into, before
    any work, as `rai convert` does: an export into a new folder used to fail
    at the very end, when the finished model was written."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def check_tokenizer_destination(tokenizer_json, dst_path):
    """Refuse when a DIFFERENT tokenizer.json already sits where this one goes.

    Exporters call this before quantizing, so a clash is refused in seconds
    instead of after the whole export, with no model written beside another
    model's tokenizer.
    """
    dst_path = Path(dst_path)
    if dst_path.exists() and dst_path.read_bytes() != tokenizer_json:
        raise RuntimeError(
            f"refusing to overwrite {dst_path}: it differs from this model's tokenizer.json. "
            f"Another model's tokenizer already lives there — export into a separate "
            f"output directory (--output <dir>/<name>.raimodel)."
        )


def place_tokenizer_json(tokenizer_json, dst_path):
    """Write tokenizer.json next to the model, never clobbering a different one.

    Returns True if the file was written, False if an identical one was
    already there. Raises RuntimeError if a DIFFERENT one is.
    """
    dst_path = Path(dst_path)
    check_tokenizer_destination(tokenizer_json, dst_path)
    if dst_path.exists():
        return False
    _write_atomically(dst_path, lambda f: f.write(tokenizer_json))
    return True


# =============================================================================
# Chat record
# =============================================================================

def chat_record(tokenizer, generation_config):
    """What the runtime needs to prompt the model as transformers does.

    Mirror of `ChatRecord::from_checkpoint` in src/chat_format.rs: the chat
    template (the one named `default` when there are several), the two tokens
    it spells, and the end-of-sequence ids from the generation config.
    """
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        template = template.get("default")
    eos_ids = getattr(generation_config, "eos_token_id", None)
    if eos_ids is None:
        eos_ids = []
    elif isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    return {
        "chat_template": template,
        "bos_token": getattr(tokenizer, "bos_token", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_ids": [int(token_id) for token_id in eos_ids],
    }


def chat_record_path(model_path):
    """`<name>.chat.json` beside `<name>.raimodel`, as the runtime looks for it."""
    return Path(model_path).with_suffix(".chat.json")


def write_chat_record(model_path, record):
    """Write `record` beside the model, as `rai convert` does."""
    path = chat_record_path(model_path)
    data = (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _write_atomically(path, lambda f: f.write(data))
    return path


# ---------------------------------------------------------------------------
# HuggingFace loading and architecture compatibility
#
# torch/transformers are imported lazily inside these functions so this module
# keeps its numpy-only import contract (test_raimodel.py runs without them).
# ---------------------------------------------------------------------------

# The model_types whose forward pass the runtime implements and has checked
# against the reference: tensor names alone prove nothing about the arithmetic.
# Same list as VERIFIED_MODEL_TYPES in src/convert.rs, held equal by
# tests/limits_are_single_sourced.rs.
VERIFIED_MODEL_TYPES = (
    "llama",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "gemma",
    "gemma2",
    "gemma3_text",
    "phi3",
    "olmo2",
    "olmoe",
)

# Verified families whose maths container v1 cannot express even though their
# module tree looks Llama-shaped; `rai convert` writes v2 and stores them.
_UNSUPPORTED_MODEL_TYPES = {
    "gemma": "Gemma scales embeddings by sqrt(hidden) and its RMSNorm applies (1 + weight)",
    "gemma2": "Gemma2 adds logit softcapping and (1 + weight) RMSNorm",
    "gemma3_text": "Gemma3 adds per-head QK norm and (1 + weight) RMSNorm",
}

# Scales applied outside any stored tensor (Granite's multipliers, MiniCPM's
# depth and width scaling); None means no value of the field is neutral.
_UNSTORED_SCALES = (
    ("embedding_multiplier", 1.0),
    ("residual_multiplier", 1.0),
    ("logits_scaling", 1.0),
    ("attention_multiplier", None),
    ("scale_emb", 1.0),
    ("scale_depth", None),
    ("dim_model_base", None),
)


def _unrepresentable_anywhere(config):
    """Config-level maths no container version stores, so `rai convert` refuses it too."""
    found = []
    model_type = getattr(config, "model_type", None)
    if not model_type:
        found.append(
            "config.json declares no model_type, so the architecture cannot be checked "
            "against the ones this runtime implements."
        )
    elif model_type not in VERIFIED_MODEL_TYPES:
        found.append(
            f"model_type '{model_type}' has no verified forward pass in this runtime "
            f"(verified: {', '.join(VERIFIED_MODEL_TYPES)}). A family with the same tensor "
            f"names can still rotate, scale or normalize differently, and would convert "
            f"into a model that computes something else."
        )
    for field, neutral in _UNSTORED_SCALES:
        value = getattr(config, field, None)
        if value is not None and (neutral is None or value != neutral):
            found.append(
                f"config declares {field}={value}; the container has no slot for that "
                f"scale, so the model would export cleanly and compute something else."
            )
    no_rope = getattr(config, "no_rope_layers", None)
    if no_rope is not None:
        skips_rope = any(flag == 0 for flag in no_rope)
    else:
        interval = getattr(config, "no_rope_layer_interval", None)
        layers = getattr(config, "num_hidden_layers", None)
        skips_rope = bool(interval) and (layers is None or interval <= layers)
    if skips_rope:
        found.append(
            "config declares layers without rotary embeddings (no_rope_layers); the reader "
            "rotates every layer, so positions would be wrong on those layers."
        )
    for field, factor in _partial_rotary_factors(config):
        if factor != 1.0:
            found.append(
                f"config declares {field}={factor}; the reader rotates the full head, so "
                f"positions would be wrong."
            )
    for field, rope_type in _rope_schemes(config):
        # The stored scheme applies to the full-attention layers, as in Gemma3.
        sliding = field.endswith(".sliding_attention")
        if rope_type == "default" or (rope_type == "llama3" and not sliding):
            continue
        found.append(
            f"config declares {field} type '{rope_type}'; the container stores plain RoPE "
            f"and the llama3 scheme of the full-attention layers only, so positions would "
            f"be wrong."
        )
    return found


def _rope_parameter_entries(parameters):
    """(layer kind or None, entry) for a flat or a per-layer-type rope_parameters dict."""
    if any(isinstance(value, dict) for value in parameters.values()):
        return list(parameters.items())
    return [(None, parameters)]


def _partial_rotary_factors(config):
    """Every partial_rotary_factor the config declares, in either transformers layout."""
    found = []
    legacy = getattr(config, "partial_rotary_factor", None)
    if legacy is not None:
        found.append(("partial_rotary_factor", legacy))
    parameters = getattr(config, "rope_parameters", None)
    if isinstance(parameters, dict):
        for kind, entry in _rope_parameter_entries(parameters):
            if isinstance(entry, dict) and entry.get("partial_rotary_factor") is not None:
                field = "rope_parameters" + (f".{kind}" if kind else "")
                found.append((f"{field}.partial_rotary_factor", entry["partial_rotary_factor"]))
    return found


def _rope_schemes(config):
    """Every RoPE scheme the config names, as (field, rope_type), on any transformers version.

    transformers 5 keeps them in `rope_parameters`, flat or one entry per layer
    type, and serves `rope_scaling` as an alias of the same dict; 4.x has
    `rope_scaling` alone. A scheme that names no type is plain RoPE.
    """
    parameters = getattr(config, "rope_parameters", None)
    legacy = getattr(config, "rope_scaling", None)
    sources = [("rope_parameters", parameters)]
    if legacy is not parameters:
        sources.append(("rope_scaling", legacy))
    found = []
    for name, value in sources:
        if not value:
            continue
        if not isinstance(value, dict):
            found.append((name, str(value)))
            continue
        for kind, entry in _rope_parameter_entries(value):
            if isinstance(entry, dict):
                field = name + (f".{kind}" if kind else "")
                found.append((field, entry.get("rope_type") or entry.get("type") or "default"))
    return found


def load_hf_causal_lm(model_path, dtype_name="float16"):
    """Load a causal LM for weight extraction, on CPU, without `accelerate`.

    Export only ever reads weights, so it deliberately avoids `device_map`
    (which requires the optional `accelerate` package) and passes the dtype
    keyword under the name the installed transformers expects: `dtype` from
    5.0 onward, `torch_dtype` before it.
    """
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, dtype_name)
    try:
        major = int(str(transformers.__version__).split(".", 1)[0])
    except (TypeError, ValueError):
        major = 5
    kwargs = {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model.eval()


def _iter_export_linears(layer):
    """The seven projections this format stores, as (name, module) pairs."""
    attn = layer.self_attn
    mlp = layer.mlp
    return (
        ("q_proj", attn.q_proj),
        ("k_proj", attn.k_proj),
        ("v_proj", attn.v_proj),
        ("o_proj", attn.o_proj),
        ("gate_proj", mlp.gate_proj),
        ("up_proj", mlp.up_proj),
        ("down_proj", mlp.down_proj),
    )


def assert_exportable_architecture(model, config, max_context):
    """Refuse to export a model the .raimodel format cannot represent.

    The container stores exactly: an 8-bit embedding table, seven 4-bit
    projections and two RMSNorm weight vectors per layer, a final RMSNorm, and
    an optional 4-bit lm_head — with full causal attention and plain RoPE.
    Anything carrying state outside that set (bias vectors, QK norms, RoPE
    scaling, MoE routers, logit softcapping) would be silently dropped and
    produce a model that loads cleanly and generates nonsense, so every such
    case is a hard error here instead.

    Raises RuntimeError listing every problem found.
    """
    problems = []

    model_type = getattr(config, "model_type", None)
    if model_type in _UNSUPPORTED_MODEL_TYPES:
        problems.append(
            f"model_type '{model_type}' is not supported: "
            f"{_UNSUPPORTED_MODEL_TYPES[model_type]}, which this format does not store."
        )

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError(
            "this checkpoint does not expose model.model.layers; the exporter supports "
            "Llama-style causal LMs (LlamaForCausalLM, MistralForCausalLM, and "
            "architecturally identical models)."
        )

    biased = []
    qk_normed = []
    norm_biased = []
    for index, layer in enumerate(layers):
        try:
            linears = _iter_export_linears(layer)
        except AttributeError as error:
            raise RuntimeError(
                f"layer {index} does not expose the expected Llama-style projections "
                f"({error}); this architecture is not supported."
            ) from error
        for name, linear in linears:
            if getattr(linear, "bias", None) is not None:
                biased.append(f"layer {index}.{name}")
        for norm_attr in ("q_norm", "k_norm"):
            if getattr(layer.self_attn, norm_attr, None) is not None:
                qk_normed.append(f"layer {index}.self_attn.{norm_attr}")
        for norm_attr in ("input_layernorm", "post_attention_layernorm"):
            if getattr(getattr(layer, norm_attr, None), "bias", None) is not None:
                norm_biased.append(f"layer {index}.{norm_attr}")

    # Checked before the v1-only problems below, whose message sends the user
    # to `rai convert`: that path refuses these as well.
    unrepresentable = _unrepresentable_anywhere(config)
    if norm_biased:
        unrepresentable.append(
            f"{len(norm_biased)} layer norm(s) carry a bias (e.g. {norm_biased[0]}); the "
            f"container stores RMSNorm weights only, so a biased LayerNorm would be computed "
            f"as a different operation."
        )
    if unrepresentable:
        raise RuntimeError(
            "this checkpoint cannot be represented by any .raimodel container version, so "
            "`rai convert` refuses it too:\n  - " + "\n  - ".join(unrepresentable)
        )

    if biased:
        problems.append(
            f"{len(biased)} projection(s) carry bias vectors (e.g. {biased[0]}); the "
            f"format stores weights only, so the biases would be silently dropped. "
            f"Qwen2/Qwen2.5 are the common case here."
        )
    if qk_normed:
        problems.append(
            f"{len(qk_normed)} per-head QK norm(s) present (e.g. {qk_normed[0]}); the "
            f"format has no place to store them."
        )

    # transformers >= 5 normalizes plain RoPE into {"rope_type": "default"},
    # so the presence of the field means nothing on its own. Types no container
    # stores were refused above; llama3 is stored by v2 only.
    for field, rope_type in _rope_schemes(config):
        if rope_type == "llama3":
            problems.append(
                f"config declares {field} type 'llama3'; container v1 builds a plain RoPE "
                f"table from rope_theta alone, so positions would be wrong. "
                f"Llama-3.1/3.2 are the common case here."
            )
            break

    for attr in ("num_experts", "num_local_experts"):
        if getattr(config, attr, None):
            problems.append(
                f"config declares {attr}={getattr(config, attr)}; mixture-of-experts "
                f"routing is not supported."
            )
            break

    for attr in ("attn_logit_softcapping", "final_logit_softcapping"):
        if getattr(config, attr, None):
            problems.append(f"config declares {attr}; logit softcapping is not supported.")

    sliding_window = getattr(config, "sliding_window", None)
    uses_sliding = getattr(config, "use_sliding_window", True)
    if sliding_window and uses_sliding and max_context > sliding_window:
        problems.append(
            f"config declares sliding_window={sliding_window} but --max-context is "
            f"{max_context}; the reader always uses full causal attention, so exports "
            f"beyond the window would diverge. Re-run with --max-context {sliding_window} "
            f"or lower."
        )

    if problems:
        # Naming the way out matters more here than usual. This path is the only
        # one that does calibrated GPTQ, and the fallback is not free: on
        # SmolLM2-1.7B, the one advertised checkpoint both paths accept, GPTQ
        # costs +11.2% perplexity against fp16 where round-to-nearest costs
        # +30.3% (BENCHMARKS.md). A user who hits this and is told only "cannot
        # be represented" has no way to know they are about to pay that.
        raise RuntimeError(
            "this checkpoint cannot be represented by the .raimodel container "
            "version these scripts write (v1):\n  - "
            + "\n  - ".join(problems)
            + "\n\n`rai convert` writes container v2 and does represent all of "
            "the above, so it will convert this checkpoint. It quantizes by "
            "round-to-nearest rather than by calibrated GPTQ, and that is a "
            "measurable difference, not a formality: on the one checkpoint both "
            "paths accept, round-to-nearest costs +30.3% perplexity against the "
            "fp16 model where this path costs +11.2%. See "
            "'The calibrated quantizer, measured end to end' in BENCHMARKS.md."
        )
