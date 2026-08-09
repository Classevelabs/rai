#!/usr/bin/env python3
"""Export any HuggingFace LLaMA/Mistral-family model to .raimodel format.

Supports: SmolLM-135M, Mistral-7B-Instruct-v0.3, LLaMA-2/3, and similar architectures.

Pipeline:
1. Loads model in FP16 on GPU (or CPU)
2. Collects Hessians for all layers in one calibration pass
3. GPTQ-4bit quantizes each linear layer — returns INTEGER CODES + group params
4. Packs codes as nibbles (2 per byte), group params as f16
5. Quantizes embedding at 8-bit uniform (per-row-per-group scale/zero)
6. Writes .raimodel binary file with header + sections
7. Copies tokenizer.json alongside

CRITICAL: Group params are round-tripped through f16 BEFORE computing final codes,
so Rust dequant exactly matches Python: weight = code * f16_scale + f16_zero.

File format:
  [64 bytes]            Header
  [num_sections * 16]   Section index table
  [variable]            Section 0: Embedding (8-bit)
  [variable]            Sections 1..N: Transformer layers (4-bit linears + f32 norms)
  [variable]            Section N+1: Final RMSNorm (f32)
"""

import argparse
import struct
import time
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MIN_F16_SCALE = float(np.nextafter(np.float16(0), np.float16(1)))


def validate_export_options(parser, args):
    if args.bits != 4:
        parser.error("--bits must be 4; the .raimodel reader supports only 4-bit weights")
    if args.embed_bits != 8:
        parser.error("--embed-bits must be 8; the .raimodel reader supports only 8-bit embeddings")
    for name, value in (
        ("--group-size", args.group_size),
        ("--embed-group-size", args.embed_group_size),
    ):
        if value < 2 or value > 254 or value % 2:
            parser.error(f"{name} must be an even integer in 2..=254")
    if not 1 <= args.max_context <= 1_000_000:
        parser.error("--max-context must be in 1..=1000000")
    if args.cal_chunks < 1 or args.seq_len < 1:
        parser.error("--cal-chunks and --seq-len must be greater than zero")


def validate_f16_params(scale, zero_point, label):
    if (not np.isfinite(scale).all() or not np.isfinite(zero_point).all()
            or np.any(scale <= 0)):
        raise ValueError(f"{label} produced non-finite or non-positive FP16 quantization parameters")


# =============================================================================
# GPTQ Quantization (returns integer codes, not dequantized weights)
# =============================================================================

def gptq_quantize_to_codes(weight_np, hessian_np, bits=4, block_size=128, group_size=128):
    """GPTQ quantization that returns integer codes and f16-rounded group params.

    Returns:
        codes: np.ndarray[uint8] shape [rows, cols] — integer quantization codes (0..15)
        scales: np.ndarray[float16] shape [rows, num_groups] — per-row-per-group scales
        zeros: np.ndarray[float16] shape [rows, num_groups] — per-row-per-group zero points
        mse: float — weight MSE
    """
    rows, cols = weight_np.shape
    n_levels = 2 ** bits
    num_groups = (cols + group_size - 1) // group_size

    W = weight_np.copy().astype(np.float64)
    H = hessian_np.copy().astype(np.float64)

    # Damp the Hessian
    damp = 0.01 * np.mean(np.diag(H))
    H += damp * np.eye(cols)

    # Cholesky inversion
    try:
        L = np.linalg.cholesky(H)
        H_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(cols)))
    except np.linalg.LinAlgError:
        H += 0.1 * np.mean(np.diag(H)) * np.eye(cols)
        L = np.linalg.cholesky(H)
        H_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(cols)))

    codes = np.zeros((rows, cols), dtype=np.uint8)
    # We'll collect scale/zero per group, round-trip through f16
    scales_f16 = np.zeros((rows, num_groups), dtype=np.float16)
    zeros_f16 = np.zeros((rows, num_groups), dtype=np.float16)

    for block_start in range(0, cols, block_size):
        block_end = min(block_start + block_size, cols)
        err_block = np.zeros((rows, block_end - block_start))

        for j in range(block_start, block_end):
            gid = j // group_size
            g_start = gid * group_size
            g_end = min(g_start + group_size, cols)

            if j == g_start or j == block_start:
                # Compute group params from CURRENT (error-compensated) weights
                group_slice = W[:, g_start:g_end]
                row_min = group_slice.min(axis=1)
                row_max = group_slice.max(axis=1)
                row_range = row_max - row_min
                scale = np.maximum(row_range / (n_levels - 1), MIN_F16_SCALE)
                zero_point = row_min

                # CRITICAL: Round-trip through f16 so Rust exactly matches
                scale_f16 = scale.astype(np.float16)
                zero_f16 = zero_point.astype(np.float16)
                validate_f16_params(scale_f16, zero_f16, "GPTQ group")
                scales_f16[:, gid] = scale_f16
                zeros_f16[:, gid] = zero_f16
                # Use f16-rounded values for quantization
                scale = scale_f16.astype(np.float64)
                zero_point = zero_f16.astype(np.float64)

            w_col = W[:, j]
            q_col = np.clip(np.round((w_col - zero_point) / scale), 0, n_levels - 1).astype(np.uint8)
            codes[:, j] = q_col
            w_hat = q_col.astype(np.float64) * scale + zero_point

            err = (w_col - w_hat) / H_inv[j, j]
            err_block[:, j - block_start] = err

            if j + 1 < block_end:
                W[:, j+1:block_end] -= np.outer(err, H_inv[j, j+1:block_end])

        if block_end < cols:
            W[:, block_end:] -= err_block @ H_inv[block_start:block_end, block_end:]

    # Compute MSE using f16-rounded params (matches Rust dequant exactly)
    total_sq_err = 0.0
    for gid in range(num_groups):
        c_start = gid * group_size
        c_end = min(c_start + group_size, cols)
        s = scales_f16[:, gid].astype(np.float64)
        z = zeros_f16[:, gid].astype(np.float64)
        for c in range(c_start, c_end):
            recon = codes[:, c].astype(np.float64) * s + z
            total_sq_err += np.sum((weight_np[:, c] - recon) ** 2)
    mse = total_sq_err / (rows * cols)

    return codes, scales_f16, zeros_f16, mse


def quantize_embedding_8bit(weight_np, group_size=64):
    """8-bit uniform quantization of embedding matrix.

    Returns:
        codes: np.ndarray[uint8] shape [vocab_size, hidden_size]
        scales: np.ndarray[float16] shape [vocab_size, num_groups]
        zeros: np.ndarray[float16] shape [vocab_size, num_groups]
    """
    vocab_size, hidden_size = weight_np.shape
    num_groups = (hidden_size + group_size - 1) // group_size
    n_levels = 256  # 8-bit

    codes = np.zeros((vocab_size, hidden_size), dtype=np.uint8)
    scales = np.zeros((vocab_size, num_groups), dtype=np.float16)
    zeros = np.zeros((vocab_size, num_groups), dtype=np.float16)

    for gid in range(num_groups):
        c_start = gid * group_size
        c_end = min(c_start + group_size, hidden_size)
        group_data = weight_np[:, c_start:c_end].astype(np.float64)

        row_min = group_data.min(axis=1)
        row_max = group_data.max(axis=1)
        row_range = row_max - row_min
        scale = np.maximum(row_range / (n_levels - 1), MIN_F16_SCALE)

        # Round-trip through f16
        scale_f16 = scale.astype(np.float16)
        zero_f16 = row_min.astype(np.float16)
        validate_f16_params(scale_f16, zero_f16, "embedding group")
        scales[:, gid] = scale_f16
        zeros[:, gid] = zero_f16

        # Quantize using f16-rounded params
        scale_64 = scale_f16.astype(np.float64)
        zero_64 = zero_f16.astype(np.float64)

        for c in range(c_start, c_end):
            q = np.clip(np.round((weight_np[:, c].astype(np.float64) - zero_64) / scale_64), 0, 255).astype(np.uint8)
            codes[:, c] = q

    return codes, scales, zeros


# =============================================================================
# Binary format packing
# =============================================================================

def pack_nibbles(codes):
    """Pack uint8 codes (0-15) into nibble pairs: low nibble = even col, high = odd."""
    rows, cols = codes.shape
    assert cols % 2 == 0, f"cols must be even, got {cols}"
    packed = np.zeros((rows, cols // 2), dtype=np.uint8)
    for c in range(0, cols, 2):
        packed[:, c // 2] = (codes[:, c] & 0x0F) | ((codes[:, c + 1] & 0x0F) << 4)
    return packed


def pack_group_params(scales, zeros):
    """Pack per-row-per-group f16 scales and zeros into bytes.

    Layout: for each row r, for each group g: [f16_scale, f16_zero] = 4 bytes.
    """
    rows, num_groups = scales.shape
    buf = bytearray(rows * num_groups * 4)
    for r in range(rows):
        for g in range(num_groups):
            off = (r * num_groups + g) * 4
            s_bytes = np.float16(scales[r, g]).tobytes()
            z_bytes = np.float16(zeros[r, g]).tobytes()
            buf[off:off+2] = s_bytes
            buf[off+2:off+4] = z_bytes
    return bytes(buf)


def pack_linear_section(codes, scales, zeros, rows, cols):
    """Pack a quantized linear layer: sub-header + group_params + nibble_data."""
    data = bytearray()
    # Sub-header: [u32 rows, u32 cols]
    data.extend(struct.pack('<II', rows, cols))
    # Group params
    data.extend(pack_group_params(scales, zeros))
    # Nibble data
    packed = pack_nibbles(codes)
    data.extend(packed.tobytes())
    return bytes(data)


def pack_norm_section(weights_np):
    """Pack RMSNorm weights as f32."""
    return weights_np.astype(np.float32).tobytes()


def write_header(f, config, num_sections):
    """Write 64-byte header."""
    header = bytearray(64)
    # Magic
    header[0:4] = b'RAIM'
    # Version
    struct.pack_into('<I', header, 4, 1)
    # Config
    struct.pack_into('<I', header, 8, config['hidden_size'])
    struct.pack_into('<I', header, 12, config['num_layers'])
    struct.pack_into('<I', header, 16, config['num_heads'])
    struct.pack_into('<I', header, 20, config['num_kv_heads'])
    struct.pack_into('<I', header, 24, config['head_dim'])
    struct.pack_into('<I', header, 28, config['intermediate_size'])
    struct.pack_into('<I', header, 32, config['vocab_size'])
    struct.pack_into('<I', header, 36, config['max_context'])
    struct.pack_into('<f', header, 40, config['rope_theta'])
    struct.pack_into('<f', header, 44, config['norm_eps'])
    header[48] = config['bits']
    header[49] = config['group_size']
    header[50] = config['embed_bits']
    header[51] = config['embed_group_size']
    struct.pack_into('<I', header, 52, num_sections)
    f.write(header)


# =============================================================================
# Main export pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Export HuggingFace LLM to .raimodel')
    parser.add_argument('--model', type=str, required=True,
                       help='HuggingFace model name or path (e.g. HuggingFaceTB/SmolLM-135M, mistralai/Mistral-7B-Instruct-v0.3)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output .raimodel file path (auto-derived from model name if not specified)')
    parser.add_argument('--bits', type=int, default=4, help='Weight bits (default 4)')
    parser.add_argument('--group-size', type=int, default=128, help='GPTQ group size')
    parser.add_argument('--embed-bits', type=int, default=8, help='Embedding bits')
    parser.add_argument('--embed-group-size', type=int, default=64, help='Embedding group size')
    parser.add_argument('--cal-chunks', type=int, default=128, help='Calibration chunks')
    parser.add_argument('--seq-len', type=int, default=2048, help='Calibration sequence length')
    parser.add_argument('--max-context', type=int, default=2048, help='Max context length')
    parser.add_argument('--hessian-dtype', type=str, default='float64', choices=['float32', 'float64'],
                       help='Hessian accumulation dtype (float32 halves RAM for 7B+)')
    args = parser.parse_args()
    validate_export_options(parser, args)

    # Auto-derive output name from model
    if args.output is None:
        model_short = args.model.split('/')[-1].lower().replace(' ', '-')
        args.output = f"{model_short}-q{args.bits}.raimodel"
        print(f"Output: {args.output}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_name = args.model
    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.is_fast:
        raise RuntimeError(
            "this exporter requires a fast tokenizer that can emit tokenizer.json"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    cfg = model.config
    arch = getattr(cfg, 'model_type', 'unknown')
    supported = ['llama', 'mistral']
    if arch not in supported:
        print(f"WARNING: Architecture '{arch}' not explicitly supported (expected one of {supported}).")
        print(f"         Attempting export anyway — model must use LLaMA-style layers.")

    n_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, 'num_key_value_heads', num_heads)
    head_dim = hidden_size // num_heads
    intermediate_size = cfg.intermediate_size
    vocab_size = cfg.vocab_size
    rope_theta = getattr(cfg, 'rope_theta', 10000.0)
    norm_eps = getattr(cfg, 'rms_norm_eps', 1e-5)

    print(f"Architecture: {arch}, {n_layers} layers, hidden={hidden_size}, inter={intermediate_size}")
    print(f"Heads: {num_heads} query, {num_kv_heads} KV, head_dim={head_dim}")
    print(f"Vocab: {vocab_size}, RoPE theta={rope_theta}, norm eps={norm_eps}")
    model_params = sum(p.numel() for p in model.parameters())
    fp16_size = model_params * 2 / 1e9
    q4_est = model_params * 0.5 / 1e6  # 4-bit = 0.5 bytes per param
    print(f"Parameters: {model_params/1e6:.0f}M, FP16={fp16_size:.1f}GB, est Q4={q4_est:.0f}MB")

    # =========================================================================
    # Step 1: Collect Hessians
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 1: COLLECTING HESSIANS")
    print(f"{'='*60}")

    dataset_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    train_text = "\n\n".join(dataset_train["text"])
    train_tokens = tokenizer(train_text, return_tensors="pt")["input_ids"][0]

    chunks = []
    for i in range(args.cal_chunks):
        start = i * args.seq_len
        if start + args.seq_len > len(train_tokens):
            break
        chunks.append(train_tokens[start:start + args.seq_len].unsqueeze(0).to(device))
    print(f"Calibration: {len(chunks)} chunks of {args.seq_len} tokens")

    hess_dtype_np = np.float32 if args.hessian_dtype == 'float32' else np.float64
    hess_dtype_torch = torch.float32  # always accumulate in float32 on GPU
    print(f"Hessian dtype: {args.hessian_dtype} (numpy), float32 (GPU accumulation)")

    hessians = {}
    hooks = []

    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]
        for key_suffix, module in [
            (f"L{layer_idx}_qkv", layer.self_attn.q_proj),
            (f"L{layer_idx}_o", layer.self_attn.o_proj),
            (f"L{layer_idx}_gu", layer.mlp.gate_proj),
            (f"L{layer_idx}_down", layer.mlp.down_proj),
        ]:
            hessians[key_suffix] = None
            def make_hook(k):
                def hook_fn(module, inp, out):
                    x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                    h = x.T @ x
                    if hessians[k] is None:
                        hessians[k] = h.cpu()
                    else:
                        hessians[k] += h.cpu()
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(key_suffix)))

    print(f"Registered {len(hooks)} hooks")
    t0 = time.time()
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            model(chunk)
            if (i + 1) % 16 == 0:
                print(f"  Chunk {i+1}/{len(chunks)}")
    t_cal = time.time() - t0
    print(f"Calibration done in {t_cal:.1f}s")

    for h in hooks:
        h.remove()

    n_tokens = len(chunks) * args.seq_len
    for key in hessians:
        if hessians[key] is not None:
            hessians[key] = (hessians[key] / n_tokens).numpy().astype(hess_dtype_np)
            if args.hessian_dtype == 'float32':
                # GPTQ needs float64 for Cholesky, upcast just before quantization
                pass

    # =========================================================================
    # Step 2: GPTQ quantize all layers
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 2: GPTQ-{args.bits}BIT QUANTIZATION")
    print(f"{'='*60}")

    t_quant_start = time.time()

    # Store quantized layers as (list of 7 (codes, scales, zeros, rows, cols)), plus norm weights
    layer_data = []

    for layer_idx in range(n_layers):
        t0 = time.time()
        layer = model.model.layers[layer_idx]

        linear_map = [
            ("q_proj", f"L{layer_idx}_qkv", layer.self_attn.q_proj),
            ("k_proj", f"L{layer_idx}_qkv", layer.self_attn.k_proj),
            ("v_proj", f"L{layer_idx}_qkv", layer.self_attn.v_proj),
            ("o_proj", f"L{layer_idx}_o", layer.self_attn.o_proj),
            ("gate_proj", f"L{layer_idx}_gu", layer.mlp.gate_proj),
            ("up_proj", f"L{layer_idx}_gu", layer.mlp.up_proj),
            ("down_proj", f"L{layer_idx}_down", layer.mlp.down_proj),
        ]

        linears_packed = []
        for name, hkey, linear in linear_map:
            w = linear.weight.data.float().cpu().numpy().astype(np.float64)
            H = hessians[hkey].astype(np.float64)  # GPTQ Cholesky needs float64
            codes, scales, zeros, mse = gptq_quantize_to_codes(
                w, H, bits=args.bits, group_size=args.group_size
            )
            rows, cols = w.shape
            linears_packed.append((codes, scales, zeros, rows, cols))
            print(f"  L{layer_idx}.{name}: [{rows}x{cols}] mse={mse:.2e}")

        # Extract norm weights
        input_ln = layer.input_layernorm.weight.data.float().cpu().numpy()
        post_attn_ln = layer.post_attention_layernorm.weight.data.float().cpu().numpy()

        layer_data.append((linears_packed, input_ln, post_attn_ln))
        elapsed = time.time() - t0
        print(f"  Layer {layer_idx}: {elapsed:.1f}s")

    t_quant = time.time() - t_quant_start
    print(f"\nQuantization done in {t_quant:.1f}s ({t_quant/60:.1f} min)")

    # =========================================================================
    # Step 3: Quantize embedding
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 3: EMBEDDING QUANTIZATION (8-bit)")
    print(f"{'='*60}")

    embed_weight = model.model.embed_tokens.weight.data.float().cpu().numpy().astype(np.float64)
    print(f"Embedding shape: {embed_weight.shape}")
    t0 = time.time()
    embed_codes, embed_scales, embed_zeros = quantize_embedding_8bit(
        embed_weight, group_size=args.embed_group_size
    )
    t_embed = time.time() - t0
    # Compute embedding MSE
    num_embed_groups = (hidden_size + args.embed_group_size - 1) // args.embed_group_size
    embed_mse_sum = 0.0
    for gid in range(num_embed_groups):
        c_start = gid * args.embed_group_size
        c_end = min(c_start + args.embed_group_size, hidden_size)
        s = embed_scales[:, gid].astype(np.float64)
        z = embed_zeros[:, gid].astype(np.float64)
        for c in range(c_start, c_end):
            recon = embed_codes[:, c].astype(np.float64) * s + z
            embed_mse_sum += np.sum((embed_weight[:, c] - recon) ** 2)
    embed_mse = embed_mse_sum / (vocab_size * hidden_size)
    print(f"Embedding 8-bit MSE: {embed_mse:.2e}, time: {t_embed:.1f}s")

    # Final norm
    final_norm_weight = model.model.norm.weight.data.float().cpu().numpy()

    # =========================================================================
    # Step 4: Write .raimodel file
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 4: WRITING .raimodel FILE")
    print(f"{'='*60}")

    output_path = Path(args.output)

    # Prepare all section data first
    num_sections = 1 + n_layers + 1  # embed + layers + final_norm = 32

    # Build section data
    sections_data = []

    # Section 0: Embedding
    embed_section = bytearray()
    embed_section.extend(pack_group_params(embed_scales, embed_zeros))
    embed_section.extend(embed_codes.tobytes())
    sections_data.append(bytes(embed_section))

    # Sections 1..30: Transformer layers
    for layer_idx in range(n_layers):
        linears_packed, input_ln, post_attn_ln = layer_data[layer_idx]
        layer_section = bytearray()
        for codes, scales, zeros, rows, cols in linears_packed:
            layer_section.extend(pack_linear_section(codes, scales, zeros, rows, cols))
        layer_section.extend(pack_norm_section(input_ln))
        layer_section.extend(pack_norm_section(post_attn_ln))
        sections_data.append(bytes(layer_section))

    # Section 31: Final RMSNorm
    sections_data.append(pack_norm_section(final_norm_weight))

    # Compute offsets
    header_size = 64
    table_size = num_sections * 16
    data_start = header_size + table_size

    offsets = []
    current_offset = data_start
    for data in sections_data:
        offsets.append(current_offset)
        current_offset += len(data)

    model_config = {
        'hidden_size': hidden_size,
        'num_layers': n_layers,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'intermediate_size': intermediate_size,
        'vocab_size': vocab_size,
        'max_context': args.max_context,
        'rope_theta': rope_theta,
        'norm_eps': norm_eps,
        'bits': args.bits,
        'group_size': args.group_size,
        'embed_bits': args.embed_bits,
        'embed_group_size': args.embed_group_size,
    }

    with open(output_path, 'wb') as f:
        # Header
        write_header(f, model_config, num_sections)

        # Section index table
        for i in range(num_sections):
            f.write(struct.pack('<QQ', offsets[i], len(sections_data[i])))

        # Section data
        for data in sections_data:
            f.write(data)

    total_size = output_path.stat().st_size
    print(f"\nWrote: {output_path} ({total_size / 1e6:.1f} MB)")

    # Section breakdown
    print("\nSection sizes:")
    print(f"  Header + index: {data_start / 1024:.1f} KB")
    print(f"  Embedding (8-bit): {len(sections_data[0]) / 1e6:.1f} MB")
    for i in range(n_layers):
        print(f"  Layer {i}: {len(sections_data[1+i]) / 1024:.1f} KB")
    print(f"  Final norm: {len(sections_data[-1])} bytes")

    # =========================================================================
    # Step 5: Copy tokenizer
    # =========================================================================
    tokenizer_dst = output_path.parent / "tokenizer.json"
    with tempfile.TemporaryDirectory(prefix="rai_tokenizer_") as tmp_dir:
        tokenizer.save_pretrained(tmp_dir)
        src_json = Path(tmp_dir) / "tokenizer.json"
        if src_json.exists():
            shutil.copy2(src_json, tokenizer_dst)
            print(f"Tokenizer copied to: {tokenizer_dst}")
        else:
            raise RuntimeError("tokenizer export did not produce the required tokenizer.json")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Source model: {model_name}")
    print(f"Model file:   {output_path} ({total_size / 1e6:.1f} MB)")
    print(f"Tokenizer:    {tokenizer_dst}")
    print(f"Config:       {args.bits}-bit weights, {args.embed_bits}-bit embeddings")
    print(f"Group sizes:  weights={args.group_size}, embed={args.embed_group_size}")
    print(f"Cal time:     {t_cal:.1f}s")
    print(f"Quant time:   {t_quant:.1f}s")
    print(f"\nTo generate text:")
    print(f"  cargo build -p rai-infer --release")
    print(f"  ./target/release/rai-generate \\")
    print(f"    --model {output_path} \\")
    print(f"    --tokenizer {tokenizer_dst} \\")
    print(f"    --prompt \"The future of AI is\" \\")
    print(f"    --max-tokens 64")


if __name__ == "__main__":
    main()
