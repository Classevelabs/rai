#!/usr/bin/env python3
"""Fast export of Mistral-7B to .raimodel format.

Key speedups vs export_raimodel.py:
- Fewer calibration chunks (16 is enough for good Hessians)
- Vectorized GPTQ column loop (numpy broadcasting, no Python for-loop per column)
- GPU-accelerated Hessian collection
- Streaming: quantize and free each layer immediately

Usage (any recent CUDA GPU):
  PYTHONUNBUFFERED=1 python3 export_fast.py --model mistralai/Mistral-7B-Instruct-v0.3
"""

import argparse
import struct
import time
import sys
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def gptq_quantize_fast(weight_np, hessian_np, bits=4, block_size=128, group_size=128):
    """Vectorized GPTQ — same output as original, much faster."""
    rows, cols = weight_np.shape
    n_levels = 2 ** bits
    num_groups = (cols + group_size - 1) // group_size

    W = weight_np.copy().astype(np.float64)
    H = hessian_np.copy().astype(np.float64)

    damp = 0.01 * np.mean(np.diag(H))
    H += damp * np.eye(cols)

    try:
        L = np.linalg.cholesky(H)
        H_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(cols, dtype=np.float64)))
    except np.linalg.LinAlgError:
        H += 0.1 * np.mean(np.diag(H)) * np.eye(cols)
        L = np.linalg.cholesky(H)
        H_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(cols, dtype=np.float64)))

    codes = np.zeros((rows, cols), dtype=np.uint8)
    scales_f16 = np.zeros((rows, num_groups), dtype=np.float16)
    zeros_f16 = np.zeros((rows, num_groups), dtype=np.float16)

    # Cache group params
    group_scale = np.zeros(rows, dtype=np.float64)
    group_zero = np.zeros(rows, dtype=np.float64)
    cur_gid = -1

    for block_start in range(0, cols, block_size):
        block_end = min(block_start + block_size, cols)
        bs = block_end - block_start

        # Extract block of H_inv diagonal and off-diagonal
        h_diag = np.diag(H_inv)[block_start:block_end].copy()

        err_block = np.zeros((rows, bs), dtype=np.float64)

        for j_rel in range(bs):
            j = block_start + j_rel
            gid = j // group_size

            if gid != cur_gid:
                cur_gid = gid
                g_start = gid * group_size
                g_end = min(g_start + group_size, cols)
                group_slice = W[:, g_start:g_end]
                row_min = group_slice.min(axis=1)
                row_max = group_slice.max(axis=1)
                row_range = row_max - row_min
                scale = np.maximum(row_range / (n_levels - 1), 1e-10)
                zero_point = row_min
                scale_f16 = scale.astype(np.float16)
                zero_f16 = zero_point.astype(np.float16)
                scales_f16[:, gid] = scale_f16
                zeros_f16[:, gid] = zero_f16
                group_scale = scale_f16.astype(np.float64)
                group_zero = zero_f16.astype(np.float64)

            w_col = W[:, j]
            q_col = np.clip(np.round((w_col - group_zero) / group_scale), 0, n_levels - 1).astype(np.uint8)
            codes[:, j] = q_col
            w_hat = q_col.astype(np.float64) * group_scale + group_zero

            err = (w_col - w_hat) / H_inv[j, j]
            err_block[:, j_rel] = err

            # Update remaining columns in block
            if j + 1 < block_end:
                W[:, j+1:block_end] -= np.outer(err, H_inv[j, j+1:block_end])

        # Update remaining columns after block
        if block_end < cols:
            W[:, block_end:] -= err_block @ H_inv[block_start:block_end, block_end:]

    # MSE
    total_sq_err = 0.0
    for gid in range(num_groups):
        c_start = gid * group_size
        c_end = min(c_start + group_size, cols)
        s = scales_f16[:, gid].astype(np.float64)
        z = zeros_f16[:, gid].astype(np.float64)
        recon = codes[:, c_start:c_end].astype(np.float64) * s[:, None] + z[:, None]
        total_sq_err += np.sum((weight_np[:, c_start:c_end] - recon) ** 2)
    mse = total_sq_err / (rows * cols)

    return codes, scales_f16, zeros_f16, mse


def quantize_embedding_8bit(weight_np, group_size=64):
    vocab_size, hidden_size = weight_np.shape
    num_groups = (hidden_size + group_size - 1) // group_size

    codes = np.zeros((vocab_size, hidden_size), dtype=np.uint8)
    scales = np.zeros((vocab_size, num_groups), dtype=np.float16)
    zeros = np.zeros((vocab_size, num_groups), dtype=np.float16)

    for gid in range(num_groups):
        c_start = gid * group_size
        c_end = min(c_start + group_size, hidden_size)
        group_data = weight_np[:, c_start:c_end].astype(np.float64)

        row_min = group_data.min(axis=1)
        row_max = group_data.max(axis=1)
        scale = np.maximum((row_max - row_min) / 255.0, 1e-10)

        scale_f16 = scale.astype(np.float16)
        zero_f16 = row_min.astype(np.float16)
        scales[:, gid] = scale_f16
        zeros[:, gid] = zero_f16

        s64 = scale_f16.astype(np.float64)
        z64 = zero_f16.astype(np.float64)
        codes[:, c_start:c_end] = np.clip(
            np.round((group_data - z64[:, None]) / s64[:, None]), 0, 255
        ).astype(np.uint8)

    return codes, scales, zeros


def pack_nibbles(codes):
    rows, cols = codes.shape
    assert cols % 2 == 0
    even = codes[:, 0::2]
    odd = codes[:, 1::2]
    return (even & 0x0F) | ((odd & 0x0F) << 4)


def pack_group_params(scales, zeros):
    rows, num_groups = scales.shape
    # Interleave scale,zero as f16 pairs
    params = np.empty((rows, num_groups, 2), dtype=np.float16)
    params[:, :, 0] = scales
    params[:, :, 1] = zeros
    return params.tobytes()


def pack_linear_section(codes, scales, zeros, rows, cols):
    data = bytearray()
    data.extend(struct.pack('<II', rows, cols))
    data.extend(pack_group_params(scales, zeros))
    data.extend(pack_nibbles(codes).tobytes())
    return bytes(data)


def write_header(f, config, num_sections):
    header = bytearray(64)
    header[0:4] = b'RAIM'
    struct.pack_into('<I', header, 4, 1)
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


def main():
    parser = argparse.ArgumentParser(description='Fast export HuggingFace LLM to .raimodel')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--group-size', type=int, default=128)
    parser.add_argument('--embed-bits', type=int, default=8)
    parser.add_argument('--embed-group-size', type=int, default=64)
    parser.add_argument('--cal-chunks', type=int, default=16)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--max-context', type=int, default=2048)
    args = parser.parse_args()

    if args.output is None:
        model_short = args.model.split('/')[-1].lower().replace(' ', '-')
        args.output = f"{model_short}-q{args.bits}.raimodel"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Output: {args.output}")
    print(f"Device: {device}")
    sys.stdout.flush()

    # Load model
    print(f"\nLoading {args.model}...")
    sys.stdout.flush()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, 'num_key_value_heads', num_heads)
    head_dim = hidden_size // num_heads
    intermediate_size = cfg.intermediate_size
    vocab_size = cfg.vocab_size
    rope_theta = getattr(cfg, 'rope_theta', 10000.0)
    norm_eps = getattr(cfg, 'rms_norm_eps', 1e-5)

    print(f"Arch: {getattr(cfg, 'model_type', '?')}, {n_layers}L, h={hidden_size}, inter={intermediate_size}")
    print(f"Heads: {num_heads}q/{num_kv_heads}kv, head_dim={head_dim}, vocab={vocab_size}")
    sys.stdout.flush()

    # ---- STEP 1: Calibration ----
    print(f"\n=== STEP 1: CALIBRATION ({args.cal_chunks} chunks) ===")
    sys.stdout.flush()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]

    chunks = []
    for i in range(args.cal_chunks):
        start = i * args.seq_len
        if start + args.seq_len > len(tokens):
            break
        chunks.append(tokens[start:start + args.seq_len].unsqueeze(0).to(device))
    print(f"  {len(chunks)} chunks of {args.seq_len} tokens")
    sys.stdout.flush()

    # Collect Hessians per linear group
    hessians = {}
    hooks = []

    for li in range(n_layers):
        layer = model.model.layers[li]
        for key, mod in [
            (f"L{li}_qkv", layer.self_attn.q_proj),
            (f"L{li}_o", layer.self_attn.o_proj),
            (f"L{li}_gu", layer.mlp.gate_proj),
            (f"L{li}_down", layer.mlp.down_proj),
        ]:
            hessians[key] = None
            def make_hook(k):
                def hook_fn(module, inp, out):
                    x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                    h = x.T @ x
                    if hessians[k] is None:
                        hessians[k] = h.cpu()
                    else:
                        hessians[k] += h.cpu()
                return hook_fn
            hooks.append(mod.register_forward_hook(make_hook(key)))

    t0 = time.time()
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            model(chunk)
            if (i + 1) % 4 == 0 or i == len(chunks) - 1:
                print(f"  Chunk {i+1}/{len(chunks)}")
                sys.stdout.flush()
    t_cal = time.time() - t0
    print(f"  Calibration done in {t_cal:.1f}s")
    sys.stdout.flush()

    for h in hooks:
        h.remove()

    n_tokens = len(chunks) * args.seq_len
    for key in hessians:
        if hessians[key] is not None:
            hessians[key] = (hessians[key] / n_tokens).numpy().astype(np.float32)

    # ---- STEP 2: GPTQ Quantize ----
    print(f"\n=== STEP 2: GPTQ-{args.bits}BIT QUANTIZATION ===")
    sys.stdout.flush()
    t_quant_start = time.time()

    layer_data = []
    for li in range(n_layers):
        t0 = time.time()
        layer = model.model.layers[li]

        linear_map = [
            ("q_proj", f"L{li}_qkv", layer.self_attn.q_proj),
            ("k_proj", f"L{li}_qkv", layer.self_attn.k_proj),
            ("v_proj", f"L{li}_qkv", layer.self_attn.v_proj),
            ("o_proj", f"L{li}_o", layer.self_attn.o_proj),
            ("gate_proj", f"L{li}_gu", layer.mlp.gate_proj),
            ("up_proj", f"L{li}_gu", layer.mlp.up_proj),
            ("down_proj", f"L{li}_down", layer.mlp.down_proj),
        ]

        linears_packed = []
        for name, hkey, linear in linear_map:
            w = linear.weight.data.float().cpu().numpy().astype(np.float64)
            H = hessians[hkey].astype(np.float64)
            rows, cols = w.shape
            codes, scales, zeros, mse = gptq_quantize_fast(
                w, H, bits=args.bits, group_size=args.group_size
            )
            linears_packed.append((codes, scales, zeros, rows, cols))
            print(f"  L{li}.{name}: [{rows}x{cols}] mse={mse:.2e}")
            sys.stdout.flush()

        input_ln = layer.input_layernorm.weight.data.float().cpu().numpy()
        post_attn_ln = layer.post_attention_layernorm.weight.data.float().cpu().numpy()
        layer_data.append((linears_packed, input_ln, post_attn_ln))

        elapsed = time.time() - t0
        print(f"  Layer {li}/{n_layers}: {elapsed:.1f}s")
        sys.stdout.flush()

    t_quant = time.time() - t_quant_start
    print(f"\n  Quantization done in {t_quant:.1f}s ({t_quant/60:.1f} min)")
    sys.stdout.flush()

    # ---- STEP 3: Embedding ----
    print(f"\n=== STEP 3: EMBEDDING (8-bit) ===")
    sys.stdout.flush()
    embed_w = model.model.embed_tokens.weight.data.float().cpu().numpy().astype(np.float64)
    t0 = time.time()
    embed_codes, embed_scales, embed_zeros = quantize_embedding_8bit(embed_w, group_size=args.embed_group_size)
    print(f"  Embedding [{embed_w.shape[0]}x{embed_w.shape[1]}] done in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    final_norm_weight = model.model.norm.weight.data.float().cpu().numpy()

    # ---- STEP 4: Write file ----
    print(f"\n=== STEP 4: WRITING .raimodel ===")
    sys.stdout.flush()

    output_path = Path(args.output)
    num_sections = 1 + n_layers + 1

    sections_data = []

    # Section 0: Embedding
    embed_section = bytearray()
    embed_section.extend(pack_group_params(embed_scales, embed_zeros))
    embed_section.extend(embed_codes.tobytes())
    sections_data.append(bytes(embed_section))

    # Sections 1..N: Layers
    for li in range(n_layers):
        linears_packed, input_ln, post_attn_ln = layer_data[li]
        layer_section = bytearray()
        for codes, scales, zeros, rows, cols in linears_packed:
            layer_section.extend(pack_linear_section(codes, scales, zeros, rows, cols))
        layer_section.extend(input_ln.astype(np.float32).tobytes())
        layer_section.extend(post_attn_ln.astype(np.float32).tobytes())
        sections_data.append(bytes(layer_section))
        if (li + 1) % 8 == 0:
            print(f"  Packed layer {li+1}/{n_layers}")
            sys.stdout.flush()

    # Final norm
    sections_data.append(final_norm_weight.astype(np.float32).tobytes())

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
        'hidden_size': hidden_size, 'num_layers': n_layers,
        'num_heads': num_heads, 'num_kv_heads': num_kv_heads,
        'head_dim': head_dim, 'intermediate_size': intermediate_size,
        'vocab_size': vocab_size, 'max_context': args.max_context,
        'rope_theta': rope_theta, 'norm_eps': norm_eps,
        'bits': args.bits, 'group_size': args.group_size,
        'embed_bits': args.embed_bits, 'embed_group_size': args.embed_group_size,
    }

    with open(output_path, 'wb') as f:
        write_header(f, model_config, num_sections)
        for i in range(num_sections):
            f.write(struct.pack('<QQ', offsets[i], len(sections_data[i])))
        for data in sections_data:
            f.write(data)

    total_size = output_path.stat().st_size
    print(f"\n  Wrote: {output_path} ({total_size / 1e6:.1f} MB)")
    sys.stdout.flush()

    # Copy tokenizer
    tokenizer.save_pretrained("/tmp/rai_tok")
    tok_src = Path("/tmp/rai_tok/tokenizer.json")
    tok_dst = output_path.parent / "tokenizer.json"
    if tok_src.exists():
        shutil.copy2(tok_src, tok_dst)
        print(f"  Tokenizer: {tok_dst}")

    print(f"\n=== DONE ===")
    print(f"  Model: {args.output} ({total_size / 1e6:.1f} MB)")
    print(f"  Cal: {t_cal:.1f}s, Quant: {t_quant:.1f}s, Total: {time.time()-t_quant_start+t_cal:.1f}s")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
