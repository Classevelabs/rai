#!/usr/bin/env python3
"""Ultra-fast RTN (Round-to-Nearest) export to .raimodel.

No calibration, no Hessian, no GPTQ column loop.
Just per-group min/max quantization — finishes in ~5 min for 7B.
Quality: ~0.3 perplexity worse than GPTQ, but still very usable.

Usage:
  PYTHONUNBUFFERED=1 python3 export_rtn.py --model mistralai/Mistral-7B-Instruct-v0.3
"""

import argparse, struct, time, sys, shutil
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def rtn_quantize(weight_np, bits=4, group_size=128):
    """Simple round-to-nearest quantization per group."""
    rows, cols = weight_np.shape
    n_levels = 2 ** bits
    num_groups = (cols + group_size - 1) // group_size

    codes = np.zeros((rows, cols), dtype=np.uint8)
    scales_f16 = np.zeros((rows, num_groups), dtype=np.float16)
    zeros_f16 = np.zeros((rows, num_groups), dtype=np.float16)

    for gid in range(num_groups):
        c_start = gid * group_size
        c_end = min(c_start + group_size, cols)
        group = weight_np[:, c_start:c_end].astype(np.float64)

        row_min = group.min(axis=1)
        row_max = group.max(axis=1)
        scale = np.maximum((row_max - row_min) / (n_levels - 1), 1e-10)

        sf16 = scale.astype(np.float16)
        zf16 = row_min.astype(np.float16)
        scales_f16[:, gid] = sf16
        zeros_f16[:, gid] = zf16

        s64 = sf16.astype(np.float64)
        z64 = zf16.astype(np.float64)

        codes[:, c_start:c_end] = np.clip(
            np.round((group - z64[:, None]) / s64[:, None]), 0, n_levels - 1
        ).astype(np.uint8)

    # MSE
    total_err = 0.0
    for gid in range(num_groups):
        c_start = gid * group_size
        c_end = min(c_start + group_size, cols)
        s = scales_f16[:, gid].astype(np.float64)
        z = zeros_f16[:, gid].astype(np.float64)
        recon = codes[:, c_start:c_end].astype(np.float64) * s[:, None] + z[:, None]
        total_err += np.sum((weight_np[:, c_start:c_end] - recon) ** 2)
    mse = total_err / (rows * cols)
    return codes, scales_f16, zeros_f16, mse


def quantize_embedding_8bit(weight_np, group_size=64):
    rows, cols = weight_np.shape
    num_groups = (cols + group_size - 1) // group_size
    codes = np.zeros((rows, cols), dtype=np.uint8)
    scales = np.zeros((rows, num_groups), dtype=np.float16)
    zeros = np.zeros((rows, num_groups), dtype=np.float16)

    for gid in range(num_groups):
        c_start = gid * group_size
        c_end = min(c_start + group_size, cols)
        group = weight_np[:, c_start:c_end].astype(np.float64)
        row_min = group.min(axis=1)
        row_max = group.max(axis=1)
        scale = np.maximum((row_max - row_min) / 255.0, 1e-10)
        sf16 = scale.astype(np.float16)
        zf16 = row_min.astype(np.float16)
        scales[:, gid] = sf16
        zeros[:, gid] = zf16
        s64 = sf16.astype(np.float64)
        z64 = zf16.astype(np.float64)
        codes[:, c_start:c_end] = np.clip(
            np.round((group - z64[:, None]) / s64[:, None]), 0, 255
        ).astype(np.uint8)
    return codes, scales, zeros


def pack_nibbles(codes):
    even = codes[:, 0::2]
    odd = codes[:, 1::2]
    return (even & 0x0F) | ((odd & 0x0F) << 4)


def pack_group_params(scales, zeros):
    rows, ng = scales.shape
    params = np.empty((rows, ng, 2), dtype=np.float16)
    params[:, :, 0] = scales
    params[:, :, 1] = zeros
    return params.tobytes()


def pack_linear(codes, scales, zeros, rows, cols):
    data = bytearray()
    data.extend(struct.pack('<II', rows, cols))
    data.extend(pack_group_params(scales, zeros))
    data.extend(pack_nibbles(codes).tobytes())
    return bytes(data)


def write_header(f, c, ns):
    h = bytearray(64)
    h[0:4] = b'RAIM'
    struct.pack_into('<I', h, 4, 1)
    struct.pack_into('<I', h, 8, c['hidden_size'])
    struct.pack_into('<I', h, 12, c['num_layers'])
    struct.pack_into('<I', h, 16, c['num_heads'])
    struct.pack_into('<I', h, 20, c['num_kv_heads'])
    struct.pack_into('<I', h, 24, c['head_dim'])
    struct.pack_into('<I', h, 28, c['intermediate_size'])
    struct.pack_into('<I', h, 32, c['vocab_size'])
    struct.pack_into('<I', h, 36, c['max_context'])
    struct.pack_into('<f', h, 40, c['rope_theta'])
    struct.pack_into('<f', h, 44, c['norm_eps'])
    h[48] = c['bits']; h[49] = c['group_size']
    h[50] = c['embed_bits']; h[51] = c['embed_group_size']
    struct.pack_into('<I', h, 52, ns)
    f.write(h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--group-size', type=int, default=128)
    parser.add_argument('--max-context', type=int, default=2048)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.model.split('/')[-1].lower()}-q{args.bits}.raimodel"

    print(f"Output: {args.output}"); sys.stdout.flush()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model}..."); sys.stdout.flush()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s"); sys.stdout.flush()

    cfg = model.config
    nl = cfg.num_hidden_layers
    hs = cfg.hidden_size
    nh = cfg.num_attention_heads
    nkv = getattr(cfg, 'num_key_value_heads', nh)
    hd = hs // nh
    inter = cfg.intermediate_size
    vs = cfg.vocab_size
    rope = getattr(cfg, 'rope_theta', 10000.0)
    eps = getattr(cfg, 'rms_norm_eps', 1e-5)
    print(f"{getattr(cfg,'model_type','?')}: {nl}L h={hs} inter={inter} heads={nh}/{nkv} vocab={vs}")
    sys.stdout.flush()

    # Quantize all layers
    print(f"\n=== RTN-{args.bits}BIT QUANTIZATION ==="); sys.stdout.flush()
    t_start = time.time()
    layer_data = []

    for li in range(nl):
        t0 = time.time()
        layer = model.model.layers[li]
        linears = [
            ("q_proj", layer.self_attn.q_proj),
            ("k_proj", layer.self_attn.k_proj),
            ("v_proj", layer.self_attn.v_proj),
            ("o_proj", layer.self_attn.o_proj),
            ("gate_proj", layer.mlp.gate_proj),
            ("up_proj", layer.mlp.up_proj),
            ("down_proj", layer.mlp.down_proj),
        ]
        packed = []
        for name, lin in linears:
            w = lin.weight.data.float().cpu().numpy()
            r, c = w.shape
            codes, scales, zeros, mse = rtn_quantize(w, bits=args.bits, group_size=args.group_size)
            packed.append((codes, scales, zeros, r, c))
            if name in ("q_proj", "down_proj"):
                print(f"  L{li}.{name}: [{r}x{c}] mse={mse:.2e}")
                sys.stdout.flush()

        iln = layer.input_layernorm.weight.data.float().cpu().numpy()
        pln = layer.post_attention_layernorm.weight.data.float().cpu().numpy()
        layer_data.append((packed, iln, pln))

        # Free GPU memory for this layer
        for name, lin in linears:
            lin.weight.data = torch.empty(0)
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"  Layer {li}/{nl}: {elapsed:.1f}s"); sys.stdout.flush()

    t_quant = time.time() - t_start
    print(f"\nQuantization: {t_quant:.1f}s ({t_quant/60:.1f} min)"); sys.stdout.flush()

    # Embedding
    print(f"\n=== EMBEDDING 8-BIT ==="); sys.stdout.flush()
    t0 = time.time()
    ew = model.model.embed_tokens.weight.data.float().cpu().numpy()
    ec, es, ez = quantize_embedding_8bit(ew)
    print(f"  Embedding [{ew.shape[0]}x{ew.shape[1]}]: {time.time()-t0:.1f}s"); sys.stdout.flush()

    fnorm = model.model.norm.weight.data.float().cpu().numpy()

    # Check if lm_head is separate (untied)
    tied = getattr(cfg, 'tie_word_embeddings', True)
    lm_head_packed = None
    if not tied:
        print(f"\n=== LM_HEAD 4-BIT (untied) ==="); sys.stdout.flush()
        t0 = time.time()
        lmw = model.lm_head.weight.data.float().cpu().numpy()
        r, c = lmw.shape
        lm_codes, lm_scales, lm_zeros, lm_mse = rtn_quantize(lmw, bits=args.bits, group_size=args.group_size)
        lm_head_packed = (lm_codes, lm_scales, lm_zeros, r, c)
        print(f"  lm_head [{r}x{c}] mse={lm_mse:.2e}: {time.time()-t0:.1f}s"); sys.stdout.flush()
    else:
        print(f"\n  lm_head: tied to embedding"); sys.stdout.flush()

    # Write
    print(f"\n=== WRITING ==="); sys.stdout.flush()
    outpath = Path(args.output)
    # Sections: embed + layers + norm [+ lm_head if untied]
    nsec = 1 + nl + 1 + (1 if lm_head_packed else 0)
    sections = []

    # Embed section
    esec = bytearray()
    esec.extend(pack_group_params(es, ez))
    esec.extend(ec.tobytes())
    sections.append(bytes(esec))

    # Layer sections
    for li in range(nl):
        packed, iln, pln = layer_data[li]
        lsec = bytearray()
        for codes, scales, zeros, r, c in packed:
            lsec.extend(pack_linear(codes, scales, zeros, r, c))
        lsec.extend(iln.astype(np.float32).tobytes())
        lsec.extend(pln.astype(np.float32).tobytes())
        sections.append(bytes(lsec))

    # Final norm section
    sections.append(fnorm.astype(np.float32).tobytes())

    # lm_head section (if untied)
    if lm_head_packed:
        codes, scales, zeros, r, c = lm_head_packed
        sections.append(pack_linear(codes, scales, zeros, r, c))

    # Offsets
    hdr = 64; tbl = nsec * 16; dstart = hdr + tbl
    offsets = []; cur = dstart
    for s in sections:
        offsets.append(cur); cur += len(s)

    mc = {
        'hidden_size': hs, 'num_layers': nl, 'num_heads': nh,
        'num_kv_heads': nkv, 'head_dim': hd, 'intermediate_size': inter,
        'vocab_size': vs, 'max_context': args.max_context,
        'rope_theta': rope, 'norm_eps': eps,
        'bits': args.bits, 'group_size': args.group_size,
        'embed_bits': 8, 'embed_group_size': 64,
    }

    with open(outpath, 'wb') as f:
        write_header(f, mc, nsec)
        for i in range(nsec):
            f.write(struct.pack('<QQ', offsets[i], len(sections[i])))
        for s in sections:
            f.write(s)

    sz = outpath.stat().st_size
    print(f"\nWrote: {outpath} ({sz/1e6:.1f} MB)"); sys.stdout.flush()

    # Tokenizer
    tokenizer.save_pretrained("/tmp/rai_tok")
    ts = Path("/tmp/rai_tok/tokenizer.json")
    td = outpath.parent / "tokenizer.json"
    if ts.exists():
        shutil.copy2(ts, td)
        print(f"Tokenizer: {td}")

    total = time.time() - t_start
    print(f"\n=== DONE in {total:.1f}s ({total/60:.1f} min) ===")
    print(f"  {outpath} ({sz/1e6:.1f} MB)")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
