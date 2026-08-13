#!/usr/bin/env python3
"""Fast export of Mistral-7B to .raimodel format.

Key speedups vs export_raimodel.py:
- Fewer calibration chunks (16 is enough for good Hessians)
- GPU-accelerated Hessian collection
- Frequent progress output for long-running steps

The quantizers and container writers are shared with the other exporters via
raimodel.py; see its docstring for the binary layout.

Usage (any recent CUDA GPU):
  PYTHONUNBUFFERED=1 python3 export_fast.py --model mistralai/Mistral-7B-Instruct-v0.3
"""

import argparse
import time
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

import raimodel


def main():
    parser = argparse.ArgumentParser(description='Fast export HuggingFace LLM to .raimodel')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--group-size', type=int, default=128)
    parser.add_argument('--embed-bits', type=int, default=8)
    parser.add_argument('--embed-group-size', type=int, default=64)
    parser.add_argument('--cal-chunks', type=int, default=16)
    # datasets >= 5 requires a namespaced repo id: a bare "wikitext" is rejected.
    parser.add_argument('--calibration-dataset', type=str, default='Salesforce/wikitext')
    parser.add_argument('--calibration-config', type=str, default='wikitext-2-raw-v1')
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--max-context', type=int, default=2048)
    args = parser.parse_args()
    raimodel.validate_export_options(parser, args)

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
    model = raimodel.load_hf_causal_lm(args.model)
    model.to(device)  # calibration runs where the chunks are
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.is_fast:
        raise RuntimeError(
            "this exporter requires a fast tokenizer that can emit tokenizer.json"
        )
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
    head_dim = raimodel.resolve_head_dim(getattr(cfg, 'head_dim', None), hidden_size, num_heads)
    intermediate_size = cfg.intermediate_size
    vocab_size = cfg.vocab_size
    rope_theta = getattr(cfg, 'rope_theta', 10000.0)
    norm_eps = getattr(cfg, 'rms_norm_eps', 1e-5)
    tied = getattr(cfg, 'tie_word_embeddings', True)

    model_config = {
        'hidden_size': hidden_size, 'num_layers': n_layers,
        'num_heads': num_heads, 'num_kv_heads': num_kv_heads,
        'head_dim': head_dim, 'intermediate_size': intermediate_size,
        'vocab_size': vocab_size, 'max_context': args.max_context,
        'rope_theta': rope_theta, 'norm_eps': norm_eps,
        'bits': args.bits, 'group_size': args.group_size,
        'embed_bits': args.embed_bits, 'embed_group_size': args.embed_group_size,
    }
    # Fail fast on anything the Rust reader would reject, BEFORE calibration.
    raimodel.validate_model_config(model_config)
    raimodel.assert_exportable_architecture(model, cfg, args.max_context)

    print(f"Arch: {getattr(cfg, 'model_type', '?')}, {n_layers}L, h={hidden_size}, inter={intermediate_size}")
    print(f"Heads: {num_heads}q/{num_kv_heads}kv, head_dim={head_dim}, vocab={vocab_size}")
    print(f"Embeddings: {'tied' if tied else 'untied (separate lm_head)'}")
    sys.stdout.flush()

    # ---- STEP 1: Calibration ----
    print(f"\n=== STEP 1: CALIBRATION ({args.cal_chunks} chunks) ===")
    sys.stdout.flush()

    ds = load_dataset(args.calibration_dataset, args.calibration_config, split="train")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]

    chunks = []
    for i in range(args.cal_chunks):
        start = i * args.seq_len
        if start + args.seq_len > len(tokens):
            break
        chunks.append(tokens[start:start + args.seq_len].unsqueeze(0).to(device))
    raimodel.require_calibration_chunks(len(chunks), args.seq_len, len(tokens))
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
            codes, scales, zeros, mse = raimodel.gptq_quantize(
                w, H, bits=args.bits, group_size=args.group_size,
                label=f"L{li}.{name}"
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

    # ---- STEP 3: Embedding (and lm_head when untied) ----
    print(f"\n=== STEP 3: EMBEDDING (8-bit) ===")
    sys.stdout.flush()
    embed_w = model.model.embed_tokens.weight.data.float().cpu().numpy().astype(np.float64)
    t0 = time.time()
    embed_codes, embed_scales, embed_zeros, embed_mse = raimodel.quantize_embedding_8bit(
        embed_w, group_size=args.embed_group_size
    )
    print(f"  Embedding [{embed_w.shape[0]}x{embed_w.shape[1]}] mse={embed_mse:.2e} done in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    final_norm_weight = model.model.norm.weight.data.float().cpu().numpy()

    lm_head_packed = None
    if not tied:
        print(f"\n=== STEP 3b: LM_HEAD {args.bits}-BIT (untied) ===")
        sys.stdout.flush()
        t0 = time.time()
        lm_w = model.lm_head.weight.data.float().cpu().numpy().astype(np.float64)
        lm_rows, lm_cols = lm_w.shape
        lm_codes, lm_scales, lm_zeros, lm_mse = raimodel.rtn_quantize(
            lm_w, bits=args.bits, group_size=args.group_size, label="lm_head"
        )
        lm_head_packed = (lm_codes, lm_scales, lm_zeros, lm_rows, lm_cols)
        print(f"  lm_head [{lm_rows}x{lm_cols}] mse={lm_mse:.2e}: {time.time()-t0:.1f}s")
        sys.stdout.flush()
    else:
        print(f"  lm_head: tied to embedding")
        sys.stdout.flush()

    # ---- STEP 4: Write file ----
    print(f"\n=== STEP 4: WRITING .raimodel ===")
    sys.stdout.flush()

    output_path = Path(args.output)

    sections_data = []
    sections_data.append(raimodel.build_embedding_section(embed_codes, embed_scales, embed_zeros))
    for li in range(n_layers):
        linears_packed, input_ln, post_attn_ln = layer_data[li]
        sections_data.append(raimodel.build_layer_section(linears_packed, input_ln, post_attn_ln))
        if (li + 1) % 8 == 0:
            print(f"  Packed layer {li+1}/{n_layers}")
            sys.stdout.flush()
    sections_data.append(raimodel.pack_norm_section(final_norm_weight, "final norm"))
    if lm_head_packed is not None:
        codes, scales, zeros, rows, cols = lm_head_packed
        sections_data.append(raimodel.pack_linear_section(codes, scales, zeros, rows, cols))

    total_size = raimodel.write_raimodel(output_path, model_config, sections_data)
    print(f"\n  Wrote: {output_path} ({total_size / 1e6:.1f} MB, {len(sections_data)} sections)")
    sys.stdout.flush()

    # Copy tokenizer
    tok_dst = output_path.parent / "tokenizer.json"
    with tempfile.TemporaryDirectory(prefix="rai_tokenizer_") as tmp_dir:
        tokenizer.save_pretrained(tmp_dir)
        tok_src = Path(tmp_dir) / "tokenizer.json"
        if not tok_src.exists():
            raise RuntimeError("tokenizer export did not produce the required tokenizer.json")
        if raimodel.copy_tokenizer_json(tok_src, tok_dst):
            print(f"  Tokenizer: {tok_dst}")
        else:
            print(f"  Tokenizer already present (identical): {tok_dst}")

    print(f"\n=== DONE ===")
    print(f"  Model: {args.output} ({total_size / 1e6:.1f} MB)")
    print(f"  Cal: {t_cal:.1f}s, Quant: {t_quant:.1f}s, Total: {time.time()-t_quant_start+t_cal:.1f}s")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
