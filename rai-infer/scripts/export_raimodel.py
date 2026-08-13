#!/usr/bin/env python3
"""Export any HuggingFace LLaMA/Mistral-family model to .raimodel format.

Supports: SmolLM-135M, Mistral-7B-Instruct-v0.3, LLaMA-2/3, and similar architectures.

Pipeline:
1. Loads model in FP16 on GPU (or CPU)
2. Collects Hessians for all layers in one calibration pass
3. GPTQ-4bit quantizes each linear layer — returns INTEGER CODES + group params
4. Packs codes as nibbles (2 per byte), group params as f16
5. Quantizes embedding at 8-bit uniform (per-row-per-group scale/zero)
6. Quantizes lm_head at 4-bit when the model does not tie word embeddings
7. Writes .raimodel binary file with header + sections
8. Copies tokenizer.json alongside (refusing to clobber a different one)

CRITICAL: Group params are round-tripped through f16 BEFORE computing final codes,
so Rust dequant exactly matches Python: weight = code * f16_scale + f16_zero.

The container format, quantizers, and writers live in raimodel.py (shared with
export_fast.py and export_rtn.py); see its docstring for the binary layout.
"""

import argparse
import time
import tempfile
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

import raimodel


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
    raimodel.validate_export_options(parser, args)

    # Auto-derive output name from model
    if args.output is None:
        model_short = args.model.split('/')[-1].lower().replace(' ', '-')
        args.output = f"{model_short}-q{args.bits}.raimodel"
        print(f"Output: {args.output}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_name = args.model
    print(f"\nLoading {model_name}...")
    model = raimodel.load_hf_causal_lm(model_name)
    model.to(device)  # calibration runs where the chunks are
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
    head_dim = raimodel.resolve_head_dim(getattr(cfg, 'head_dim', None), hidden_size, num_heads)
    intermediate_size = cfg.intermediate_size
    vocab_size = cfg.vocab_size
    rope_theta = getattr(cfg, 'rope_theta', 10000.0)
    norm_eps = getattr(cfg, 'rms_norm_eps', 1e-5)
    tied = getattr(cfg, 'tie_word_embeddings', True)

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
    # Fail fast on anything the Rust reader would reject, BEFORE spending
    # an hour on calibration.
    raimodel.validate_model_config(model_config)
    raimodel.assert_exportable_architecture(model, cfg, args.max_context)

    print(f"Architecture: {arch}, {n_layers} layers, hidden={hidden_size}, inter={intermediate_size}")
    print(f"Heads: {num_heads} query, {num_kv_heads} KV, head_dim={head_dim}")
    print(f"Vocab: {vocab_size}, RoPE theta={rope_theta}, norm eps={norm_eps}")
    print(f"Word embeddings: {'tied' if tied else 'untied (separate lm_head section)'}")
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
    raimodel.require_calibration_chunks(len(chunks), args.seq_len, len(train_tokens))
    print(f"Calibration: {len(chunks)} chunks of {args.seq_len} tokens")

    hess_dtype_np = np.float32 if args.hessian_dtype == 'float32' else np.float64
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
            codes, scales, zeros, mse = raimodel.gptq_quantize(
                w, H, bits=args.bits, group_size=args.group_size,
                label=f"L{layer_idx}.{name}"
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
    # Step 3: Quantize embedding (and lm_head when untied)
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 3: EMBEDDING QUANTIZATION (8-bit)")
    print(f"{'='*60}")

    embed_weight = model.model.embed_tokens.weight.data.float().cpu().numpy().astype(np.float64)
    print(f"Embedding shape: {embed_weight.shape}")
    t0 = time.time()
    embed_codes, embed_scales, embed_zeros, embed_mse = raimodel.quantize_embedding_8bit(
        embed_weight, group_size=args.embed_group_size
    )
    t_embed = time.time() - t0
    print(f"Embedding 8-bit MSE: {embed_mse:.2e}, time: {t_embed:.1f}s")

    # Final norm
    final_norm_weight = model.model.norm.weight.data.float().cpu().numpy()

    # Untied models carry a separate lm_head; the reader expects it as an
    # extra 4-bit section after the final norm.  Quantized round-to-nearest
    # (no Hessian is collected for the lm_head input).
    lm_head_packed = None
    if not tied:
        print(f"\nlm_head is untied — quantizing at {args.bits}-bit")
        t0 = time.time()
        lm_weight = model.lm_head.weight.data.float().cpu().numpy().astype(np.float64)
        lm_rows, lm_cols = lm_weight.shape
        lm_codes, lm_scales, lm_zeros, lm_mse = raimodel.rtn_quantize(
            lm_weight, bits=args.bits, group_size=args.group_size, label="lm_head"
        )
        lm_head_packed = (lm_codes, lm_scales, lm_zeros, lm_rows, lm_cols)
        print(f"lm_head: [{lm_rows}x{lm_cols}] mse={lm_mse:.2e}, time: {time.time()-t0:.1f}s")
    else:
        print(f"\nlm_head: tied to embedding (no extra section)")

    # =========================================================================
    # Step 4: Write .raimodel file
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 4: WRITING .raimodel FILE")
    print(f"{'='*60}")

    output_path = Path(args.output)

    # Build section data: embed + layers + final norm [+ lm_head if untied]
    sections_data = []
    sections_data.append(raimodel.build_embedding_section(embed_codes, embed_scales, embed_zeros))
    for layer_idx in range(n_layers):
        linears_packed, input_ln, post_attn_ln = layer_data[layer_idx]
        sections_data.append(raimodel.build_layer_section(linears_packed, input_ln, post_attn_ln))
    sections_data.append(raimodel.pack_norm_section(final_norm_weight, "final norm"))
    if lm_head_packed is not None:
        codes, scales, zeros, rows, cols = lm_head_packed
        sections_data.append(raimodel.pack_linear_section(codes, scales, zeros, rows, cols))

    num_sections = len(sections_data)
    total_size = raimodel.write_raimodel(output_path, model_config, sections_data)
    print(f"\nWrote: {output_path} ({total_size / 1e6:.1f} MB, {num_sections} sections)")

    # Section breakdown
    data_start = raimodel.HEADER_SIZE + num_sections * raimodel.SECTION_ENTRY_SIZE
    print("\nSection sizes:")
    print(f"  Header + index: {data_start / 1024:.1f} KB")
    print(f"  Embedding (8-bit): {len(sections_data[0]) / 1e6:.1f} MB")
    for i in range(n_layers):
        print(f"  Layer {i}: {len(sections_data[1+i]) / 1024:.1f} KB")
    if lm_head_packed is not None:
        print(f"  Final norm: {len(sections_data[1 + n_layers])} bytes")
        print(f"  lm_head (4-bit): {len(sections_data[-1]) / 1e6:.1f} MB")
    else:
        print(f"  Final norm: {len(sections_data[-1])} bytes")

    # =========================================================================
    # Step 5: Copy tokenizer
    # =========================================================================
    tokenizer_dst = output_path.parent / "tokenizer.json"
    with tempfile.TemporaryDirectory(prefix="rai_tokenizer_") as tmp_dir:
        tokenizer.save_pretrained(tmp_dir)
        src_json = Path(tmp_dir) / "tokenizer.json"
        if not src_json.exists():
            raise RuntimeError("tokenizer export did not produce the required tokenizer.json")
        if raimodel.copy_tokenizer_json(src_json, tokenizer_dst):
            print(f"Tokenizer copied to: {tokenizer_dst}")
        else:
            print(f"Tokenizer already present (identical): {tokenizer_dst}")

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
