#!/usr/bin/env python3
"""Ultra-fast RTN (Round-to-Nearest) export to .raimodel.

No calibration, no Hessian, no GPTQ column loop.
Just per-group min/max quantization — finishes in ~5 min for 7B.
Quality: ~0.3 perplexity worse than GPTQ, but still very usable.

The quantizers and container writers are shared with the GPTQ exporters via
raimodel.py; see its docstring for the binary layout.

Usage:
  PYTHONUNBUFFERED=1 python3 export_rtn.py --model mistralai/Mistral-7B-Instruct-v0.3
"""

import argparse, time, sys, tempfile
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import raimodel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--group-size', type=int, default=128)
    parser.add_argument('--max-context', type=int, default=2048)
    args = parser.parse_args()
    raimodel.validate_export_options(parser, args)

    if args.output is None:
        args.output = f"{args.model.split('/')[-1].lower()}-q{args.bits}.raimodel"

    print(f"Output: {args.output}"); sys.stdout.flush()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model}..."); sys.stdout.flush()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.is_fast:
        raise RuntimeError(
            "this exporter requires a fast tokenizer that can emit tokenizer.json"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s"); sys.stdout.flush()

    cfg = model.config
    nl = cfg.num_hidden_layers
    hs = cfg.hidden_size
    nh = cfg.num_attention_heads
    nkv = getattr(cfg, 'num_key_value_heads', nh)
    hd = raimodel.resolve_head_dim(getattr(cfg, 'head_dim', None), hs, nh)
    inter = cfg.intermediate_size
    vs = cfg.vocab_size
    rope = getattr(cfg, 'rope_theta', 10000.0)
    eps = getattr(cfg, 'rms_norm_eps', 1e-5)

    mc = {
        'hidden_size': hs, 'num_layers': nl, 'num_heads': nh,
        'num_kv_heads': nkv, 'head_dim': hd, 'intermediate_size': inter,
        'vocab_size': vs, 'max_context': args.max_context,
        'rope_theta': rope, 'norm_eps': eps,
        'bits': args.bits, 'group_size': args.group_size,
        'embed_bits': 8, 'embed_group_size': 64,
    }
    # Fail fast on anything the Rust reader would reject.
    raimodel.validate_model_config(mc)

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
            codes, scales, zeros, mse = raimodel.rtn_quantize(
                w, bits=args.bits, group_size=args.group_size, label=f"L{li}.{name}"
            )
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
    ec, es, ez, emse = raimodel.quantize_embedding_8bit(ew)
    print(f"  Embedding [{ew.shape[0]}x{ew.shape[1]}] mse={emse:.2e}: {time.time()-t0:.1f}s"); sys.stdout.flush()

    fnorm = model.model.norm.weight.data.float().cpu().numpy()

    # Check if lm_head is separate (untied)
    tied = getattr(cfg, 'tie_word_embeddings', True)
    lm_head_packed = None
    if not tied:
        print(f"\n=== LM_HEAD 4-BIT (untied) ==="); sys.stdout.flush()
        t0 = time.time()
        lmw = model.lm_head.weight.data.float().cpu().numpy()
        r, c = lmw.shape
        lm_codes, lm_scales, lm_zeros, lm_mse = raimodel.rtn_quantize(
            lmw, bits=args.bits, group_size=args.group_size, label="lm_head"
        )
        lm_head_packed = (lm_codes, lm_scales, lm_zeros, r, c)
        print(f"  lm_head [{r}x{c}] mse={lm_mse:.2e}: {time.time()-t0:.1f}s"); sys.stdout.flush()
    else:
        print(f"\n  lm_head: tied to embedding"); sys.stdout.flush()

    # Write
    print(f"\n=== WRITING ==="); sys.stdout.flush()
    outpath = Path(args.output)

    # Sections: embed + layers + norm [+ lm_head if untied]
    sections = []
    sections.append(raimodel.build_embedding_section(ec, es, ez))
    for li in range(nl):
        packed, iln, pln = layer_data[li]
        sections.append(raimodel.build_layer_section(packed, iln, pln))
    sections.append(raimodel.pack_norm_section(fnorm, "final norm"))
    if lm_head_packed is not None:
        codes, scales, zeros, r, c = lm_head_packed
        sections.append(raimodel.pack_linear_section(codes, scales, zeros, r, c))

    sz = raimodel.write_raimodel(outpath, mc, sections)
    print(f"\nWrote: {outpath} ({sz/1e6:.1f} MB, {len(sections)} sections)"); sys.stdout.flush()

    # Tokenizer
    td = outpath.parent / "tokenizer.json"
    with tempfile.TemporaryDirectory(prefix="rai_tokenizer_") as tmp_dir:
        tokenizer.save_pretrained(tmp_dir)
        ts = Path(tmp_dir) / "tokenizer.json"
        if not ts.exists():
            raise RuntimeError("tokenizer export did not produce the required tokenizer.json")
        if raimodel.copy_tokenizer_json(ts, td):
            print(f"Tokenizer: {td}")
        else:
            print(f"Tokenizer already present (identical): {td}")

    total = time.time() - t_start
    print(f"\n=== DONE in {total:.1f}s ({total/60:.1f} min) ===")
    print(f"  {outpath} ({sz/1e6:.1f} MB)")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
