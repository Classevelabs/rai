#!/usr/bin/env python3
"""Reference perplexity for an unquantized HuggingFace checkpoint.

This is the number `rai perplexity` is measured against. Quantization cost is
the difference between the two, so the two have to be computed the same way —
otherwise the difference measures the methods, not the weights.

The method is `rai perplexity`'s, which is `llama-perplexity`'s:

  1. Tokenize the whole file once, no special tokens.
  2. Non-overlapping chunks of --context tokens.
  3. One forward pass per chunk, from no cached state.
  4. Score only the second half of each chunk, so every scored token has at
     least context/2 tokens of real context behind it.
  5. perplexity = exp(mean NLL over every scored token).

The accumulator is float64 for the same reason the Rust one is: a float32 sum
of six-figure magnitude loses the low bits of every later addition, which shows
up as a perplexity that drifts with corpus length.

Usage:
  python reference_ppl.py --model <hf-dir> --text <file> [--context 512]
                          [--max-chunks N] [--dtype float16|float32] [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HuggingFace checkpoint directory")
    parser.add_argument("--text", required=True, help="UTF-8 text file to score")
    parser.add_argument("--context", type=int, default=512, help="tokens per chunk")
    parser.add_argument("--max-chunks", type=int, default=0, help="0 = whole file")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help="reference precision; float16 is the checkpoint's own storage dtype",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dump-topk",
        type=int,
        default=0,
        help="also record the top-K next-token distribution at each scored position",
    )
    parser.add_argument("--dump-path", help="where to write the --dump-topk record")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.dump_topk and not args.dump_path:
        parser.error("--dump-topk requires --dump-path")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.context < 2:
        parser.error("--context must be at least 2 tokens")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=None
    )
    model.to(args.device)
    model.eval()

    with open(args.text, "r", encoding="utf-8") as handle:
        text = handle.read()

    # add_special_tokens=False: continuous prose cut into chunks. A BOS at every
    # chunk boundary would score a position the text does not contain, and the
    # Rust side does not insert one either.
    token_ids = tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]
    total_chunks = len(token_ids) // args.context
    if total_chunks == 0:
        print(
            f"the text is {len(token_ids)} tokens, fewer than one "
            f"{args.context}-token chunk",
            file=sys.stderr,
        )
        return 1
    chunks = total_chunks if args.max_chunks == 0 else min(total_chunks, args.max_chunks)

    first_scored = args.context // 2
    nll_sum = 0.0
    # Sum of squares, for the standard error. A perplexity is a sample mean;
    # quoting one to four decimals without its error is decoration. Matches the
    # Rust side so the two reports are read the same way.
    nll_sq_sum = 0.0
    scored = 0

    # Byte-for-byte the container `rai perplexity --dump-topk` writes, so
    # scripts/compare_topk.py reads both with one parser. Little-endian
    # throughout; see the TopKWriter doc comment in cli/perplexity.rs.
    dump = None
    if args.dump_topk:
        dump = open(args.dump_path, "wb")
        # Must match TOPK_MAGIC in rai-infer/src/cli/perplexity.rs byte for byte.
        dump.write(b"RAITOPK\x00")
        dump.write(struct.pack("<I", 1))                       # version
        dump.write(struct.pack("<I", args.dump_topk))          # k
        dump.write(struct.pack("<Q", 0))                       # positions, patched on close
        dump.write(struct.pack("<I", int(model.config.vocab_size)))
        dump.write(struct.pack("<I", 0))                       # reserved

    started = time.perf_counter()

    with torch.inference_mode():
        for chunk in range(chunks):
            window = token_ids[chunk * args.context : (chunk + 1) * args.context]
            ids = torch.tensor([window], dtype=torch.long, device=args.device)

            logits = model(ids).logits[0]  # [context, vocab]

            # Position i predicts token i+1; the final position has no target
            # inside the chunk, exactly as in the Rust implementation.
            predicted = logits[first_scored : args.context - 1]
            targets = ids[0, first_scored + 1 : args.context]

            # log_softmax in float32 even when the model ran in float16: the
            # reduction over a 150k vocabulary is where half precision would
            # actually cost accuracy, and the reference must not be the noisy
            # side of this comparison.
            log_probs = torch.log_softmax(predicted.float(), dim=-1)
            chunk_nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

            chunk_nll_f64 = chunk_nll.double()
            nll_sum += float(chunk_nll_f64.sum().item())
            nll_sq_sum += float((chunk_nll_f64 * chunk_nll_f64).sum().item())
            scored += int(targets.numel())

            if dump is not None:
                k = args.dump_topk
                values, indices = torch.topk(log_probs, k, dim=-1)
                values = values.cpu().numpy()
                indices = indices.cpu().numpy().astype("<u4")
                target_lp = (-chunk_nll).float().cpu().numpy()
                targets_np = targets.cpu().numpy().astype("<u4")
                for row in range(indices.shape[0]):
                    dump.write(struct.pack("<I", int(targets_np[row])))
                    dump.write(struct.pack("<f", float(target_lp[row])))
                    for column in range(k):
                        dump.write(struct.pack("<I", int(indices[row, column])))
                        dump.write(struct.pack("<f", float(values[row, column])))

            if not args.json:
                running = math.exp(nll_sum / scored)
                print(
                    f"\r  chunk {chunk + 1}/{chunks}   ppl {running:.4f}   ",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )

    if dump is not None:
        # The count is only known at the end; the header reserved room for it.
        dump.seek(16)
        dump.write(struct.pack("<Q", scored))
        dump.close()

    elapsed = time.perf_counter() - started
    mean_nll = nll_sum / scored
    # Clamped at zero: the subtraction is catastrophic cancellation when the
    # variance is small, and a NaN standard error reads as if it meant nothing
    # rather than as if it were broken.
    variance = max(0.0, nll_sq_sum / scored - mean_nll * mean_nll)
    nll_stderr = math.sqrt(variance / scored)
    evaluated = chunks * args.context

    report = {
        "perplexity": round(math.exp(mean_nll), 4),
        "nll": round(mean_nll, 4),
        "tokens_scored": scored,
        "tokens_total": evaluated,
        "chunks": chunks,
        "context": args.context,
        "elapsed_ms": int(elapsed * 1000),
        "tokens_per_second": round(evaluated / elapsed, 4),
        "nll_stderr": round(nll_stderr, 4),
        "perplexity_stderr_pct": round((math.exp(nll_stderr) - 1.0) * 100.0, 4),
        "dtype": args.dtype,
        "device": args.device,
        "topk_dump": args.dump_path if args.dump_topk else None,
        "method": "llama.cpp-compatible: non-overlapping chunks, second half scored",
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("\n", file=sys.stderr)
        print(f"perplexity   {report['perplexity']:.2f}  "
              f"(+/- {report['perplexity_stderr_pct']:.2f}% standard error)")
        print(f"nll          {report['nll']:.4f} +/- {nll_stderr:.4f} nats/token")
        print(f"scored       {scored} tokens over {chunks} chunks of {args.context}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
