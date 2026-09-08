#!/usr/bin/env python3
"""Compare two top-K records: how far did quantization move the model?

Perplexity is one number over a whole corpus, and two models can land on the
same one while disagreeing about individual tokens in opposite directions. This
answers the sharper questions, from records written by
`rai perplexity --dump-topk` and `reference_ppl.py --dump-topk`:

  target_logprob_delta   exact. Both files record the log-probability the model
                         assigned to the token that actually came next, so the
                         mean and the spread of that difference are computed
                         with no approximation at all. This is the honest
                         headline: how much probability mass quantization moved
                         off the right answer.

  top1_agreement         exact. How often the two models would greedily emit
                         the same token. A user running --temperature 0 feels
                         this one directly.

  topk_overlap           exact. Mean size of the intersection of the two top-K
                         sets, over K.

  kl_topk                approximate, and labelled so. True KL needs both full
                         distributions; a full-vocabulary record over 150k
                         entries would be gigabytes. This is the KL restricted
                         to the base model's top-K support, renormalized over
                         it, and `kl_topk_coverage` reports how much of the
                         base model's probability mass that support holds — at
                         K=64 it is normally well over 0.99, and a run where it
                         is not says so rather than quietly reporting a number
                         that means less than it appears to.

Usage:
  python compare_topk.py --base fp16.bin --test rai-q4.bin [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys

# Must match TOPK_MAGIC in rai-infer/src/cli/perplexity.rs byte for byte.
MAGIC = b"RAITOPK\x00"
HEADER = struct.Struct("<8sIIQII")


def read_record(path: str):
    """Parse one record into (k, vocab, [(target, target_lp, {id: lp}), ...])."""
    with open(path, "rb") as handle:
        raw = handle.read()

    if len(raw) < HEADER.size:
        raise SystemExit(f"{path}: too short to hold a header")
    magic, version, k, positions, vocab, _ = HEADER.unpack_from(raw, 0)
    if magic != MAGIC:
        raise SystemExit(f"{path}: not a top-K record (magic {magic!r})")
    if version != 1:
        raise SystemExit(f"{path}: version {version} is not supported")

    entry = struct.Struct("<If" + "If" * k)
    expected = HEADER.size + positions * entry.size
    if len(raw) != expected:
        raise SystemExit(
            f"{path}: header claims {positions} positions "
            f"({expected} bytes) but the file is {len(raw)} bytes; "
            "the writer did not finish"
        )

    rows = []
    for index in range(positions):
        values = entry.unpack_from(raw, HEADER.size + index * entry.size)
        target, target_lp = values[0], values[1]
        rest = values[2:]
        ids = rest[0::2]
        lps = rest[1::2]
        rows.append((target, target_lp, ids, dict(zip(ids, lps))))
    return k, vocab, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="reference record (fp16)")
    parser.add_argument("--test", required=True, help="record under test (4-bit)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    base_k, base_vocab, base_rows = read_record(args.base)
    test_k, test_vocab, test_rows = read_record(args.test)

    if base_k != test_k:
        raise SystemExit(f"K differs: base {base_k}, test {test_k}")
    if len(base_rows) != len(test_rows):
        raise SystemExit(
            f"position count differs: base {len(base_rows)}, test {len(test_rows)}; "
            "the two runs did not score the same corpus"
        )

    deltas = []
    top1_hits = 0
    overlap_total = 0
    kl_total = 0.0
    coverage_total = 0.0

    for index, (base_row, test_row) in enumerate(zip(base_rows, test_rows)):
        base_target, base_lp, base_ids, base_map = base_row
        test_target, test_lp, test_ids, test_map = test_row

        # Both runs must have scored the same token at the same position, or
        # every number below compares unrelated things.
        if base_target != test_target:
            raise SystemExit(
                f"position {index}: base scored token {base_target}, "
                f"test scored {test_target}; the corpora or tokenizers differ"
            )

        deltas.append(test_lp - base_lp)
        if base_ids[0] == test_ids[0]:
            top1_hits += 1
        overlap_total += len(set(base_ids) & set(test_ids))

        # KL(base || test) over the base model's top-K support, renormalized.
        # An id the test model did not rank is floored at its K-th logprob,
        # which is an upper bound on what it actually assigned.
        floor = test_map[test_ids[-1]]
        base_probs = [math.exp(base_map[i]) for i in base_ids]
        mass = sum(base_probs)
        coverage_total += mass
        kl = 0.0
        for token, probability in zip(base_ids, base_probs):
            share = probability / mass
            kl += share * (base_map[token] - test_map.get(token, floor))
        kl_total += kl

    n = len(base_rows)
    mean_delta = sum(deltas) / n
    variance = sum((d - mean_delta) ** 2 for d in deltas) / n
    ordered = sorted(deltas)

    report = {
        "positions": n,
        "k": base_k,
        "target_logprob_delta": {
            "mean": round(mean_delta, 6),
            "stdev": round(math.sqrt(variance), 6),
            "p50": round(ordered[n // 2], 6),
            "p01": round(ordered[max(0, n // 100)], 6),
            "p99": round(ordered[min(n - 1, (99 * n) // 100)], 6),
            "note": "test minus base, in nats; negative means the 4-bit model "
                    "assigned less probability to the correct token",
        },
        "top1_agreement": round(top1_hits / n, 6),
        "topk_overlap": round(overlap_total / (n * base_k), 6),
        "kl_topk": round(kl_total / n, 6),
        "kl_topk_coverage": round(coverage_total / n, 6),
        "kl_topk_note": "KL(base||test) restricted to the base model's top-K, "
                        "renormalized; coverage is the base mass that support holds",
        "base": args.base,
        "test": args.test,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        d = report["target_logprob_delta"]
        print(f"positions              {n}")
        print(f"top-1 agreement        {report['top1_agreement']:.4%}")
        print(f"top-{base_k} overlap          {report['topk_overlap']:.4%}")
        print(f"target logprob delta   {d['mean']:+.6f} nats (median {d['p50']:+.6f})")
        print(f"KL over top-{base_k}          {report['kl_topk']:.6f} nats "
              f"(coverage {report['kl_topk_coverage']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
