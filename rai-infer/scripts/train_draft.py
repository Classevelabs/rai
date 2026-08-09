#!/usr/bin/env python3
"""Train a small draft model for speculative decoding via knowledge distillation.

Creates a tiny (2-layer, hidden=1024) model with the SAME tokenizer as the teacher,
trained to match the teacher's output distribution. The draft model is fast enough
for high-throughput speculative decoding on CPU.

Architecture: 2-layer MistralForCausalLM (same arch, just smaller)
  - Embedding: 32768 × 1024 → 33.5M params
  - 2 transformer layers → 33M params
  - lm_head: 32768 × 1024 → 33.5M params
  - Total: ~100M parameters before quantization

Runtime size, latency, acceptance rate, and speculative-decoding throughput depend on
the exported representation, target model, hardware, prompt distribution, and sampler.
Measure them on the intended deployment; this script does not guarantee a speedup.

Usage (any recent CUDA GPU):
  python3 train_draft.py --teacher mistralai/Mistral-7B-Instruct-v0.3 --epochs 2

Then export:
  python3 export_rtn.py --model ./mistral-draft-100m --output mistral-draft-100m-q4.raimodel
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MistralConfig,
    MistralForCausalLM,
)

# Save only top-K teacher logits per position to avoid OOM.
# K=128 captures >99% of probability mass; uses ~400 MB vs 84 GB for full vocab.
TEACHER_TOP_K = 128

# Diverse prompts for teacher text generation
PROMPT_TEMPLATES = [
    "The quick brown fox",
    "In the year 2024,",
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time in a",
    "The scientific method involves",
    "According to recent studies,",
    "The main difference between",
    "To solve this problem, we",
    "In machine learning,",
    "The history of computing",
    "Climate change affects",
    "The recipe for chocolate",
    "In mathematics, a prime",
    "The theory of relativity",
    "When debugging code,",
    "The human brain contains",
    "In economics, supply and",
    "The periodic table of",
    "Shakespeare wrote many",
    "Artificial intelligence is",
    "The solar system contains",
    "In Python, you can use",
    "The French Revolution began",
    "DNA stands for",
    "The Internet was created",
    "In philosophy, ethics is",
    "The speed of light is",
    "To train a neural network,",
    "The Amazon rainforest is",
    "In statistics, the mean",
    "The Great Wall of China",
    "Quantum computing uses",
    "The stock market crashed",
    "In biology, cells are",
    "The Mediterranean Sea is",
    "Machine learning models can",
    "The Renaissance period was",
    "In chemistry, atoms consist",
    "The Olympic Games originated",
]


class TeacherDataset(Dataset):
    """Pre-generated teacher top-K logits dataset."""

    def __init__(self, path):
        with open(os.path.join(path, "metadata.json")) as f:
            self.meta = json.load(f)
        self.input_ids = np.load(os.path.join(path, "input_ids.npy"), mmap_mode="r")
        self.top_k_indices = np.load(
            os.path.join(path, "top_k_indices.npy"), mmap_mode="r"
        )
        self.top_k_logits = np.load(
            os.path.join(path, "top_k_logits.npy"), mmap_mode="r"
        )
        self.attention_mask = np.load(
            os.path.join(path, "attention_mask.npy"), mmap_mode="r"
        )
        self.num_samples = self.input_ids.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        ids = torch.tensor(self.input_ids[idx].copy(), dtype=torch.long)
        indices = torch.tensor(self.top_k_indices[idx].copy(), dtype=torch.long)
        logits = torch.tensor(self.top_k_logits[idx].copy(), dtype=torch.float32)
        mask = torch.tensor(self.attention_mask[idx].copy(), dtype=torch.float32)
        return ids, indices, logits, mask


def generate_teacher_data(
    teacher,
    tokenizer,
    num_samples=2000,
    seq_len=256,
    batch_size=8,
    output_dir="teacher_data",
):
    """Generate training data from teacher model.

    Phase 1: Use teacher.generate() to create full-length sequences from prompts.
             This gives meaningful tokens at every position (no padding waste).
    Phase 2: Run teacher forward on generated sequences, save top-K logits.

    Memory: num_samples × seq_len × K × 6 bytes ≈ 400 MB (vs 84 GB for full vocab).
    Time: ~10-20 min on a mid-range CUDA GPU (one-time cost, cached to disk).
    """
    os.makedirs(output_dir, exist_ok=True)
    pad_token_id = tokenizer.pad_token_id or 0
    vocab_size = teacher.config.vocab_size
    device = next(teacher.parameters()).device

    # === Phase 1: Generate full-length sequences with teacher ===
    print(f"Phase 1: Generating {num_samples} sequences (seq_len={seq_len})...")
    sys.stdout.flush()

    all_sequences = []
    all_masks = []
    t0 = time.time()

    teacher.eval()
    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_prompts = [
                PROMPT_TEMPLATES[i % len(PROMPT_TEMPLATES)] for i in range(start, end)
            ]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            ).to(device)

            generated = teacher.generate(
                **inputs,
                max_new_tokens=seq_len,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=pad_token_id,
            )

            for seq in generated:
                seq_np = seq.cpu().numpy().astype(np.int32)
                # Truncate to seq_len
                if len(seq_np) > seq_len:
                    seq_np = seq_np[:seq_len]
                # Pad if shorter (rare — only if EOS hit very early)
                elif len(seq_np) < seq_len:
                    pad_len = seq_len - len(seq_np)
                    seq_np = np.concatenate(
                        [seq_np, np.full(pad_len, pad_token_id, dtype=np.int32)]
                    )
                # Mask: mark trailing pad tokens as padding
                mask = np.ones(seq_len, dtype=np.bool_)
                for i in range(seq_len - 1, -1, -1):
                    if seq_np[i] == pad_token_id:
                        mask[i] = False
                    else:
                        break
                all_sequences.append(seq_np)
                all_masks.append(mask)

            done = min(end, num_samples)
            elapsed = time.time() - t0
            if done == num_samples or elapsed > 0:
                eta = elapsed / max(done, 1) * (num_samples - done)
                print(
                    f"  {done}/{num_samples} generated ({elapsed:.0f}s, ETA {eta:.0f}s)"
                )
                sys.stdout.flush()

    all_sequences = np.array(all_sequences[:num_samples])
    all_masks = np.array(all_masks[:num_samples])
    gen_time = time.time() - t0
    real_tokens = all_masks.sum()
    total_tokens = all_masks.size
    print(
        f"Phase 1 done: {gen_time:.0f}s, {real_tokens}/{total_tokens} real tokens "
        f"({100*real_tokens/total_tokens:.0f}%)"
    )
    sys.stdout.flush()

    # === Phase 2: Get teacher logits (fast batched forward) ===
    print(f"Phase 2: Computing teacher logits (top-{TEACHER_TOP_K})...")
    sys.stdout.flush()

    all_top_k_indices = []
    all_top_k_logits = []
    t1 = time.time()

    seq_tensor = torch.tensor(all_sequences, dtype=torch.long)
    mask_tensor = torch.tensor(all_masks, dtype=torch.long)

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            input_ids = seq_tensor[start:end].to(device)
            attention_mask = mask_tensor[start:end].to(device)

            outputs = teacher(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [bs, seq_len, vocab]

            topk = logits.topk(TEACHER_TOP_K, dim=-1)
            all_top_k_indices.append(topk.indices.cpu().numpy().astype(np.int32))
            all_top_k_logits.append(topk.values.cpu().half().numpy())

            done = min(end, num_samples)
            if done % (batch_size * 50) < batch_size or done == num_samples:
                elapsed = time.time() - t1
                eta = elapsed / max(done, 1) * (num_samples - done)
                print(f"  {done}/{num_samples} logits ({elapsed:.0f}s, ETA {eta:.0f}s)")
                sys.stdout.flush()

    all_top_k_indices = np.concatenate(all_top_k_indices, axis=0)
    all_top_k_logits = np.concatenate(all_top_k_logits, axis=0)
    fwd_time = time.time() - t1

    # Save
    np.save(os.path.join(output_dir, "input_ids.npy"), all_sequences)
    np.save(os.path.join(output_dir, "top_k_indices.npy"), all_top_k_indices)
    np.save(os.path.join(output_dir, "top_k_logits.npy"), all_top_k_logits)
    np.save(os.path.join(output_dir, "attention_mask.npy"), all_masks)

    metadata = {
        "num_samples": int(all_sequences.shape[0]),
        "seq_len": seq_len,
        "top_k": TEACHER_TOP_K,
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    data_mb = (all_top_k_indices.nbytes + all_top_k_logits.nbytes) / 1e6
    total_time = gen_time + fwd_time
    print(
        f"Teacher data saved to {output_dir}/ "
        f"({total_time:.0f}s total, {data_mb:.0f} MB on disk)"
    )
    sys.stdout.flush()


def create_draft_model(
    teacher_config, num_layers=2, hidden_size=1024, intermediate_size=4096
):
    """Create a small draft model with the same tokenizer as the teacher."""
    config = MistralConfig(
        vocab_size=teacher_config.vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=8,
        num_key_value_heads=4,  # GQA with 4 KV heads
        max_position_embeddings=teacher_config.max_position_embeddings,
        rms_norm_eps=teacher_config.rms_norm_eps,
        rope_theta=getattr(teacher_config, "rope_theta", 10000.0),
        tie_word_embeddings=False,
        torch_dtype=torch.float16,
    )

    model = MistralForCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Draft model: {total_params/1e6:.1f}M params")
    print(f"  hidden={hidden_size}, layers={num_layers}, inter={intermediate_size}")
    print(f"  heads=8, kv_heads=4, vocab={teacher_config.vocab_size}")
    sys.stdout.flush()
    return model


def init_from_teacher(draft_model, teacher_model, teacher_config, hidden_size=1024):
    """Initialize draft embeddings from teacher via SVD projection."""
    with torch.no_grad():
        # Project teacher embeddings to draft size via truncated SVD
        teacher_embed = teacher_model.model.embed_tokens.weight.data.float()
        U, S, _Vh = torch.linalg.svd(teacher_embed, full_matrices=False)
        # Keep top hidden_size components
        projected = U[:, :hidden_size] * S[:hidden_size].unsqueeze(0)
        draft_model.model.embed_tokens.weight.data.copy_(projected.half())

        # Similarly for lm_head
        teacher_lmh = teacher_model.lm_head.weight.data.float()
        U2, S2, _Vh2 = torch.linalg.svd(teacher_lmh, full_matrices=False)
        projected2 = U2[:, :hidden_size] * S2[:hidden_size].unsqueeze(0)
        draft_model.lm_head.weight.data.copy_(projected2.half())

    print(f"  Initialized embeddings + lm_head from teacher via SVD projection")
    sys.stdout.flush()


def train_distillation(
    draft_model,
    teacher_data_dir,
    batch_size=16,
    epochs=2,
    lr=3e-4,
    temperature=2.0,
    device="cuda",
):
    """Train draft model via knowledge distillation from pre-computed teacher logits.

    Uses top-K teacher logits (reconstructed to full vocab with -inf for non-top-K)
    and attention masking to ignore padding positions in both KL and CE losses.
    """
    dataset = TeacherDataset(teacher_data_dir)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    vocab_size = dataset.meta["vocab_size"]
    top_k = dataset.meta["top_k"]

    draft_model = draft_model.to(device).train()
    optimizer = torch.optim.AdamW(draft_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader)
    )

    print(f"\nTraining: {epochs} epochs, {len(dataset)} samples, bs={batch_size}")
    print(f"  lr={lr}, temperature={temperature}, top_k={top_k}, vocab={vocab_size}")
    sys.stdout.flush()

    for epoch in range(epochs):
        t0 = time.time()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (input_ids, top_k_indices, top_k_logits, attention_mask) in enumerate(loader):
            input_ids = input_ids.to(device)
            top_k_indices = top_k_indices.to(device)
            top_k_logits = top_k_logits.to(device)
            attention_mask = attention_mask.to(device)

            # Reconstruct teacher logits from top-K
            # Non-top-K tokens get -inf → zero probability after softmax
            teacher_logits = torch.full(
                (input_ids.shape[0], input_ids.shape[1], vocab_size),
                float("-inf"),
                device=device,
                dtype=torch.float32,
            )
            teacher_logits.scatter_(-1, top_k_indices, top_k_logits)

            # Draft forward (with attention mask so padding doesn't pollute)
            outputs = draft_model(
                input_ids, attention_mask=attention_mask.long()
            )
            student_logits = outputs.logits.float()

            # KL divergence loss (temperature-scaled), masked for padding
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            kl_per_token = F.kl_div(
                student_log_probs, teacher_probs, reduction="none"
            ).sum(dim=-1)  # [batch, seq_len]
            num_real = attention_mask.sum().clamp(min=1)
            kl_loss = (kl_per_token * attention_mask).sum() / num_real

            # Hard label CE loss (shifted for next-token prediction), masked
            hard_labels = teacher_logits.argmax(dim=-1)
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = hard_labels[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            ce_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_logits.shape[0], -1)  # [batch, seq_len-1]
            num_shift = shift_mask.sum().clamp(min=1)
            ce_loss = (ce_per_token * shift_mask).sum() / num_shift

            loss = temperature**2 * kl_loss + 0.5 * ce_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                avg = total_loss / num_batches
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch+1} [{batch_idx+1}/{len(loader)}] "
                    f"loss={avg:.4f} ({elapsed:.0f}s)"
                )
                sys.stdout.flush()

        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - t0
        print(
            f"Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f} ({epoch_time:.1f}s)"
        )
        sys.stdout.flush()

    return draft_model


def save_draft_model(draft_model, tokenizer, output_dir):
    """Save draft model in HuggingFace format (for export_rtn.py)."""
    os.makedirs(output_dir, exist_ok=True)
    draft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Draft model saved to {output_dir}/")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Train a draft model for speculative decoding"
    )
    parser.add_argument(
        "--teacher",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Teacher model name/path",
    )
    parser.add_argument(
        "--output", default="mistral-draft-100m", help="Output directory"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=1024, help="Draft hidden size"
    )
    parser.add_argument("--num-layers", type=int, default=2, help="Draft layers")
    parser.add_argument(
        "--intermediate-size", type=int, default=4096, help="Draft MLP size"
    )
    parser.add_argument(
        "--num-samples", type=int, default=2000, help="Training samples"
    )
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--temperature", type=float, default=2.0, help="Distillation temperature"
    )
    parser.add_argument(
        "--teacher-data",
        default="teacher_data",
        help="Directory for cached teacher data",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip teacher data generation (use cached data)",
    )
    parser.add_argument(
        "--no-init", action="store_true", help="Skip teacher weight initialization"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    sys.stdout.flush()

    # Load teacher + tokenizer
    print(f"\n=== Loading teacher: {args.teacher} ===")
    sys.stdout.flush()
    t0 = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.float16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher.eval()
    print(f"Teacher loaded in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    # Generate teacher data (if not cached)
    if not args.skip_generation:
        generate_teacher_data(
            teacher,
            tokenizer,
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            output_dir=args.teacher_data,
        )

    # Create draft model
    print(f"\n=== Creating draft model ===")
    sys.stdout.flush()
    draft = create_draft_model(
        teacher.config,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
    )

    # Initialize from teacher
    if not args.no_init:
        print(f"\n=== Initializing from teacher ===")
        sys.stdout.flush()
        init_from_teacher(draft, teacher, teacher.config, args.hidden_size)

    # Free teacher GPU memory
    del teacher
    torch.cuda.empty_cache()

    # Train
    print(f"\n=== Training via knowledge distillation ===")
    sys.stdout.flush()
    draft = train_distillation(
        draft,
        args.teacher_data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        device=device,
    )

    # Save
    print(f"\n=== Saving ===")
    sys.stdout.flush()
    save_draft_model(draft, tokenizer, args.output)

    print(f"\n=== Done! ===")
    print(f"Next steps:")
    print(
        f"  1. Export: python3 export_rtn.py --model {args.output} "
        f"--output {args.output}-q4.raimodel"
    )
    print(
        f"  2. Run: rai-generate --model mistral-7b-q4.raimodel "
        f"--draft {args.output}-q4.raimodel "
        f"--tokenizer tokenizer.json --prompt '...'"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
