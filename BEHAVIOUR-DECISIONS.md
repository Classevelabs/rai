# Why RAI behaves the way it does

Every entry here is a decision that looks arbitrary, slower than it needs to
be, or removable until you know what it cost. Each states the decision, the
symptom that caused it, what it was verified against, and what a user sees if
someone regresses it. The reason also sits in a comment at the code; this file
is the second copy, for people reading the product rather than the file.

---

## 2026-11-16 — Zero points meet the quantized activation sum, not the exact one

**Decision.** The W4A8 kernels apply each group's zero point to the sum of the
*quantized* activations (`scale · Σq`), not to the exact f32 sum
(`rai-infer/src/gemm.rs`).

**Symptom.** With the exact sum, the int8 rounding error was weighted by the
raw 4-bit codes instead of the dequantized weights.

**Verified against.** SmolLM2-135M against the fp32 transformers reference,
10,200 wikitext tokens (`scripts/reference_ppl.py`, `scripts/compare_topk.py`):
perplexity 29.71 to 27.34, KL over the top 64 tokens 0.500 to 0.422, top-1
agreement 62.8% to 66.6%, same speed.

**If regressed.** Every model is measurably worse, with no error anywhere:
perplexity and top-1 agreement drift back by the amounts above.

## 2026-11-16 — `exp` is a range-reduced polynomial, not a bit trick

**Decision.** Softmax and SiLU use Cody-Waite range reduction with a Cephes
polynomial, within 2 ulp (`rai-infer/src/layers.rs`).

**Symptom.** Schraudolph's bit trick was off by up to 3.9% on `exp` and 4.2% on
attention weights.

**Verified against.** The layer unit tests' accuracy checks against `f32::exp`,
and the perplexity run above.

**If regressed.** Attention spreads weight a few percent wrong at every
position; quality drops slightly and the perplexity gate shows it.

## 2026-11-16 — End tokens are control tokens, never ordinary vocabulary

**Decision.** A reply ends on the checkpoint's own end-of-sequence ids and on
end markers found among the tokenizer's *added* tokens
(`rai-infer/src/chat_format.rs`, `END_MARKERS`).

**Symptom.** Qwen2.5's byte-level vocabulary holds `</s>` as an ordinary merged
token (id 128247); a fixed list looked up through the whole vocabulary ended
any reply that wrote `</s>`, and Gemma's `<eos>` and Llama 3.1's `<|eom_id|>`
never ended anything.

**Verified against.** `chat_format` tests (`end_ids_come_from_control_tokens_and_the_record`)
and the real Qwen2.5 and SmolLM2 tokenizers.

**If regressed.** Replies stop mid-sentence on HTML or markup, or run on to
`--max-tokens` after the model has finished.

## 2026-11-16 — The checkpoint's own chat template, recorded at conversion

**Decision.** Conversion writes `<name>.chat.json` beside the model with the
checkpoint's Jinja template, its two special tokens and its end ids;
`--chat-template auto` renders it with minijinja exactly as transformers does
(`trim_blocks`, `lstrip_blocks`, `raise_exception`, `strftime_now`). Built-in
templates never spell the start token: the tokenizer adds it, as
`tokenizer(text)` does.

**Symptom.** Vocabulary probing sent TinyLlama-Chat, Zephyr and Mistral
v0.1/v0.2 to the few-shot prompt, the built-in ChatML lacked the default system
prompts Qwen2.5 and SmolLM2 ship, and raw prompts went to Llama-family models
without the start token they were trained with.

**Verified against.** Prompt text and token ids identical to
`apply_chat_template` on Qwen2.5-0.5B and SmolLM2-135M over four conversations
(multi-turn, whitespace, Unicode, literal `{{ }}` in user text); the
`chat_format` and `convert_matches_python` tests.

**If regressed.** Chat models answer in the wrong format, ignore instructions,
or print nothing because they end the turn at once.

## 2026-11-16 — The repetition penalty covers the reply only

**Decision.** `rai run` and `rai serve` penalize tokens the reply has produced,
never the prompt (`cli/run.rs`, `cli/serve.rs`).

**Symptom.** Penalizing the prompt penalized the template's own end-of-turn
marker, which every chat prompt contains, so replies struggled to end.

**Verified against.** The server checks against SmolLM2 (`stop_reason` "eos" on
a merged thread); the help text always said "already generated".

**If regressed.** Chat replies run long and end at `max_tokens` instead of at
the end of the answer.

## 2026-11-16 — `rai serve` reuses the shared prompt prefix and trims by exchange

**Decision.** The server keeps the K/V of the prefix a request shares with the
previous one (`Loaded::cached_tokens`), trims an over-long thread at user turns
only, by binary search, and merges consecutive messages from one role.

**Symptom.** Every turn re-prefilled the whole thread token by token (7.1 s of
prefill on turn five of a Qwen2.5-0.5B chat), trimming could open on an orphaned
reply that Gemma's template refuses, and Studio sends two user turns in a row
after a failed reply.

**Verified against.** Turn-five prefill 0.3 s; `serve.rs` tests
(`thread_trimming_drops_whole_exchanges_and_never_the_last_turn`,
`consecutive_turns_from_one_role_become_one`).

**If regressed.** Long chats get slower every turn, or a Gemma chat fails with a
template error once it outgrows the window.

## 2026-11-16 — A client that leaves stops its generation

**Decision.** Between prefill chunks and between tokens the server peeks the
connection (`cli/http.rs` `client_gone`) and stops when the client has closed
it.

**Symptom.** Studio's Stop, or a dropped `/api/chat` call, held the single
request thread until the reply finished, up to the whole window.

**Verified against.** A client that leaves mid-stream frees the server in
0.25 s, and one that leaves during prefill in 0.88 s (SmolLM2, release build).

**If regressed.** After pressing Stop, the next message waits for the
abandoned reply to finish.

## 2026-11-16 — Only verified families convert

**Decision.** `rai convert` and the Python exporters accept a `model_type` only
from `VERIFIED_MODEL_TYPES`, held equal across the two by
`tests/limits_are_single_sourced.rs`.

**Symptom.** Tensor names prove nothing about the arithmetic: ERNIE-4.5 rotates
interleaved pairs and EXAONE-4 skips RoPE on its global layers under Llama
names, and both would convert into a model that computes something else.

**Verified against.** `only_verified_families_convert` and the Python suite's
`test_preflight_converts_only_verified_families`.

**If regressed.** A new family converts cleanly, loads, and generates fluent
nonsense.

## 2026-11-16 — Conversions refuse before working and write atomically

**Decision.** A different `tokenizer.json` in the output folder, a malformed
`tokenizer_config.json` or `generation_config.json`, and shard names with a path
in them are refused before quantizing; the model, the tokenizer and the chat
record are written through per-process `.partial` files and renamed into place.

**Symptom.** A clash used to be found after the model was written beside another
model's tokenizer; two conversions to one output shared a `.partial`; an index
naming `\\host\share` made Windows authenticate to that host.

**Verified against.** `a_tokenizer_clash_is_refused_before_the_model_is_written`,
`an_index_naming_a_shard_outside_the_folder_is_refused`,
`test_a_failed_write_leaves_nothing_behind`.

**If regressed.** A model sits beside the wrong tokenizer and produces nonsense,
or an interrupted conversion leaves a truncated file under the real name.

## 2026-11-16 — A shorter draft window hands decoding to the target

**Decision.** `SpeculativeDecoder` keeps the target's window and drafts only
while the draft has room; past it the target decodes alone.

**Symptom.** A TinyLlama draft (2048) for a 4096-token target stopped
generation with an error near position 2048 and refused prompts longer than the
draft's window.

**Verified against.** `a_shorter_draft_window_hands_over_to_the_target`.

**If regressed.** `rai run --draft` dies part-way through long replies.

## 2026-11-16 — A release builds only what CI passed, and is never rebuilt

**Decision.** `release.yml` resolves the tag to one commit through the tag
namespace, waits for green CI on that commit, resumes drafts only, and attests
each archive's provenance; `publish.yml` publishes the tag's commit from the
`crates-io` environment and skips crates already published.

**Symptom.** v0.2.5 was tagged on a tree that failed clippy and the Python
suite; a re-run replaced published archives and rewrote `SHA256SUMS`.

**Verified against.** The workflows' own steps (`Require green CI on this exact
commit`, the published-release refusal); docs/RELEASE.md.

**If regressed.** Users download binaries no test passed, or hashes they
pinned stop matching what the release serves.

## 2026-11-16 — `abs_tol` is in weight units for every compressor

**Decision.** HRC and SAC convert `abs_tol` into each block's row-normalized
units (`ChannelNorm::block_tolerance`).

**Symptom.** Applied as is to normalized values, the bound scaled with each
row's spread, and the HRC-against-RC report compared different error budgets.

**Verified against.** `abs_tol_bounds_the_error_in_weight_units` (rows a
thousand times apart in spread).

**If regressed.** Reported accuracy wins come from a looser budget, not the
method.

## 2026-11-16 — The Windows launcher never expands a path bare

**Decision.** `rai-studio.cmd` runs with delayed expansion off and expands
every path inside double quotes, outside parenthesized blocks.

**Symptom.** `&` in a folder name ran a command, "rai (1)" failed to parse, and
`!` vanished from paths.

**Verified against.** `the_windows_launcher_survives_awkward_folder_names`,
which fails against the previous launcher ("x was unexpected at this time").

**If regressed.** Double-clicking the launcher in an awkwardly named folder
shows a cmd syntax error, or runs part of the folder name as a command.

## 2026-11-16 — Transitive dependency updates wait two weeks

**Decision.** When a security fix pulls fresh transitive releases, they are held
to the newest version at least two weeks old that still carries the fix
(mio 1.2.4, syn 3.0.9, rustls-webpki 0.103.16, webpki-roots 1.0.11 in this
cycle).

**Symptom.** A days-old release is the most likely place for a compromised or
broken crate to land.

**Verified against.** `cargo audit --deny warnings` clean, and the workspace
builds on Rust 1.87.

**If regressed.** Nothing visible until a bad release is pulled in; the cooldown
is the only defence before an advisory exists.
