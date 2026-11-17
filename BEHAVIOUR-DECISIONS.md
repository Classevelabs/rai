# Why RAI behaves the way it does

Every entry here is a decision that looks arbitrary, slower than it needs to
be, or removable until you know what it cost. Each states the decision, the
symptom that caused it, what it was verified against, and what a user sees if
someone regresses it. The reason also sits in a comment at the code; this file
is the second copy, for people reading the product rather than the file.

---

## 2026-11-17 — Nothing ships that was not installed on a clean machine

**Decision.** `dist.yml` builds every download once; `clean-room.yml` installs
each on machines that never saw a build: Windows with the Visual C++ runtime
renamed away and PATH cut to Windows' own folders, both kinds of Mac, and ten
Linux distributions in bare containers. Each must download a model, convert
it and answer "Paris", from `rai run` and from Studio's `/api/chat`. The
release attaches only the files that passed, and runs the test again on what
GitHub serves.

**Symptom.** 0.2.5 passed every test on the machines that built it and failed
on a fresh one: `VCRUNTIME140.dll` missing on Windows, `GLIBC_2.34` missing on
Ubuntu 20.04 and RHEL 8, and no route from download to first answer without
Python.

**Verified against.** The `Release candidate` workflow, green on the commit
that was tagged.

**If regressed.** The first thing a new user sees is a missing-DLL or glibc
error, or an empty chat page.

## 2026-11-17 — The program stays offline; one script downloads

**Decision.** `rai` never opens a network connection. Downloading lives in
`rai-get-model`, a shell and a PowerShell script that call `curl`, check each
file against a pinned SHA-256 (starter models) and run `rai check` before the
weights (any other repository).

**Symptom.** None; this keeps a promise the README makes. A downloader inside
the binary would bring a TLS stack and a network path into a program that
says it has none.

**Verified against.** `every_starter_model_is_pinned_and_complete`, and the
clean-room runs, which download through the script on every platform.

**If regressed.** `rai` gains a network dependency and an attack surface its
users were told it does not have.

## 2026-11-17 — Windows links the C runtime statically; Linux builds on glibc 2.28

**Decision.** `.cargo/config.toml` sets `+crt-static` for
`x86_64-pc-windows-msvc`; the Linux release builds in
`manylinux_2_28` and fails if any symbol needs a newer glibc.

**Symptom.** See the first entry of this date.

**Verified against.** `dist.yml`'s import check (no `VCRUNTIME140.dll`) and
its `objdump` glibc check; the clean room on Ubuntu 20.04 and Rocky 8.

**If regressed.** "VCRUNTIME140.dll was not found" on Windows, or
"GLIBC_2.xx not found" on older Linux.

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

## 2026-11-16 — A stop that is a control token ends a reply by id only

**Decision.** A template's stop marker that the tokenizer holds as an added
token (`</s>`, `<|im_end|>`, `[INST]` where it is one) joins the end ids and is
dropped from the text stops; markers the tokenizer spells as plain text, such
as TinyLlama's `<|user|>`, stay text stops (`chat_format.rs`, `ChatFormat::new`).

**Symptom.** The control token already ended the reply by id, so its text match
could only fire on ordinary tokens spelling it: "Use `<s>old</s>`" was cut at
`</s>`, where transformers stops on ids alone.

**Verified against.** `control_token_stops_end_by_id_not_by_text`.

**If regressed.** Replies that quote markup or a template's own markers stop
at the quote.

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

The record's fields come from the files transformers 5.8 reads them from
(checked in its source): end ids from `config.json` when there is no
`generation_config.json`, a pre-`added_tokens_decoder` tokenizer config's
tokens from `special_tokens_map.json`, and `strftime_now` in local time.

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

**Decision.** Between prefill chunks and between tokens the server checks the
connection (`cli/http.rs` `client_gone`) and stops when the client has closed
it.

**Symptom.** Studio's Stop, or a dropped `/api/chat` call, held the single
request thread until the reply finished, up to the whole window.

**Verified against.** A client that leaves mid-stream frees the server in
0.25 s, and one that leaves during prefill in 0.88 s (SmolLM2, release build).

Bytes still arriving after the request are read and discarded, not peeked: one
stray byte hid the end of stream behind it for good
(`surplus_bytes_do_not_hide_a_departed_client`). A client that shuts only its
sending side after the request looks the same as one that left, and is
treated as gone.

**If regressed.** After pressing Stop, the next message waits for the
abandoned reply to finish.

## 2026-11-16 — A request has 15 seconds from connecting, not 15 per byte

**Decision.** `READ_TIMEOUT` is a deadline set when the connection is
accepted; each read waits only for what is left of it (`cli/http.rs`
`read_by`). Refusals drain what the client is still sending, within `LINGER`,
before closing.

**Symptom.** The timeout restarted with every byte, so a local process sending
one byte every 14 seconds held a reader slot for hours, and 32 of them made the
server answer 503 to everyone, Studio included.

**Verified against.** `a_dribbling_client_is_cut_off_at_the_deadline`, which
hangs on the old timeout; `a_refusal_arrives_whole_with_the_request_still_in_flight`.

**If regressed.** Any local process can lock Studio out of its own server.

## 2026-11-16 — Only verified families convert

**Decision.** `rai convert` and the Python exporters accept a `model_type` only
from `VERIFIED_MODEL_TYPES`, held equal across the two by
`tests/limits_are_single_sourced.rs`.

**Symptom.** Tensor names prove nothing about the arithmetic: ERNIE-4.5 rotates
interleaved pairs and EXAONE-4 skips RoPE on its global layers under Llama
names, and both would convert into a model that computes something else.

**Verified against.** `only_verified_families_convert` and the Python suite's
`test_preflight_converts_only_verified_families`.

The family is settled before any other field is read, and a config that nests
its language model under `text_config` is refused as multimodal. `gemma3` is
not on the list: it is the multimodal Gemma3 (4B and larger), never verified;
the verified Gemma3 is `gemma3_text` (`the_family_is_the_reason_a_foreign_config_is_refused`).

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

Studio's Check runs these same checks once the weights are on disk
(`check_checkpoint_tensors`, `read_companion_files`), so it cannot call a
checkpoint convertible that the conversion then refuses
(`check_refuses_what_the_conversion_refuses`). The Python exporters copy the
checkpoint's own `tokenizer.json`, as `rai convert` does, so the clash check
compares the same bytes whichever tool wrote first.

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
namespace, waits for green CI on that commit (a push or dispatched run of the
commit itself; a pull-request run tests a merge), looks for the tag's draft
before opening one, resumes drafts only, and attests each archive's
provenance; `publish.yml` publishes the tag's commit from the
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
