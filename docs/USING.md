# Using RAI

- [RAI Studio](#rai-studio), the chat in your browser
- [The command line](#the-command-line)
- [Settings](#settings)
- [The chat API](#the-chat-api), for your own programs
- [Faster replies with a draft model](#faster-replies-with-a-draft-model)

## RAI Studio

Start it from the Start menu or your app menu, or with `rai-studio` in a
terminal. A small window stays open while it runs; close that window, or
press Ctrl+C in it, to stop RAI Studio. The chat opens in your browser at
<http://localhost:8090>.

The buttons along the left edge open its three screens:

- **Library** lists the models on this computer. Pick one to load it.
- **Convert** turns a model folder you downloaded yourself into a RAI model.
  **Check** tells you first whether it will work, before anything is written.
  Studio suggests saving it as `models/<name>/<name>-q4.raimodel` in your RAI
  folder (`~/RAI`, or `%USERPROFILE%\RAI` on Windows): one folder per model,
  where RAI finds it. Keep that form; two models in one folder would share
  one tokenizer file, and the second conversion is refused.
- **Chat** is where you talk to the model. Conversations stay in the chat
  list until you delete them, and **Stop** ends a reply early.

Along the top of the chat, the model menu switches models and the settings
button opens **Tuning**: how long a reply may be (**Max new tokens**), how
adventurous the wording is (**Temperature**; 0 always picks the most likely
word), the chat format and the context window, which is how much of the
conversation the model can see at once. The system monitor button
beside it shows your processor, memory and how much of the model's context
the conversation has used.

Conversations stay in your browser, on this computer. RAI Studio answers only
pages it served itself, so other websites can't reach it.

### Which model starts

RAI Studio loads, in this order: the model named in `RAI_MODEL`, the only
model next to the program, or the only model in `~/RAI/models` (on Windows,
`%USERPROFILE%\RAI\models`). With several models and none named, it starts
with none loaded, and you pick one in the Library.

## The command line

Ask one question:

```bash
rai run ~/RAI/models/qwen2.5-1.5b/qwen2.5-1.5b-q4.raimodel --prompt "Give me three names for a cat."
```

On Windows, in PowerShell, the same model is at
`"$env:USERPROFILE\RAI\models\qwen2.5-1.5b\qwen2.5-1.5b-q4.raimodel"`.

| Option | What it changes |
| --- | --- |
| `--max-tokens 1000` | Longest reply allowed (default 512) |
| `--temperature 0` | Always the most likely word: the same answer every time (default 0.7) |
| `--seed 7` | Another sample with the same settings (default 42) |
| `--chat-template none` | Sends your text as it is, for a base model that continues text |
| `--top-k`, `--top-p`, `--repetition-penalty` | Finer sampling controls; `rai run --help` explains each |

Start the chat page and API for a model of your choice:

```bash
rai serve path/to/model.raimodel --port 8090
```

The rest:

```bash
rai models                       # the models here and in ~/RAI/models
rai check ./Some-Model           # will this downloaded folder convert?
rai convert ./Some-Model -o some-model.raimodel
rai perplexity model.raimodel --text sample.txt   # how well the model predicts a text
```

[MODELS.md](MODELS.md) explains converting in full.

## Settings

All optional. Set them in your shell, or in Windows' environment variables
for your account.

| Setting | What it does |
| --- | --- |
| `RAI_HOME` | Where models are kept. Default `~/RAI` (Windows: `%USERPROFILE%\RAI`) |
| `RAI_MODEL` | The model RAI Studio loads at start |
| `RAI_PORT` | The port RAI Studio uses. Default 8090 |
| `RAI_NO_BROWSER=1` | Start RAI Studio without opening the browser |
| `RAYON_NUM_THREADS` | How many processor threads RAI may use. Default: one per performance core, plus one |
| `HF_TOKEN` | Your Hugging Face token, for models that ask you to log in |

## The chat API

`rai serve`, and RAI Studio with it, answers HTTP on your own computer. Send
the whole conversation each time, oldest message first; the server keeps no
conversation of its own:

```bash
curl http://localhost:8090/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"temperature":0}'
```

The answer, shortened to the fields most programs need:

```json
{"text": "The capital of France is Paris.", "tokens": 8, "stop_reason": "eos"}
```

| Field | Meaning |
| --- | --- |
| `messages` | The conversation: `role` is `user` or `assistant`, and the last one is `user` |
| `message` | Instead of `messages`, a single question |
| `temperature` | 0 to 2. Default 0.7 |
| `max_tokens` | Longest reply allowed. Default 512, or what the context has room for if that is less |
| `repetition_penalty` | 1 turns it off. Default 1.1 |

The answer also carries the prompt's token count, the speed, and
`stop_reason`: `eos` when the model finished its turn, `stop_sequence` when
it wrote one of its chat format's stop markers, and `max_tokens` when it hit
the limit.

`POST /api/chat/stream` takes the same request and answers one JSON object per
line while the reply is written, ending with a `done` record. Closing the
connection stops the reply.

Other routes: `GET /api/info` (the loaded model), `GET /api/models` (models on
this computer), `POST /api/load` (load another), `GET /api/system` (processor
and memory).

The server listens on your computer only (127.0.0.1). It refuses requests
from other websites and requests without a JSON `Content-Type`, so a page you
visit can't use your model behind your back. Bodies are limited to 64 KB.

## Faster replies with a draft model

With two models that share a tokenizer (the same vocabulary, as models of
one family do), a small one can guess a few words
ahead and the large one checks them, which can make the large one faster:

```bash
rai run big-model.raimodel --draft small-model.raimodel \
  --top-k 0 --top-p 1 --repetition-penalty 1 --prompt "..."
```

`--lookup-k 2` instead copies the guesses from the prompt itself. That helps
when the reply repeats the prompt's text, as in summaries or code edits, and
slows down free writing; [BENCHMARKS.md](../BENCHMARKS.md) has the numbers.
