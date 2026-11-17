# Security Policy

## Supported versions

RAI is pre-1.0. Security fixes are made on `main` and in the newest release
when practical. Older crates and GitHub releases may not receive backports.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use the repository's
private GitHub Security Advisory form, or email <security@classeve.com>.
Include the affected component,
reproduction steps, likely impact, and any safe-to-share model or request
fixture.

## Runtime security model

### `rai serve` (Studio and the chat API)

- It listens on loopback only — `127.0.0.1`, or `[::1]` on a machine without
  an IPv4 loopback address — and no flag or variable widens that. It serves
  plain HTTP.
- It has no authentication. Like the `rai` command line, it acts for whoever
  can reach it, and on loopback that is every process on the machine, under
  every account. Run it on a machine whose other users and processes you
  trust; on a shared machine, do not run it.
- Web pages cannot use it. Every request must carry this server's own
  `Host`; every `/api` request, whatever its method, is refused when it comes
  from another origin (`Origin`, `Sec-Fetch-Site`); every `POST` must be JSON,
  which a page cannot send cross-origin without a preflight the server
  refuses; and the page sets a CSP that forbids framing it.
- Paths in requests are the user's own filesystem, read as the command line
  would read them. Network and device paths (`\\host\share`, `\\?\UNC\…`,
  `\\.\…`) are refused, because opening one makes Windows authenticate to the
  host it names. A conversion writes only a `.raimodel` file, the
  `<name>.chat.json` record beside it, and a `tokenizer.json`.
- Request bodies are limited to 64 KiB, and a larger declared body is refused
  unread. Each connection has a deadline, and a bounded number are read at
  once, so a stalled client cannot hold the server.

### Installers and `rai-get-model`

- `rai` itself opens no network connections; it has no HTTP client and no TLS
  library. The installers and `rai-get-model` are the only parts that
  download, and only when run.
- `install.sh` and `install.ps1` download over HTTPS from this repository's
  GitHub releases and refuse any file whose SHA-256 differs from the release's
  `SHA256SUMS`. Every release file also carries a signed build-provenance
  attestation (`gh attestation verify <file> --repo Classevelabs/rai`).
- `rai-get-model` downloads over HTTPS from huggingface.co, or from the
  address in `HF_ENDPOINT` if you set one for a mirror. A starter model is
  pinned to one commit, and each of its files must match a SHA-256 recorded in
  `launchers/models.tsv` before it is converted. Any other repository is taken
  as Hugging Face serves it, and `rai check` reads its configuration before
  the weights are fetched. A `HF_TOKEN` you set is sent only to that same
  address.
- Only `.safetensors` weights are read. Pickle formats (`pytorch_model.bin`,
  `.pt`, `.pth`) are refused, because loading one can run code.

### `rai-server` (the memory service; not part of the RAI product)

- The REST listener is local-only and serves plain HTTP. It refuses
  non-loopback `RAI_HOST` values. `RAI_API_TOKEN`, when configured, must contain
  at least 32 bytes and applies to all `/v1/*` routes. Use a TLS-terminating
  reverse proxy for any remote access.
- MCP uses stdio and inherits the permissions of the process that launches it;
  it is not a network authentication boundary. Mutating MCP tools are hidden
  and denied unless `RAI_MCP_MUTATIONS_ENABLED=true` is explicitly set.
- The built-in mock embedder is deterministic demonstration behavior, not a
  production semantic-retrieval provider.

### Model files

- Model files and persisted memory can drive large allocations and parser
  paths. Treat files from untrusted sources cautiously and report any panic,
  out-of-bounds access, excessive allocation, or unsafe-kernel issue.

## Scope

In scope are memory-safety bugs, denial of service, malformed `.raimodel`
handling, authentication or transport bypasses, persistence corruption or data
exposure, and vulnerable published dependencies. Model hallucinations and the
behavior of third-party embedding/model services are out of scope unless RAI
handles their response unsafely.
