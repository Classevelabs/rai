# Security Policy

## Supported versions

RAI is pre-1.0. Security fixes are made on `main` and in the newest release
when practical. Older crates and GitHub releases may not receive backports.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use the repository's
private GitHub Security Advisory form. If that is unavailable, use the contact
channel at <https://classeve.com> and include the affected component,
reproduction steps, likely impact, and any safe-to-share model or request
fixture.

## Runtime security model

- The REST listener is local-only and serves plain HTTP. It refuses
  non-loopback `RAI_HOST` values. `RAI_API_TOKEN`, when configured, must contain
  at least 32 bytes and applies to all `/v1/*` routes. Use a TLS-terminating
  reverse proxy for any remote access.
- MCP uses stdio and inherits the permissions of the process that launches it;
  it is not a network authentication boundary. Mutating MCP tools are hidden
  and denied unless `RAI_MCP_MUTATIONS_ENABLED=true` is explicitly set.
- The built-in mock embedder is deterministic demonstration behavior, not a
  production semantic-retrieval provider.
- Model files and persisted memory can drive large allocations and parser
  paths. Treat files from untrusted sources cautiously and report any panic,
  out-of-bounds access, excessive allocation, or unsafe-kernel issue.

## Scope

In scope are memory-safety bugs, denial of service, malformed `.raimodel`
handling, authentication or transport bypasses, persistence corruption or data
exposure, and vulnerable published dependencies. Model hallucinations and the
behavior of third-party embedding/model services are out of scope unless RAI
handles their response unsafely.
