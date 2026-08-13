# Operations

`rai-server` is a local service, not an internet-facing TLS endpoint. Run REST
on loopback and put any remote-access policy, TLS, rate limiting, and user
authentication in a separately operated reverse proxy. The proxy must connect
to the configured loopback address and send a loopback `Host` header.

## Start and verify

```bash
RAI_API_TOKEN="replace-with-at-least-32-random-bytes" \
RAI_DATA_PATH="./state/rai-memory.json" \
./target/release/rai-server rest

curl -fsS \
  -H "Authorization: Bearer $RAI_API_TOKEN" \
  http://127.0.0.1:3000/v1/health
```

Use `rai-server mcp` for stdio. MCP inherits the launching process's identity
and filesystem/network access. Leave `RAI_MCP_MUTATIONS_ENABLED=false` unless
that client is explicitly trusted to persist state.

## Persistence and recovery

- Without `RAI_DATA_PATH`, all state is lost at process exit.
- With it, startup fails on an unreadable or invalid snapshot instead of
  replacing it. Keep versioned backups outside the live path.
- Stop writes before copying the live snapshot. Restore a known-good snapshot
  to a new path first, start RAI against that path, and check `/v1/health`
  before switching over.
- Unix snapshots are created with mode `0600`. On Windows, use a directory
  whose ACL grants access only to the service identity.

## Monitoring

Treat process exit, repeated 4xx/5xx responses, embedding-provider failures,
persistence errors, and memory growth as alerts. The local `/v1/health`
endpoint is a liveness/diagnostic signal, not proof that the experimental
retrieval or reasoning results are semantically correct. Do not log bearer
tokens, API keys, request bodies, or stored memory content.

No committed load test establishes production capacity. Set external
concurrency and rate limits conservatively, observe latency and memory on the
deployment hardware, and retain those measurements before raising limits.

## Upgrade

Back up state, record the current binary/crate/container digest, run the full
[release gate](./RELEASE.md#gate), and exercise a copy of the snapshot with the
candidate. A crate version cannot be overwritten after it is published. Follow
the [rollback procedure](./RELEASE.md#rollback) if validation fails.
