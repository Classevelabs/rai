# Two images from one file:
#
#   docker build -t rai-server .                 -> MCP/REST memory server (default)
#   docker build --target cli -t rai-cli .       -> the `rai` inference CLI
#
# The stages are split so the default build stays exactly as cheap as it was:
# Docker only builds the stages the chosen target depends on, so asking for the
# server never compiles the inference CLI and vice versa.

# The optimized amd64 image is intentionally x86-64-v3. Unlike
# target-cpu=native, this does not bake the builder host's exact CPU into the
# release binary. Override it if the image has to run on older hardware —
# x86-64-v2 is the portable floor the published release archives use, and it
# costs nothing measurable, because the AVX2/FMA/F16C kernels are chosen at
# runtime rather than by the compile-time baseline:
#
#   docker build --build-arg RUST_TARGET_CPU=x86-64-v2 .
#
# This value is applied only when the image being built is actually x86-64.
# `docker build .` on an Apple Silicon Mac builds linux/arm64 by default, and
# handing an aarch64 compiler an x86 processor name got a wall of LLVM
# "not a recognized processor for this target" warnings and a silently ignored
# flag. On arm64 the build falls through to .cargo/config.toml, which scopes
# its own baseline to x86-64 and so leaves aarch64 at the toolchain default.
ARG RUST_TARGET_CPU=x86-64-v3

# The tag tracks rust-toolchain.toml: the container must compile with the
# same compiler every other build of this repo uses, or the image is the one
# artifact built by a compiler nothing else tested.
FROM rust:1.95.0-slim-bookworm@sha256:d7482085ff5b415f84dba5647ae71606650bdef00db7aeb69f4b3d170c3e4082 AS toolchain

WORKDIR /src
COPY . .

# `uname -m` rather than the BuildKit TARGETARCH ARG: this stage runs on the
# target platform either way, and uname needs no particular builder to be right.
FROM toolchain AS build-server
ARG RUST_TARGET_CPU
RUN if [ "$(uname -m)" = "x86_64" ]; then export RUSTFLAGS="-C target-cpu=${RUST_TARGET_CPU}"; fi; \
    cargo build --locked --release --package classeve-rai-server

FROM toolchain AS build-cli
ARG RUST_TARGET_CPU
RUN if [ "$(uname -m)" = "x86_64" ]; then export RUSTFLAGS="-C target-cpu=${RUST_TARGET_CPU}"; fi; \
    cargo build --locked --release --package classeve-rai-infer

FROM debian:bookworm-slim@sha256:abd67ffcfa541b485a3dff59865ab629aa048a6c613e639d36e7456b0b229241 AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 rai \
    && useradd --create-home --uid 10001 --gid 10001 --shell /usr/sbin/nologin rai

# Inference CLI image. Not the default target: it carries no server, and it is
# useless without a model, which has to be mounted in. Models are far larger
# than this image and are deliberately not baked into it.
#
#   docker run --rm -v "$PWD:/work" -w /work rai-cli \
#     rai convert ./TinyLlama-1.1B-Chat-v1.0 -o ./tinyllama.raimodel
#
#   docker run --rm -v "$PWD:/work" -w /work rai-cli \
#     rai run ./tinyllama.raimodel --chat-template zephyr --prompt "Hello"
#
# `rai serve` is reachable from this image only in principle. It hard-binds
# 127.0.0.1 and rejects any request whose Host/Origin is not localhost, which is
# the right behaviour for a local UI and makes it unreachable through published
# container ports. Run `rai serve` on the host instead.
FROM base AS cli

COPY --from=build-cli /src/target/release/rai /usr/local/bin/rai

LABEL org.opencontainers.image.source="https://github.com/Classevelabs/rai" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="RAI inference CLI"

USER 10001:10001
CMD ["rai", "--help"]

# Default target — keep this stage last so a bare `docker build .` still
# produces the MCP server image CI smoke-tests.
FROM base AS runtime

COPY --from=build-server /src/target/release/rai-server /usr/local/bin/rai-server

LABEL org.opencontainers.image.source="https://github.com/Classevelabs/rai" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="RAI server"

USER 10001:10001
ENTRYPOINT ["/usr/local/bin/rai-server"]
CMD ["mcp"]
