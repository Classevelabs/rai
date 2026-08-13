# Two images from one file:
#
#   docker build -t rai-server .                 -> MCP/REST memory server (default)
#   docker build --target cli -t rai-cli .       -> rai-convert / rai-generate
#
# The stages are split so the default build stays exactly as cheap as it was:
# Docker only builds the stages the chosen target depends on, so asking for the
# server never compiles the inference CLI and vice versa.

# The optimized image is intentionally amd64/x86-64-v3. Unlike
# target-cpu=native, this does not bake the builder host's exact CPU into the
# release binary. Override it if the image has to run on older hardware —
# x86-64-v2 is the portable floor the published release archives use, and it
# costs nothing measurable, because the AVX2/FMA/F16C kernels are chosen at
# runtime rather than by the compile-time baseline:
#
#   docker build --build-arg RUST_TARGET_CPU=x86-64-v2 .
ARG RUST_TARGET_CPU=x86-64-v3

FROM rust:1.97.1-slim-bookworm@sha256:96c0af8cf054fd006435089f0076729716784ec9be485bd655de59c55df105ce AS toolchain

WORKDIR /src
COPY . .

FROM toolchain AS build-server
ARG RUST_TARGET_CPU
ENV RUSTFLAGS="-C target-cpu=${RUST_TARGET_CPU}"
RUN cargo build --locked --release --package classeve-rai-server

FROM toolchain AS build-cli
ARG RUST_TARGET_CPU
ENV RUSTFLAGS="-C target-cpu=${RUST_TARGET_CPU}"
RUN cargo build --locked --release --package classeve-rai-infer

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
#     rai-convert --model ./TinyLlama-1.1B-Chat-v1.0 --output ./tinyllama.raimodel
#
#   docker run --rm -v "$PWD:/work" -w /work rai-cli \
#     rai-generate --model ./tinyllama.raimodel --tokenizer ./tokenizer.json \
#                  --chat-template zephyr --prompt "Hello"
#
# rai-chat is deliberately NOT in this image. It hard-binds 127.0.0.1 and
# rejects any request whose Host/Origin is not localhost, which is the right
# behaviour for a local UI and makes it unreachable through published container
# ports. Shipping it here would only produce a connection that always refuses.
# Run rai-chat from the release archive on the host instead.
FROM base AS cli

COPY --from=build-cli /src/target/release/rai-convert /usr/local/bin/rai-convert
COPY --from=build-cli /src/target/release/rai-generate /usr/local/bin/rai-generate

LABEL org.opencontainers.image.source="https://github.com/Classevelabs/rai" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="RAI inference CLI"

USER 10001:10001
CMD ["rai-generate", "--help"]

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
