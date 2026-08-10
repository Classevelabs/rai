FROM rust:1.95.0-slim-bookworm@sha256:d7482085ff5b415f84dba5647ae71606650bdef00db7aeb69f4b3d170c3e4082 AS build

WORKDIR /src
COPY . .

# The optimized image is intentionally amd64/x86-64-v3. Unlike
# target-cpu=native, this does not bake the builder host's exact CPU into the
# release binary.
ENV RUSTFLAGS="-C target-cpu=x86-64-v3"
RUN cargo build --locked --release --package classeve-rai-server

FROM debian:bookworm-slim@sha256:abd67ffcfa541b485a3dff59865ab629aa048a6c613e639d36e7456b0b229241 AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 rai \
    && useradd --create-home --uid 10001 --gid 10001 --shell /usr/sbin/nologin rai

COPY --from=build /src/target/release/rai-server /usr/local/bin/rai-server

LABEL org.opencontainers.image.source="https://github.com/Classevelabs/rai" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="RAI server"

USER 10001:10001
ENTRYPOINT ["/usr/local/bin/rai-server"]
CMD ["mcp"]
