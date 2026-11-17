# Building RAI

You need [Rust](https://rustup.rs) 1.87 or newer. The repository pins 1.95.0
in `rust-toolchain.toml`, and `rustup` fetches it the first time you build.
Rust links with the system's own tools: on Windows, Microsoft's C++ Build
Tools (the "Desktop development with C++" workload, which `rustup` offers to
install); on macOS, the Xcode Command Line Tools (`xcode-select --install`);
on Linux, gcc or clang from your package manager. Nothing else: no Python and
no other libraries.

## From a checkout

```bash
git clone https://github.com/Classevelabs/rai.git
cd rai
cargo build --release --locked -p classeve-rai-infer
```

The program is `target/release/rai` (`rai.exe` on Windows). To install it into
`~/.cargo/bin`, which `rustup` put on your PATH:

```bash
cargo install --locked --path rai-infer
```

Or straight from crates.io, without a checkout:

```bash
cargo install --locked classeve-rai-infer
```

The `launchers` folder holds `rai-studio` and `rai-get-model`. Put them next to
the `rai` you built, or on your PATH, to use them with it.

`cargo build --workspace` also builds the memory-service crates (`rai-core`,
`rai-server`, `rem-nra`). They are not part of RAI and need a C compiler for
their TLS library.

## A build that runs anywhere

`.cargo/config.toml` builds for the `x86-64-v2` baseline, the same one the
releases use, so a binary you build runs on any x86-64 processor from 2011 on.
The fast AVX2 code is chosen when the program starts, not when it is
compiled, so this costs nothing. Windows builds also carry their own C
runtime, so they run without the Visual C++ Redistributable.

For a benchmark on your own machine only, you can tune the build to it:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --locked -p classeve-rai-infer
```

That binary may crash with an illegal-instruction error on any older
processor, so don't give it to anyone.

## Checks

The same checks CI runs:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --all-targets --locked
cargo test --workspace --doc --locked
```

CI also builds every release download and installs it on clean Windows, macOS
and Linux machines; [RELEASE.md](RELEASE.md) describes that.

## The container image

The `Dockerfile` builds an image with `rai` in it. Models are mounted in, never
built into the image:

```bash
docker build -t rai .
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" -w /work rai \
  rai convert ./TinyLlama-1.1B-Chat-v1.0 -o ./tinyllama.raimodel
docker run --rm -v "$PWD:/work" -w /work rai \
  rai run ./tinyllama.raimodel --prompt "Hello"
```

`--user` makes the converted files yours. `rai serve` answers only inside the
container, so run it on the host instead. The image builds for `x86-64-v3` on
x86-64 machines; pass `--build-arg RUST_TARGET_CPU=x86-64-v2` for the release
baseline. `docker build --target server .` builds the separate memory
service, which [OPERATIONS.md](OPERATIONS.md) covers.

## Using the engine as a library

```toml
[dependencies]
classeve-rai-infer = { version = "0.2", default-features = false }
```

Without its default `cli` feature, the crate is the engine alone (the file
reader, the kernels, the model, sampling and speculative decoding) and
depends on only `half`, `rayon`, `anyhow` and `rand`. The API documentation
is on [docs.rs](https://docs.rs/classeve-rai-infer), and
[ARCHITECTURE.md](ARCHITECTURE.md) explains how the pieces fit together.
