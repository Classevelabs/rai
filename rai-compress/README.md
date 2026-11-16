# classeve-rai-compress

Weight quantization and compression research toolkit, part of
[RAI](https://github.com/Classevelabs/rai). Nothing in this crate is on RAI's
inference path: the `rai` CLI and the `.raimodel` container live in
`classeve-rai-infer`.

Two families live here:

- **GPTQ** (`gptq`): calibration-based quantization after Frantar et al.,
  using the Cholesky factor of the inverse Hessian, as an independent Rust
  implementation.
- **RC, HRC and SAC** (`compress`, `hrc`, `sac`, built on `prior`, `channel`,
  `sparse`, `quantize` and `bitpack`): experimental compressors over in-memory
  `f64` matrices. They report modeled sizes and serialize no artifact.
  `abs_tol` bounds the error each adds to any element, in weight units, for
  all three.

Every public entry point validates its input and returns a `Result`; the
crate is `#![forbid(unsafe_code)]`.

Licensed under Apache-2.0.
