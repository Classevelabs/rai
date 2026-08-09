# Support

RAI is a public Classeve engineering project. The fastest support path is a
GitHub issue with enough detail to reproduce the problem.

## Where To Ask

- Bugs: open a bug report using the issue template.
- Feature requests: open a feature request and describe the use case.
- Security issues: follow SECURITY.md instead of opening a public issue.
- General company information: https://classeve.com

## Useful Details

Please include:

- Operating system and CPU model.
- `rustc --version` and `cargo --version`.
- Command that failed.
- Relevant environment variables, with secrets removed.
- Whether the issue affects `rai-infer`, `rai-compress`, `rai-server`,
  `rai-core`, or `rem-nra`.

For inference issues, include model family, tokenizer source, export command,
and whether the model file was produced by GPTQ export or round-to-nearest
export.
