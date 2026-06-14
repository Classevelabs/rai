# Security Policy

## Supported Versions

RAI is currently pre-1.0. Security fixes are applied to the `main` branch and
to the latest GitHub release when practical.

## Reporting a Vulnerability

Please do not open a public issue for a suspected vulnerability.

Report security issues through GitHub Security Advisories for this repository.
If that is unavailable, contact Classeve through the public contact channel on
https://classeve.com and include:

- Affected component or binary.
- Reproduction steps or proof of concept.
- Expected impact.
- Any relevant model file, request payload, or configuration details that can be
  shared safely.

We aim to acknowledge reports within 5 business days. Confirmed issues are
handled with coordinated disclosure and credited when the reporter wants credit.

## Scope

In scope:

- Memory safety, denial of service, unsafe parser behavior, or malformed
  `.raimodel` handling.
- REST or MCP server behavior that can expose data, corrupt memory state, or
  permit unintended access.
- Dependency or supply-chain vulnerabilities affecting the published project.

Out of scope:

- Model quality issues, hallucinations, or prompt-level behavior.
- Vulnerabilities in external services used through optional embedding
  providers.
- Reports that require physical access to a user's machine.
