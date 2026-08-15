# Security Policy

Aletheia is built to analyze **hostile and untrusted binaries**. Crashes,
panics, and unexpected filesystem writes against crafted inputs are
security bugs — please report them.

## Supported versions

| Version | Supported |
| --- | --- |
| `main` (0.1.x) | Yes |

## Reporting a vulnerability

Please **do not** open a public issue for:

- Parser / decoder panics or memory unsafety on crafted PE / ELF / Mach-O
- Path traversal or overwrite bugs in `--patch-apply` / sibling writes
- MCP tool behavior that can escape the intended workspace

Instead, email the maintainers via the address on the GitHub profile, or
open a **private** security advisory on this repository:

https://github.com/eddinos2/aletheia/security/advisories/new

Include a minimal reproducer binary (or generator script) and the exact
CLI / MCP tool invocation.

## Scope notes

- Intentional analysis of malware samples is in-scope for the tool; shipping
  malware in PRs/issues is not. Use links to public corpora when needed.
- Issues that only affect cosmetic decompiler output quality are not
  security bugs — file those as normal issues.
