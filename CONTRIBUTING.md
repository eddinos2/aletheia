# Project Conventions & Communication Guidelines

This is an **open-source binary-analysis / reverse-engineering toolkit** built for
security research, malware analysis, software interoperability, and education. Its
lineage is the same as Ghidra (NSA, open source), radare2, Rizin, and Cutter — all
legitimate, widely used tools.

These guidelines exist for one reason: to describe the work **accurately**, so that
automated content classifiers and human reviewers correctly recognize it as the
legitimate security tooling it is. They are *not* a way to sneak anything past a
safety system — the project has nothing to hide, and if something here were actually
disallowed, no wording would (or should) change that.

## Why accurate framing matters

Safety classifiers pattern-match on text. A legitimate reverse-engineering project
described in "crack the expensive paywalled product" language pattern-matches to
piracy/circumvention intent even when that's not the goal. Describing it precisely
removes the ambiguity. Same project, accurate words.

## Naming

- The tool ships as **Aletheia**. **Avoid `NOT IDA PRO`** as a product name:
  naming something as the negation of a trademark invites trademark noise and
  "clone" framing.
- Do not copy IDA's/Hex-Rays' branding, UI chrome, icons, or proprietary strings.

## How to describe the project (do / avoid)

**Do say:**
- "binary-analysis toolkit," "disassembler," "decompiler," "control-flow recovery"
- "for security research, malware analysis, vulnerability research, education"
- "clean-room implementation from public specifications"

**Avoid saying:**
- "crack / defeat / beat the paywall," "clone $EXPENSIVE_PRODUCT," "piracy"
- "make IDA but free so nobody has to pay" — even as a joke, it reframes the goal
- Overclaiming: "a full IDA competitor, from scratch, in one shot." IDA is ~25 years
  of work. Scope to concrete components and build incrementally.

## Legal / IP hygiene (the actual constraints)

- **Clean-room only.** Implement from *public* specifications: ELF, PE/COFF, Mach-O,
  DWARF, x86/x86-64 (Intel SDM), ARM ARM, etc. Never copy IDA/Hex-Rays source,
  disassembled internals, or decompiler output.
- Respect the license of any third-party component you pull in (Capstone, LLVM,
  Zydis, etc. — check each).
- Ship under a clear OSS license (Apache-2.0 or MIT recommended).

## How to frame a task/request when working on this

Lead with the **technical component**, not the market/pricing framing:

- Good: "Implement a PE32+ header parser that extracts sections and the import table."
- Good: "Write an x86-64 length-disassembler for the linear-sweep pass."
- Weaker: "Let's build the expensive IDA thing and make it free, go."

The first two are unambiguous engineering tasks. The last is what tripped the
classifier — not because it's disallowed, but because it's vague and market-framed.

## When a false positive still happens

Automated safeguards are intentionally broad and will occasionally flag legitimate
work anyway. If that happens: it's a known false-positive mode, the harness may
reroute to another model, and the work continues. Report it via `/feedback` so the
classifier improves. Do not attempt to defeat or circumvent the safeguard itself.
