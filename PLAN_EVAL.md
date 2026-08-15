# Plan: evaluation harness — ground-truth fixtures + metrics (`evalfx`, DESIGN slice 19)

## Goal

Five fidelity passes now iterate on the same corpus numbers, and every
wave re-derives them by hand with ad-hoc greps. Adopt the published
evaluation methodology as regression infrastructure (DESIGN slice 19):
one command reports, per fixture, (a) goto count, (b) structure
similarity against the known source CFG, (c) semantic spot checks —
and `cargo test` fails when a metric regresses.

## The three metrics (authority: DESIGN.md slice 19)

- **Goto count** per fixture through the full pipeline (construct →
  optimize → forward → eliminate_dead → structure), the SAILR metric.
- **CFGED-lite**: a bounded exact graph-edit-distance between the
  recovered structure tree's CFG-shape and the fixture's known source
  CFG, on small graphs only — dependency-free, exact or refuse (a
  fixture too big for exact GED is a documented non-metric, not an
  approximation).
- **Semantic spot checks** (Liu & Wang, ISSTA 2020 lineage): the
  slice-7 SSA interpreter evaluates the SSA function on seeded inputs
  before and after each optimization stage — behavior equality as a
  regression bit. The interpreter lives `#[cfg(test)]` in `irout`;
  placement follows from that (see below), do not weaken its privacy.

## Fixtures (the provenance rule)

Checked-in fixture binaries compiled offline from checked-in C, per
DESIGN: no build-time deps — the .c files, the exact compile commands,
compiler version, and flags are recorded in a `fixtures/README.md`
(CONTRIBUTING-style provenance), and the produced Mach-O/ELF x86-64
binaries are committed as bytes. Build them once locally with the
system clang (record `clang --version` output verbatim). Start small
and known-shaped: a diamond, a switch (dense, so a table is emitted),
a loop with break/continue, a tail-merged shape, a short-circuit
`&&`/`||` chain — each with its hand-written source CFG recorded next
to it as the GED ground truth. The existing real-binary sweeps
(ls/bash/brotli) are NOT fixtures — they stay exit-criteria material,
summarized in ROADMAP, never asserted in tests.

## Placement

The interpreter privacy constraint decides: the harness is a
`#[cfg(test)]` module in `src/` (e.g. `src/evalfx.rs` registered in
lib.rs, or a tests module the builder justifies) so it can reach
`irout`'s test-only interpreter via crate-internal paths — DESIGN's
"tests/ fixtures" is satisfied by the `fixtures/` directory, not by
Cargo's integration-test layout, and the deviation is documented. The
one command: `cargo test evalfx` (a naming convention, not new CLI).
Expected numbers live in ONE table in ONE file, trivially updatable
when a fidelity slice legitimately moves them — that update friction
is the design, but it must be one-place friction.

## Module-by-module

- `fixtures/` (new, top level): .c sources, compiled binaries,
  README.md with provenance, expected-CFG descriptions.
- `src/evalfx.rs` (or justified equivalent): fixture loader, the
  bounded exact GED (small, tested against hand-computed distances),
  the metric table, the assertions.
- `src/lib.rs`: register the module.
- Do NOT touch src/irstruct.rs, src/irflow.rs, src/irssaopt.rs,
  src/aarch64*.rs, src/jumptable.rs, src/cfg.rs — three companion
  slices run in parallel; your numbers will be re-measured at merge,
  so keep the expected-number table isolated and cheap to update.
- `ROADMAP.md`: Current-thread entry; record the initial numbers.

## Test matrix (~the harness IS the matrix)

1. every fixture: pipeline runs, zero check failures, goto count
   equals the recorded expectation.
2. GED-lite: exact distances against hand-computed values on the
   small graphs; refusal on an oversized graph is explicit.
3. semantic: interpreter equality across every optimization stage on
   seeded inputs, all fixtures.
4. determinism of every metric; the harness itself byte-stable.

## Exit criteria (demonstrate, don't assert)

`cargo test evalfx` runs the full table and fails when seeded with a
deliberately wrong expectation (prove the teeth — show the failing
run in the report, then restore). Initial numbers recorded in
ROADMAP. The report states, for each of the five queued fidelity
follow-ups, which metric would catch its regression.

## Non-goals

- Approximate GED on large graphs; any new CLI surface.
- Real-binary numbers as test assertions (sweep material only).
- aarch64 fixtures (x86-64 first; arm64 is a follow-up rider).
