# Aletheia GUI/UX — North-Star Design Spec

*Phase 6 poly-frontend. This is a durable design target, not a build order.
The analysis engine is mid-construction; the GUI does not exist yet. Every
element below is grounded in a fact the engine already produces or a
principle the project already holds — and everything the engine cannot yet
back is called out in [Explicitly deferred](#explicitly-deferred), which is
a feature of this document, not an omission.*

Companion mockup: `design/aletheia-gui-mockup.html` (static, does not connect
to the engine).

---

## 1. Positioning & principles as UX commitments

Aletheia's Phase 6 is not "a GUI." It is a **poly-frontend architecture**: a
headless engine that owns all truth, and interchangeable thin clients (native
app, browser UI, TUI) that render it — the Language Server Protocol model,
applied to reverse engineering. The six ROADMAP principles are the entire
design brief. Each one below is restated as a concrete UX commitment this
design must keep.

| ROADMAP principle | UX commitment (what the user actually sees/does) |
|---|---|
| **Frontends compute nothing.** Every on-screen fact comes from the engine over a versioned protocol. | The UI has no analysis code path. If a value is on screen, a protocol message carried it. The frontend may *lay out* (sort, fold, color, position a graph) but never *derive* a boundary, name, operand, or xref. A "recomputed locally" badge is impossible by construction because the code to compute it does not live in the client. |
| **Provenance on demand.** Any fact answers *"why do you believe this?"* down to the bytes and the rule. | Every fact-bearing token has a **Why?** affordance (hover reveals, click pins) that expands the provenance chain to the originating `funcs::Source` / `anchor::Resolution` / xref tier, the byte range, and the rule name. See [§4.1](#41-provenance-on-demand). |
| **Proven vs. heuristic, visibly distinct.** Guesses render differently from metadata-backed facts. | A single, learnable **trust channel** (color + shape + optional `?` glyph) separates *proven* (metadata-backed), *heuristic* (a scan/weak reattachment guessed it), and *asserted* (a human wrote it). See [§4.2](#42-proven-vs-heuristic). |
| **Deterministic and incremental.** Same binary + engine version → byte-identical analysis, any thread count. | Views are stable and diffable: identical inputs produce an identical screen (identical scroll anchors, identical ordering). A version stamp (binary hash + engine version) is always visible; the UI never shows a value whose provenance stamp does not match the loaded analysis. Re-analysis after an edit is presented as an incremental delta, never a blocking full-screen spinner. |
| **The database stays open.** The artifact is the Phase 3 git-mergeable format. | The UI has **no private state that changes analysis**. Names, types, comments, undo/redo all write the `aletheia-annotations v1` log (`annotate::Db`) and nothing else. "Save" is "commit the log"; there is no proprietary project blob. Layout/theme/panel state is cosmetic and stored separately from analysis. |
| **Core stays zero-dep.** Engine + protocol in the dependency-free crate; toolkits live in frontend crates. | A design constraint, not a screen — but it shapes the IA: the protocol boundary in [§6](#6-protocol-implications) is drawn so a native client and a web client render the *same* messages. Nothing in this spec assumes a toolkit. |

The project's one-line thesis — **no silent guessing anywhere in the stack** —
is the through-line. The two hardest UX problems ([§4](#4-the-two-signature-ux-problems))
are both direct expressions of it.

---

## 2. Personas & core tasks

Aletheia is a professional tool for a small, expert audience. The design is
**information-dense and legible**, in the lineage of a code editor and a
debugger — never a consumer app. Three overlapping personas:

- **The RE analyst (primary).** Malware, vuln research, interop. Lives in the
  disassembly, follows xrefs for hours, renames and comments as understanding
  accrues, and — crucially — must know *which facts to trust* before betting
  an exploit or an IOC on them. Provenance is not a nicety for this persona;
  it is the job.
- **The build-comparer.** Ships or hunts across versions: what changed between
  build N and N+1? Drives the diff view (`diff` buckets), carries names
  forward, and needs the tool to say **"uncertain"** rather than guess.
- **The toolsmith / CI operator.** Runs Aletheia headless, scripts it, diffs
  its output in git. The GUI is one client among several; this persona is why
  every view is a deterministic, diffable artifact and why the database is
  the only durable state.

**Core tasks the UI must make fast:**

1. Navigate a binary — jump to a function/address/symbol, walk the call graph.
2. Read symbolized disassembly (`listing::render`) and, later, decompiled
   pseudocode.
3. Follow cross-references both directions (`xref::refs_from` / `refs_to`).
4. Rename, retype, comment — asserted facts into `annotate::Db`; undo/redo.
5. Ask **"why do you believe this?"** of any fact and get the byte-level answer.
6. Diff two builds and triage the buckets.
7. Inspect structured recoveries: C++ vtables/RTTI (`vtable`), Go types/itabs
   (`gotype`), strings (`strings`, `gostrings`), Rust panic-site metadata
   (`rustmeta`).

---

## 3. Information architecture

### 3.1 The workspace

A classic three-region professional layout, tuned for density:

```
┌───────────────────────────────────────────────────────────────────────────┐
│  Top bar: binary identity · engine/version stamp · global search · theme    │
├────────────┬────────────────────────────────────────────┬───────────────────┤
│            │                                            │                   │
│  LEFT      │              CENTER (primary)              │   RIGHT           │
│  Navigator │   Symbolized listing  ⇄  Decompile (later) │   Context         │
│            │                                            │                   │
│  functions │   the hero surface — see §3.2              │   xrefs to/from   │
│  by source │                                            │   provenance pin  │
│  imports   │                                            │   annotations     │
│  strings   │                                            │                   │
│  types     │                                            │                   │
├────────────┴────────────────────────────────────────────┴───────────────────┤
│  Status strip: analysis determinism stamp · caps hit? · legend (trust)       │
└───────────────────────────────────────────────────────────────────────────┘
```

- **Left — Navigator.** The function list, grouped or sortable **by discovery
  source** (`funcs::Source`), so an analyst can see at a glance which functions
  are metadata-backed and which are prologue-scan guesses. Also the entry
  points, imports/exports, strings, and the type/vtable indexes. Every row
  carries the trust channel ([§4.2](#42-proven-vs-heuristic)).
- **Center — the primary surface.** The symbolized listing today; a
  `--decompile` pseudocode view later, as a *toggle over the same function*
  (deferred, see [§9](#explicitly-deferred)). This is the hero screen; [§3.2](#32-the-primary-surface).
- **Right — Context.** Follows the caret. Three stacked panels: **Xrefs**
  (to/from, the two questions `Xrefs::refs_to`/`refs_from` answer), the
  **Provenance pin** (the expanded "why?" for the selected fact,
  [§4.1](#41-provenance-on-demand)), and **Annotations** (this location's
  name/type/comment history from the log).
- **Status strip.** The determinism stamp (binary hash + engine version), a
  visible marker if any resource cap was hit (the listing/analysis caps are a
  safety contract, not a tuning knob — a truncated view must never look
  complete), and the always-on trust legend.

### 3.2 The primary surface (hero)

Renders `listing::render` output as an interactive document, not a text blob.
Column model mirrors the CLI listing exactly so the two are diffable and an
analyst moving between them is never disoriented:

```
address (16-hex, tabular)  ·  raw bytes (elided at 8)  ·  mnemonic + operands  ·  ; comment
```

Load-bearing rendering rules, each inherited from `listing.rs`:

- **Symbolized operands.** Call/branch targets resolve to callee names
  (`call → inflate`), because targets come from `Flow`, which the decoder
  always knows. Symbol names are demangled at display time
  (`demangle`, `cxxdemangle`) — presentation only; the stored name is
  untouched.
- **Honest under-approximation.** An encoding the text formatter cannot render
  yet prints `db <bytes>` **but still symbolizes its control flow** — because
  `Flow` is known even when the mnemonic text is not. The UI shows this as a
  distinct, muted "raw" style so invented mnemonics are impossible.
- **Labels carry inline xrefs** (first few, then a count), matching
  `XREFS_SHOWN`.
- **Referenced strings preview inline** (`STRING_PREVIEW` chars) next to the
  instruction that loads them.
- **Caps are visible.** Hitting the function/line cap ends the view with an
  explicit footer; a folded region never hides the fact that it was cut.

### 3.3 Engine truth vs. frontend rendering (the LSP analogy)

The line the whole architecture depends on. To keep it unambiguous:

| Engine truth (protocol carries it) | Frontend rendering (client decides) |
|---|---|
| Function boundaries + their `Source` | Sort/group order of the Navigator |
| Instruction bytes, decoded operands, `Flow` | Syntax coloring, column widths, font |
| Symbolized target of a branch/call | Whether to show the numeric address too |
| Xref set for an address (both directions) | Panel layout, elision counts |
| `anchor::Resolution` of a reattached name | The `?` glyph and trust color choice |
| Diff `MatchKind` / `Uncertainty` bucket | Which bucket is expanded, its color |
| The annotation log + its fold | Undo affordance, keybindings |
| CFG edges (basic blocks, successors) | **Graph layout/positioning** (see [§9](#explicitly-deferred)) |

The rule of thumb an implementer can apply: **if getting it wrong would state
a falsehood about the binary, it is engine truth and must arrive over the
protocol. If getting it wrong is only ugly, it is the frontend's call.**

---

## 4. The two signature UX problems

These two are the differentiator. They get the most care, and the mockup shows
both in situ.

### 4.1 Provenance on demand

**The claim:** any displayed fact answers *"why do you believe this?"* down to
the bytes and the rule. The engine already carries the evidence —
`funcs::Source` tags every function start, `anchor::Resolution` records how
every stored annotation reattached, and `xref` classifies every reference as
exact or heuristic. The UI's job is to make that evidence one gesture away.

**The interaction:**

- Every fact-bearing token (a function name, a boundary, a symbolized operand,
  an xref, a diff verdict) has a **Why?** affordance. Hover surfaces a
  one-line summary; click **pins** the full chain into the right-panel
  Provenance pin so the analyst can keep reading while it stays open.
- The pin renders the **provenance chain**, from the human-facing claim down
  to the bytes, as an ordered, honest list. Nothing is summarized away; the
  last row is always physical.

**Worked example — a proven function name.** Claim: this function is `inflate`.

```
CLAIM     name = "inflate"  @ 0x0000000000401010
SOURCE    funcs::Source::Symbol   — ELF .symtab, Function-kind symbol in an
                                     executable region
IDENTITY  anchor  shape=0x9f3e_… (rebase-invariant)  bytes=0x1a2c_…
                  insns=41   resolution = Exact(0x401010)
BYTES     0x401010 + 12   55 48 89 e5 41 57 41 56 …   (open hex view →)
VERDICT   proven — metadata-backed, exact byte match
```

**Worked example — a heuristic boundary.** Claim: a function starts at
`0x4012a0`.

```
CLAIM     function start @ 0x00000000004012a0
SOURCE    funcs::Source::Prologue  — the only heuristic source; used last,
                                      only for addresses the first three missed
RULE      matched a standard prologue  push rbp ; mov rbp, rsp
NEGATIVE  no Symbol, no Unwind entry (.pdata/.eh_frame/FUNCTION_STARTS), and
          no Go pclntab entry covers this address
VERDICT   heuristic — verify before relying on it
```

The **NEGATIVE** row matters: honest provenance shows not only what fired but
what *didn't*. A prologue guess is exactly "no higher-precedence source found
this," and the analyst should see that framed as such.

**Worked example — a weakly reattached name after a rebuild.** Claim: the name
`decode_frame` still applies to this function in the new build.

```
CLAIM     name = "decode_frame"  (asserted)   reattached to 0x0000000000452118
SOURCE    annotate::Db assertion → anchor resolution against this program
RESOLUTION anchor::Resolution::Shape — rebase-invariant shape matched, unique
           after instruction-count filtering (bytes differ: the build changed)
VERDICT   asserted, reattached by shape — carried, not proven about these bytes
```

If resolution had been `Absolute` or `Ambiguous`, the same pin would show the
weaker tier and, for `Ambiguous`, list the candidate entry VAs the engine
refused to choose between — mirroring how the CLI listing marks a weak
reattachment with `?` and an explaining note line. Provenance and the trust
channel are the same fact told at two levels of detail.

### 4.2 Proven vs. heuristic (the trust channel)

**The claim:** proven and heuristic facts must look different at a glance,
everywhere, consistently — one learnable visual language, not a per-view
special case. Three trust states cover the whole engine:

| Trust state | What it means | Backed by | Visual treatment |
|---|---|---|---|
| **Proven** | Metadata says so; reproducible from the bytes. | `Source::{EntryPoint, Symbol, Unwind, GoPclntab}`; `Resolution::{Exact, Shape}`; xref **exact** tier (target from the encoding/`Flow`) | Normal weight, calm foreground, a small **solid** trust dot. No `?`. This is the restful default so heuristics stand out. |
| **Heuristic** | A scan or a weak reattachment guessed it; could be wrong. | `Source::Prologue`; `Resolution::{Absolute, Ambiguous}`; xref heuristic tier | Amber accent + a **hollow/dashed** dot + a trailing **`?`** glyph, exactly the listing's marker. Never hidden, never silent. |
| **Asserted** | A human wrote it; not recomputable, lives in the log. | `annotate::Db` name/type/comment | A distinct **cool accent** (blue-violet) + a small pen glyph. Orthogonal to proven/heuristic — an asserted name can itself be reattached proven-by-shape or weakly, and the provenance pin says which. |

Design rules for the channel:

- **Color is never the only signal** (accessibility, and the project's own
  no-silent-guessing ethic). Trust always pairs color with **shape** (solid vs
  hollow/dashed dot) and, for heuristics, the **`?` glyph**. It survives
  grayscale, colorblindness, and a screenshot pasted into a bug report.
- **One channel, everywhere.** The same three-state language colors Navigator
  rows, listing tokens, xref entries, and diff pairs. Learn it once.
- **The trust channel is distinct from the diff channel** ([§5](#5-the-diff-view)).
  Diff buckets are a *second* semantic axis and use a separate hue set;
  the spec keeps them from colliding so a "modified" function and a "heuristic"
  boundary never read as the same thing.

---

## 5. The diff view

Drives the build-comparer persona; renders `diff` directly. Two functions,
old and new, matched by the anchor layer — **no new heuristics in the view**,
exactly as `diff.rs` contributes none. Six buckets, each a distinct color on
the **diff axis** (separate from the trust axis):

| Bucket | Evidence | Color intent |
|---|---|---|
| **Unchanged** | exact bytes, same VA | neutral / receding |
| **Moved** | exact bytes, different VA (linker relocated it) | cool blue |
| **Modified** | shape match, bytes differ (patched constant, retargeted call, rebased data) | amber |
| **Uncertain** | resolved only by address, or ambiguous, or contested — with the tier shown | hatched amber-orange (visibly "not a clean answer") |
| **Added** | new function nothing old matched | green |
| **Removed** | old function nothing new matched | red |

Honesty rules the diff view inherits from the engine:

- **Uncertain is a first-class, visible bucket**, never rounded up to a match.
  Its `Uncertainty` note is shown verbatim: *address-only match*,
  *ambiguous: 0x…, 0x… (N more)*, or *contested; claimed by 0x…*. A wrong
  confident answer costs an analyst more than an honest "unsure."
- **Carried names are shown as proposals, not writes.** When a confident match
  has an old name and the new side has none, `Pair::carried_name` is offered
  with an explicit **Apply** action; nothing is written until the analyst
  accepts, and accepting writes the annotation log like any other assertion.
- The report is deterministic and diffable, so the view has a **stable order**
  and can itself be exported and committed.

---

## 6. Interaction model

- **Keyboard-first.** A pro tool is driven from the keyboard: `g` to go to
  address/symbol, `x` for xrefs, `n` to rename, `;` to comment, `y`/`u` for
  the decompile/listing toggle (when the decompile view lands), `?` to pin the
  provenance of the token under the caret. Every mouse affordance has a key.
- **Navigation is address-anchored and reversible.** Following an xref pushes a
  back-stack; `Esc`/back returns to the exact prior caret and scroll. Because
  views are deterministic, "back" always lands on the same bytes it left.
- **Rename / retype / comment = one assertion each.** Editing a name opens an
  inline field on the token; committing appends one record to `annotate::Db`
  keyed by `Anchor`, not by absolute address, so the edit survives a rebuild.
  The three editable fields are exactly the engine's: `Field::{Name, Type,
  Comment}`.
- **Undo/redo is the log's own operation.** `Ctrl-Z`/`Ctrl-Shift-Z` map
  directly to `annotate::Db::undo`/`redo` — undo moves the most recent
  assertion to the redo stack, a fresh assertion clears redo. The UI does not
  invent an undo model; it exposes the one the database already has. This means
  undo history is durable and diffable, not session-local.
- **No destructive surprises.** Because computed facts are never stored and
  asserted facts are an append-only log, "losing work" has no failure mode the
  UI must guard — but applying a carried name or a bulk rename still confirms,
  and every such action is itself undoable via the log.

---

## 7. Protocol implications

*Framed as requirements/questions for the real protocol design, per the brief
— not a finished protocol. The engine→frontend protocol must let the UI
compute nothing, which means it must **carry every fact plus its provenance**.*

For the UI in this spec to render without deriving anything, the protocol must
carry, at minimum:

1. **Facts with their provenance inline.** A function is not just
   `{va, name}`; it is `{va, name, source: Source, anchor_identity}`. An xref
   is not just `{from, to}`; it carries its **exactness tier**. A reattached
   name carries its `Resolution`. Provenance is not a second request — it rides
   with the fact, or the "compute nothing" rule leaks.
   *Open question: does the byte-level evidence (the actual `push rbp` bytes,
   the symtab index) ride inline too, or is it a cheap follow-up "explain this
   fact" request keyed by a fact id? Inline is simpler for the client; lazy is
   cheaper on the wire. Likely: summary inline, raw bytes on demand.*
2. **A determinism stamp on every response.** Binary content hash + engine
   version, so the client can prove a value belongs to the loaded analysis and
   refuse to blend two.
3. **Assertions as the only write path.** The client's sole mutation is
   "append this assertion to the log"; the engine folds and returns the new
   state. No other client→engine write exists, which is what keeps the database
   the single source of truth.
4. **Incremental deltas.** After an assertion or a re-analysis, the protocol
   should express *what changed* (which functions' facts, which reattachments
   moved tier) so the UI updates in place without a full refetch and without
   blocking. *Open question: delta granularity — per-function, per-address, or
   per-fact?*
5. **Capability + version negotiation.** The client asks what the engine
   version can answer (does it have `--decompile`? aarch64 SSA? devirt?), so a
   client never renders an affordance the engine cannot back. This is the
   protocol-level expression of the whole deferred section below.
6. **Stable ids for facts**, so a pinned provenance chain, a back-stack entry,
   and an incremental delta all refer to the same thing across responses.

---

## 8. Visual language direction

- **Theme.** Dark-first (the working default for this audience), light fully
  supported and designed *in parallel* — not an inverted afterthought. Both
  themes carry the trust and diff palettes at verified contrast.
- **Density.** High, but legible. This is a debugger/editor, not a marketing
  page. Comfortable line height for a monospace body, tight but consistent
  4/8px spacing rhythm, no wasted vertical space in the listing.
- **Type.** A **monospace** face for all code, addresses, bytes, and
  pseudocode (tabular figures so address and byte columns never shift). A clean
  humanist **sans** for chrome — panel titles, labels, buttons. Two families,
  strict roles.
- **Color, used semantically, on two orthogonal axes:**
  - **Trust axis** — *proven* (calm neutral/teal, solid), *heuristic* (amber,
    hollow + `?`), *asserted* (blue-violet, pen). The restful default is
    proven, so the eye is drawn to what to doubt.
  - **Diff axis** — unchanged/moved/modified/uncertain/added/removed, a
    separate hue set kept from colliding with the trust hues.
  - Syntax coloring in the listing is a *third*, low-saturation layer
    (mnemonics, registers, immediates, strings) that must never compete with
    the trust channel — trust wins the eye.
- **Every semantic is a token, not a raw hex in a component**, defined once
  per theme, so proven/heuristic/asserted and the six diff buckets stay
  consistent across every view and both themes.
- **Color is never load-bearing alone** — shape and glyph always accompany it.

---

## Explicitly deferred

*Honest inventory as of the 2026-08-16 multi-track replan. Items that have
since landed are struck through so this section stays a living gate, not a
stale freeze.*

- ~~**Full decompiled pseudocode view (`--decompile`).**~~ **Landed** —
  `irstruct` / `irout` / `pseudo` + `redump --decompile` on x86-64 and
  aarch64. The center-panel toggle is now engine-backed; deepen
  (`irstack` / `sig` / `irtype`) improves readability, not existence.
- **CFG / call-graph *visual graph*.** Engine produces edges; layout still
  frontend-only and undesigned.
- **Type inference & member-layout recovery.** `vtable` / `gotype` displayable;
  IR-level `irtype` / member layout / Rust trait-object resolve still open.
- ~~**AArch64 SSA / decompilation.**~~ **Landed** — `irlift` dispatches
  aarch64; do not gray out the decompile toggle for arm64.
- **Signature / FLIRT-style library identification.** Still open.
- **The wire protocol itself.** Requirements in §7; schema now starts as
  `protocol/PROTOCOL.md` + `aletheia-mcp` (Track 4) — still evolve from TUI
  use, but no longer "no schema at all."
- **Incremental re-analysis UX specifics.** Still deferred on delta granularity.
- **Multi-user / real-time collaboration.** Git branch-and-merge only.
- ~~**Debugging / dynamic analysis / patching.**~~ **Patching in scope (P1)** —
  PatchSet preview/apply (Track 3). Debugging / dynamic analysis remain
  non-goals. FairPlay decrypt remains out of engine.
