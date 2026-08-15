export const meta = {
  name: 'decompiler-research',
  description: 'Literature-grounded design phase for the Reforge decompiler: per-topic research agents checkpoint to disk, then one synthesis agent writes DESIGN.md',
  whenToUse: 'Run to execute the decompiler research/design phase. Safe to re-run: existing topic files are extended, not clobbered, and partial results survive on disk.',
  phases: [
    { title: 'Research', detail: 'one agent per topic; each writes research/decompiler/<slug>.md BEFORE reporting back' },
    { title: 'Synthesize', detail: 'one agent reads all topic files on disk and writes research/decompiler/DESIGN.md' },
  ],
}

const REPO = '/Users/aaronsarfati/Desktop/REFORGE'
const DIR = REPO + '/research/decompiler'

const TOPICS = [
  { slug: 'ssa-optimization', title: 'SSA-based simplification for decompilation', focus: 'cross-block expression/copy/constant propagation on SSA, dead-code elimination, and out-of-SSA translation. Baselines: Cytron et al.; van Emmerik thesis (SSA for decompilation). How this extends the existing per-block passes in src/irflow.rs once src/irssa.rs exists.' },
  { slug: 'structuring', title: 'Control-flow structuring', focus: 'interval/structural analysis (Cifuentes), Phoenix (Schwartz et al. 2013), pattern-independent structuring / No More Gotos (Yakdan et al. 2015), rev.ng comb. Goto minimization vs semantic fidelity; irreducible CFGs; how truncated blocks from irlift must be surfaced honestly.' },
  { slug: 'type-recovery', title: 'Type recovery on low-level code', focus: 'TIE (Lee et al. 2011), Retypd (Noonan et al. 2016), unification vs subtyping-based inference; which conclusions are sound vs heuristic, matching the repo proven-vs-heuristic convention.' },
  { slug: 'variable-recovery', title: 'Stack-slot and variable recovery', focus: 'stack layout analysis, value-set analysis / DIVINE lineage (Balakrishnan and Reps), memory SSA; conservative aliasing that never invents variables it cannot justify.' },
  { slug: 'calling-conventions', title: 'Calling-convention and signature recovery', focus: 'ABI-knowledge-driven vs dataflow-inferred parameter/return recovery for x86-64 SysV/Win64 and AArch64 AAPCS64; varargs and mixed cases; how existing funcs/cfg metadata feeds it.' },
  { slug: 'pseudocode-emission', title: 'Pseudocode emission and readability', focus: 'expression-tree construction from SSA, precedence and parenthesization, deterministic render, readability results from the Dream++ line of work; what the existing ir render conventions imply for the pseudocode renderer.' },
  { slug: 'incumbent-architectures', title: 'Architecture survey of open decompiler systems', focus: 'Ghidra (open source, public design docs), RetDec, angr: pipeline shapes, IR choices, published evaluations of their output quality. Architecture lessons only; strictly no proprietary internals of any closed tool.' },
]

const COMMON = 'You are doing published-literature research for Reforge (' + REPO + '), a clean-room, dependency-free Rust binary-analysis toolkit. FIRST read ' + DIR + '/BRIEF.md (the ground rules bind you), then skim the module doc-comments of src/ir.rs, src/irlift.rs, src/irflow.rs, src/cfg.rs so recommendations map onto the real contracts (width-explicit IR, total check functions, deterministic render, no external deps, proven-vs-heuristic honesty, resource caps, no panics). Sources: published academic papers/theses and openly documented open-source projects only — never proprietary internals of closed tools.'

phase('Research')
const done = await parallel(TOPICS.map(t => () =>
  agent(COMMON + '\n\nTopic: ' + t.title + '.\nFocus: ' + t.focus + '\n\nIf ' + DIR + '/' + t.slug + '.md already exists and is non-empty, read it and improve/extend it rather than starting over.\n\nWrite your findings to ' + DIR + '/' + t.slug + '.md following the per-topic structure in BRIEF.md. Write the file BEFORE composing your final reply, so the work survives any interruption. Then return one short paragraph stating your concrete recommendation for Reforge.', { label: t.slug, phase: 'Research' })
))
const completed = done.filter(Boolean).length
log(completed + '/' + TOPICS.length + ' topic agents returned; every finished topic is already checkpointed under research/decompiler/')

phase('Synthesize')
const design = await agent(COMMON + '\n\nList ' + DIR + ' and read every topic .md that exists (some may be missing — note gaps honestly in the output rather than inventing content for them). Also check git log and any PLAN_*.md at the repo root for slices that already landed (SSA/irssa may already exist).\n\nWrite ' + DIR + '/DESIGN.md: an engineering design for the Reforge decompiler as an ORDERED sequence of one-commit slices on top of ir/irlift/irflow/cfg (+irssa if landed). For each slice: goal, module name, chosen algorithm with citation, invariants and what its total check enforces, test matrix, exit criteria. Keep repo honesty conventions throughout. Write the file BEFORE composing your reply, then return a one-paragraph summary of the slice sequence.', { label: 'synthesize', phase: 'Synthesize' })

return { topicsCompleted: completed, totalTopics: TOPICS.length, designSummary: design }
