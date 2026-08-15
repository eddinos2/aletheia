//! Evaluation harness — ground-truth fixtures and the three fidelity
//! metrics (DESIGN slice 19).
//!
//! Five fidelity passes iterate on the same corpus numbers, and until
//! now every wave re-derived them by hand. This module adopts the
//! published evaluation methodology as regression infrastructure: one
//! command — `cargo test evalfx`, a naming convention, not new CLI —
//! reports, per checked-in fixture, and fails when a number moves:
//!
//! - **Goto count** (SAILR's metric): [`irstruct::StructStats::gotos`]
//!   after the full pipeline — construct → optimize → forward →
//!   eliminate_dead → structure — exactly the `--decompile` wiring.
//! - **CFGED-lite**: a bounded *exact* graph-edit-distance between the
//!   recovered CFG and the fixture's hand-written source CFG.
//!   Dependency-free and exact-or-refuse: a graph over
//!   [`GED_NODE_CAP`] nodes is a documented non-metric (`None`), never
//!   an approximation. The recovered graph is read off the structured
//!   function's blocks and successors, which [`irstruct::check`]
//!   proves is precisely the edge set the structure tree realizes —
//!   so this *is* the tree's CFG-shape, not a separate account of it.
//! - **Semantic spot checks** (Liu & Wang, ISSTA 2020 lineage): at
//!   every pipeline stage, slice 7's SSA interpreter
//!   ([`irout::tests::Interp`]) runs the function's SSA reading and
//!   its out-of-SSA rendition side by side on seeded inputs and
//!   demands equality of every observable value — behavior as a
//!   regression bit, before and after each optimization. The
//!   interpreter stays `#[cfg(test)]` and crate-private; this module's
//!   placement in `src/` (rather than Cargo's integration-test layout
//!   DESIGN sketched as "`tests/` fixtures") follows from exactly that
//!   constraint, and `fixtures/` satisfies the fixture half of it.
//!
//! The fixture binaries are compiled **offline** from the checked-in C
//! and committed as bytes; `fixtures/README.md` records the compiler
//! version and exact commands verbatim (the provenance rule — no
//! build-time deps). The real-binary sweeps (/bin/ls, /bin/bash,
//! libbrotlidec) are deliberately *not* here: they remain
//! exit-criteria material, summarized in ROADMAP.md, never asserted.
//!
//! Every expected number lives in [`FIXTURES`] — one table, one file.
//! A fidelity slice that legitimately moves a metric updates that
//! table in the same commit, and nothing else: that one-place update
//! friction is the design.

use std::collections::{BTreeMap, BTreeSet};

use crate::irssa::SsaFunction;
use crate::irstruct::Node;
use crate::model::Arch;
use crate::{callfx, irlift, irout, irssa, irssaopt, irstruct, jumptable};

// -- the one expected-number table ------------------------------------------

/// A fixture's hand-written source-level CFG: `nodes` vertices named
/// `0..nodes`, entry `0`, and the directed `edges` — the GED ground
/// truth, transcribed from the `.c` header comment next to the binary.
struct SourceCfg {
    nodes: usize,
    edges: &'static [(usize, usize)],
}

/// One checked-in fixture and every number the harness asserts for it.
struct Fixture {
    /// Binary under `fixtures/`.
    file: &'static str,
    /// Symbol of the function under evaluation (Mach-O spelling).
    symbol: &'static str,
    /// The known source CFG.
    source: SourceCfg,
    /// Expected goto count through the full pipeline.
    gotos: usize,
    /// Expected exact CFGED against `source`; `None` is the documented
    /// refusal — the graph exceeds [`GED_NODE_CAP`], asserted, so a
    /// shrunken recovery can never hide behind a refusal.
    cfged: Option<usize>,
    /// `Switch` nodes the structure tree must contain (the dense
    /// fixture's proof that a jump table was emitted and recovered).
    switches: usize,
}

/// The corpus. Numbers measured at the harness's introduction; a
/// companion slice that legitimately moves one updates it here.
const FIXTURES: &[Fixture] = &[
    Fixture {
        file: "diamond",
        symbol: "_diamond",
        source: SourceCfg {
            nodes: 4,
            edges: &[(0, 1), (0, 2), (1, 3), (2, 3)],
        },
        gotos: 0,
        cfged: Some(0),
        switches: 0,
    },
    Fixture {
        file: "switch_dense",
        symbol: "_switch_dense",
        source: SourceCfg {
            nodes: 9,
            edges: &[
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (0, 6),
                (0, 7),
                (1, 8),
                (2, 8),
                (3, 8),
                (4, 8),
                (5, 8),
                (6, 8),
                (7, 8),
            ],
        },
        gotos: 0,
        cfged: None,
        switches: 1,
    },
    Fixture {
        file: "loop_bc",
        symbol: "_loop_bc",
        source: SourceCfg {
            nodes: 7,
            edges: &[
                (0, 1),
                (1, 2),
                (1, 6),
                (2, 5),
                (2, 3),
                (3, 6),
                (3, 4),
                (4, 5),
                (5, 1),
            ],
        },
        gotos: 0,
        cfged: Some(6),
        switches: 0,
    },
    Fixture {
        file: "tail_merge",
        symbol: "_tail_merge",
        source: SourceCfg {
            nodes: 5,
            edges: &[(0, 1), (0, 4), (1, 2), (1, 3)],
        },
        // 2 -> 1 when the φ-web narrowing landed: one cross-jumped
        // tail's convergence edge carries no rendered copies once
        // coalescence is consulted, so its re-split is sanctioned; the
        // other still carries one and honestly keeps its goto.
        gotos: 1,
        cfged: Some(6),
        switches: 0,
    },
    Fixture {
        file: "shortcircuit",
        symbol: "_shortcircuit",
        source: SourceCfg {
            nodes: 6,
            edges: &[
                (0, 1),
                (0, 2),
                (1, 3),
                (1, 2),
                (2, 3),
                (2, 4),
                (3, 5),
                (4, 5),
            ],
        },
        // 2 -> 1 when the φ-web narrowing landed, same sanction as
        // tail_merge's: the flattened middle's re-join edge proves
        // copy-free under coalescence.
        gotos: 1,
        cfged: Some(2),
        switches: 0,
    },
];

// -- CFGED-lite: bounded exact graph edit distance --------------------------

/// Refusal bound: exact GED is exponential, so a graph over this many
/// nodes is refused, explicitly, rather than approximated.
const GED_NODE_CAP: usize = 8;

/// A tiny directed graph: vertices `0..n`, unlabeled, edge set exact.
#[derive(Clone, PartialEq, Eq)]
struct Digraph {
    n: usize,
    edges: BTreeSet<(usize, usize)>,
}

impl Digraph {
    fn new(n: usize, edges: &[(usize, usize)]) -> Digraph {
        let edges: BTreeSet<(usize, usize)> = edges.iter().copied().collect();
        assert!(
            edges.iter().all(|&(u, v)| u < n && v < n),
            "an edge names a vertex the graph does not hold"
        );
        Digraph { n, edges }
    }
}

/// Exact unit-cost graph edit distance between two unlabeled directed
/// graphs — node insert/delete and edge insert/delete, one each — or
/// `None` when either side exceeds [`GED_NODE_CAP`].
///
/// Depth-first over all partial injective vertex mappings (each vertex
/// of `a` maps to an unused vertex of `b` or is deleted), branch and
/// bound. Small and exact is the contract: the search is complete, so
/// the minimum is the true distance, and anything too big to search
/// completely is refused.
fn ged(a: &Digraph, b: &Digraph) -> Option<usize> {
    if a.n > GED_NODE_CAP || b.n > GED_NODE_CAP {
        return None;
    }
    // Delete everything, insert everything: always achievable.
    let mut best = a.n + b.n + a.edges.len() + b.edges.len();
    let mut map: Vec<Option<usize>> = Vec::with_capacity(a.n);
    let mut used = vec![false; b.n];
    ged_search(a, b, &mut map, &mut used, 0, 0, &mut best);
    Some(best)
}

/// Extend the partial mapping at vertex `map.len()`. `cost` carries the
/// node deletions and the a-edge deletions/matches decided so far;
/// `matched` counts a-edges whose images are b-edges. Insertions for
/// unmapped b-vertices and unmatched b-edges are settled at the leaf.
fn ged_search(
    a: &Digraph,
    b: &Digraph,
    map: &mut Vec<Option<usize>>,
    used: &mut [bool],
    cost: usize,
    matched: usize,
    best: &mut usize,
) {
    let i = map.len();
    let assigned = map.iter().flatten().count();
    // Even if every remaining a-vertex maps, this many b-vertices must
    // still be inserted — a sound lower bound for the cut.
    let must_insert = b.n.saturating_sub(assigned + (a.n - i));
    if cost + must_insert >= *best {
        return;
    }
    if i == a.n {
        let total = cost + (b.n - assigned) + (b.edges.len() - matched);
        *best = (*best).min(total);
        return;
    }
    // The incremental edge cost of deciding vertex `i`: every a-edge
    // whose later endpoint is `i` (self-loop included) is now decided —
    // matched if both ends map onto a b-edge, deleted otherwise.
    let decide = |to: Option<usize>,
                      map: &mut Vec<Option<usize>>,
                      used: &mut [bool],
                      best: &mut usize| {
        let mut extra = usize::from(to.is_none());
        let mut hits = 0;
        for &(u, v) in &a.edges {
            if u.max(v) != i {
                continue;
            }
            let (mu, mv) = (
                if u == i { to } else { map[u] },
                if v == i { to } else { map[v] },
            );
            match (mu, mv) {
                (Some(x), Some(y)) if b.edges.contains(&(x, y)) => hits += 1,
                _ => extra += 1,
            }
        }
        map.push(to);
        if let Some(j) = to {
            used[j] = true;
        }
        ged_search(a, b, map, used, cost + extra, matched + hits, best);
        if let Some(j) = to {
            used[j] = false;
        }
        map.pop();
    };
    for j in 0..b.n {
        if !used[j] {
            decide(Some(j), map, used, best);
        }
    }
    decide(None, map, used, best);
}

// -- the pipeline under measurement -----------------------------------------

/// Everything one fixture run produces: the four pipeline stages by
/// name (for the per-stage semantic checks) and the structuring result.
struct Run {
    stages: Vec<(&'static str, SsaFunction)>,
    root: Node,
    stats: irstruct::StructStats,
}

/// The `--decompile` pipeline, verbatim: load, recover to the
/// jump-table fixpoint, lift the fixture's function, apply the ABI's
/// call effects, then construct → optimize → forward → eliminate_dead
/// → structure. Every stage must pass its own `check`; a fixture is
/// checked in, so any failure to load or lift is a hard failure, not a
/// skip.
fn pipeline(fx: &Fixture) -> Run {
    let path = format!("{}/fixtures/{}", env!("CARGO_MANIFEST_DIR"), fx.file);
    let data = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("{}: the fixture binary is checked in: {e}", fx.file));
    let image = crate::load(&data).unwrap_or_else(|e| panic!("{}: {e}", fx.file));
    assert_eq!(image.arch(), Arch::X86_64, "{}: x86-64 fixtures only", fx.file);

    let folded = jumptable::resolve_folded(image.as_ref())
        .unwrap_or_else(|e| panic!("{}: {e}", fx.file));
    assert!(!folded.capped, "{}: table folding must not cap", fx.file);
    let tables = jumptable::successor_map(&folded.tables);

    let entry = image
        .symbols()
        .iter()
        .find(|s| s.name == fx.symbol)
        .unwrap_or_else(|| panic!("{}: no symbol {}", fx.file, fx.symbol))
        .va;
    let func = folded
        .program
        .functions
        .get(&entry)
        .unwrap_or_else(|| panic!("{}: no function at {entry:#x}", fx.file));

    let lifted = irlift::lift_function(image.as_ref(), func)
        .unwrap_or_else(|| panic!("{}: refused to lift", fx.file));
    let abi = callfx::abi_for(image.arch()).expect("x86-64 has an ABI table");
    let lifted = callfx::apply(&lifted, &abi);

    let ssa = irssa::construct(&lifted).unwrap_or_else(|e| panic!("{}: no ssa ({e})", fx.file));
    let (opt, _) = irssaopt::optimize(&ssa);
    let (fwd, _) = irssaopt::forward(&opt);
    let live_out = callfx::function_live_out(image.arch()).unwrap_or_default();
    let (swept, _) = irssaopt::eliminate_dead(&fwd, &live_out);
    let stages = vec![
        ("construct", ssa),
        ("optimize", opt),
        ("forward", fwd),
        ("eliminate_dead", swept),
    ];
    for (stage, f) in &stages {
        assert_eq!(irssa::check(f), Ok(()), "{}/{stage} must check", fx.file);
    }

    let f = &stages.last().expect("four stages").1;
    let (root, stats) = irstruct::structure(f, &tables);
    assert!(!stats.capped, "{}: structuring must not cap", fx.file);
    assert_eq!(
        irstruct::check(f, &tables, &root),
        Ok(()),
        "{}: the structure tree must check",
        fx.file
    );
    Run { stages, root, stats }
}

/// The structured function's CFG — blocks in VA order, successor edges
/// restricted to in-function blocks — which [`irstruct::check`]
/// guarantees is exactly the edge set the tree realizes.
fn recovered(f: &SsaFunction) -> Digraph {
    let index: BTreeMap<u64, usize> = f
        .blocks
        .keys()
        .enumerate()
        .map(|(i, &va)| (va, i))
        .collect();
    let edges: BTreeSet<(usize, usize)> = f
        .blocks
        .iter()
        .flat_map(|(&va, b)| {
            let index = &index;
            b.successors
                .iter()
                .filter_map(move |s| Some((index[&va], *index.get(s)?)))
        })
        .collect();
    Digraph {
        n: index.len(),
        edges,
    }
}

/// `Switch` nodes in a structure tree.
fn switches(node: &Node) -> usize {
    match node {
        Node::Block(_) | Node::Break | Node::Continue | Node::Goto(_) | Node::Opaque { .. } => 0,
        Node::Seq(children) => children.iter().map(switches).sum(),
        Node::If {
            then_body,
            else_body,
            ..
        } => switches(then_body) + else_body.as_deref().map_or(0, switches),
        Node::Loop { body, .. } => switches(body),
        Node::Switch { cases, .. } => 1 + cases.iter().map(|(_, n)| switches(n)).sum::<usize>(),
    }
}

/// The interpreter's seeds — the same four the `irout` battery uses.
const SEEDS: [u64; 4] = [0x1234_5678_9ABC_DEF0, 0xDEAD_BEEF_CAFE_F00D, 7, 99];

/// The semantic spot check for one pipeline stage: translate out of
/// SSA, insist the rendition checks, then run slice 7's interpreter —
/// the SSA reading against the rendition reading on seeded inputs —
/// and demand zero divergences.
fn spot_check(file: &str, stage: &str, f: &SsaFunction) {
    let (out, _) = irout::out_of_ssa(f);
    assert_eq!(
        irout::check(f, &out),
        Ok(()),
        "{file}/{stage}: the rendition must check"
    );
    for seed in SEEDS {
        let faults = irout::tests::Interp::new(f, &out, seed).run(64);
        assert!(
            faults.is_empty(),
            "{file}/{stage}: semantic divergence on seed {seed:#x}: {faults:?}"
        );
    }
}

/// One line per fixture — every measured number, rendered the same way
/// every run. The determinism test compares two of these byte for
/// byte; a human reads it with `--nocapture`.
fn report() -> String {
    let mut s = String::from("fixture       blocks edges gotos cfged switches\n");
    for fx in FIXTURES {
        let run = pipeline(fx);
        let g = recovered(&run.stages.last().expect("four stages").1);
        let source = Digraph::new(fx.source.nodes, fx.source.edges);
        let d = match ged(&g, &source) {
            Some(d) => d.to_string(),
            None => "refused".to_string(),
        };
        s.push_str(&format!(
            "{:<13} {:>6} {:>5} {:>5} {:>7} {:>8}\n",
            fx.file,
            g.n,
            g.edges.len(),
            run.stats.gotos,
            d,
            switches(&run.root),
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- 1: the table — every fixture, every metric ------------------------

    /// The whole corpus against the one expected-number table: goto
    /// count, CFGED-lite, and the switch obligation, per fixture. A
    /// `None` CFGED must be a *real* refusal — one side over the cap —
    /// so shrinkage can never hide behind it.
    #[test]
    fn the_expected_number_table_holds() {
        print!("{}", report());
        for fx in FIXTURES {
            let run = pipeline(fx);
            let f = &run.stages.last().expect("four stages").1;
            assert_eq!(
                run.stats.gotos, fx.gotos,
                "{}: goto count moved (expected {}, measured {})",
                fx.file, fx.gotos, run.stats.gotos
            );
            let g = recovered(f);
            let source = Digraph::new(fx.source.nodes, fx.source.edges);
            let d = ged(&g, &source);
            assert_eq!(
                d, fx.cfged,
                "{}: CFGED moved (expected {:?}, measured {:?})",
                fx.file, fx.cfged, d
            );
            if fx.cfged.is_none() {
                assert!(
                    g.n > GED_NODE_CAP || fx.source.nodes > GED_NODE_CAP,
                    "{}: a refusal must be a real over-cap refusal",
                    fx.file
                );
            }
            assert_eq!(
                switches(&run.root),
                fx.switches,
                "{}: switch obligation",
                fx.file
            );
        }
    }

    // -- 2: the metric itself, against hand-computed distances -------------

    /// Exactness on distances small enough to verify by hand. Each
    /// value below was computed on paper before the implementation ran.
    #[test]
    fn cfged_is_exact_on_hand_computed_distances() {
        let diamond = Digraph::new(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let triangle = Digraph::new(3, &[(0, 1), (0, 2), (1, 2)]);
        let p2 = Digraph::new(2, &[(0, 1)]);
        let dot = Digraph::new(1, &[]);
        let hoop = Digraph::new(1, &[(0, 0)]);
        let path = Digraph::new(3, &[(0, 1), (1, 2)]);
        let htap = Digraph::new(3, &[(2, 1), (1, 0)]);

        // Identity.
        assert_eq!(ged(&diamond, &diamond), Some(0));
        assert_eq!(ged(&dot, &dot), Some(0));
        // Isomorphic but relabeled: still zero — the search is complete.
        assert_eq!(ged(&path, &htap), Some(0));
        // One node and its edge.
        assert_eq!(ged(&p2, &dot), Some(2));
        // A self-loop is one edge deletion.
        assert_eq!(ged(&hoop, &dot), Some(1));
        // Diamond vs triangle: one node op and three edge ops, however
        // the mapping is chosen (verified by case analysis on paper).
        assert_eq!(ged(&diamond, &triangle), Some(4));
        // The metric is symmetric: the edit script inverts.
        assert_eq!(ged(&triangle, &diamond), Some(4));
        // Empty vs everything: pure insertion.
        let empty = Digraph::new(0, &[]);
        assert_eq!(ged(&empty, &diamond), Some(8));
    }

    /// The bound is a refusal, not an approximation: either side over
    /// the cap yields `None`, and just at the cap does not.
    #[test]
    fn cfged_refuses_an_oversized_graph_explicitly() {
        let big = Digraph::new(GED_NODE_CAP + 1, &[]);
        let small = Digraph::new(2, &[(0, 1)]);
        assert_eq!(ged(&big, &small), None);
        assert_eq!(ged(&small, &big), None);
        let at_cap = Digraph::new(GED_NODE_CAP, &[]);
        assert!(ged(&at_cap, &small).is_some());
    }

    // -- 3: semantics at every stage ---------------------------------------

    /// Liu & Wang-style spot checks across the corpus: at each of the
    /// four pipeline stages, the SSA reading and its out-of-SSA
    /// rendition agree on every observable value over all seeds.
    #[test]
    fn semantic_spot_checks_hold_at_every_stage() {
        for fx in FIXTURES {
            for (stage, f) in &pipeline(fx).stages {
                spot_check(fx.file, stage, f);
            }
        }
    }

    // -- 4: the harness itself is deterministic ----------------------------

    /// Two full sweeps agree byte for byte — the report, the trees, the
    /// stats. A metric that flickers is not a regression guard.
    #[test]
    fn metrics_are_deterministic() {
        assert_eq!(report(), report());
        for fx in FIXTURES {
            let (a, b) = (pipeline(fx), pipeline(fx));
            assert_eq!(a.root, b.root, "{}: tree", fx.file);
            assert_eq!(a.stats, b.stats, "{}: stats", fx.file);
            assert_eq!(a.stages, b.stages, "{}: stages", fx.file);
        }
    }
}
