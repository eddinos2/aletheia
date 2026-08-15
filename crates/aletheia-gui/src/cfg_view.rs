//! Minimal layered CFG layout over engine-owned blocks/edges.
//! Frontend may position nodes; it never invents edges.

use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2};

use crate::client::{CfgBlock, CfgInfo};
use crate::theme::Tokens;

const NODE_W: f32 = 148.0;
const NODE_H: f32 = 52.0;
const H_GAP: f32 = 36.0;
const V_GAP: f32 = 56.0;

#[derive(Clone)]
struct Laid {
    start: u64,
    end: u64,
    terminator: String,
    pos: Pos2,
}

/// Paint a readable layered CFG. Returns a clicked block start VA, if any.
pub fn show(ui: &mut egui::Ui, t: Tokens, cfg: &CfgInfo, focus: Option<u64>) -> Option<u64> {
    if cfg.blocks.is_empty() {
        ui.label(
            egui::RichText::new("No blocks from engine `cfg` for this entry.")
                .size(13.0)
                .color(t.ink3),
        );
        return None;
    }

    let laid = layout(cfg);
    let width = laid
        .iter()
        .map(|n| n.pos.x + NODE_W)
        .fold(0.0_f32, f32::max)
        + 24.0;
    let height = laid
        .iter()
        .map(|n| n.pos.y + NODE_H)
        .fold(0.0_f32, f32::max)
        + 24.0;

    let (resp, painter) = ui.allocate_painter(Vec2::new(width.max(320.0), height.max(160.0)), Sense::click());
    let origin = resp.rect.min + Vec2::new(12.0, 12.0);

    // Edges first (under nodes).
    for e in &cfg.edges {
        let Some(a) = laid.iter().find(|n| n.start == e.from) else {
            continue;
        };
        let Some(b) = laid.iter().find(|n| n.start == e.to) else {
            continue;
        };
        let p0 = origin + a.pos.to_vec2() + Vec2::new(NODE_W * 0.5, NODE_H);
        let p1 = origin + b.pos.to_vec2() + Vec2::new(NODE_W * 0.5, 0.0);
        let mid = Pos2::new(p0.x, (p0.y + p1.y) * 0.5);
        painter.line_segment([p0, mid], Stroke::new(1.5_f32, t.line_strong));
        painter.line_segment([mid, p1], Stroke::new(1.5_f32, t.line_strong));
        // Arrow head
        let tip = p1;
        painter.line_segment(
            [tip, tip + Vec2::new(-5.0, -7.0)],
            Stroke::new(1.5_f32, t.line_strong),
        );
        painter.line_segment(
            [tip, tip + Vec2::new(5.0, -7.0)],
            Stroke::new(1.5_f32, t.line_strong),
        );
    }

    let mut clicked = None;
    for n in &laid {
        let rect = Rect::from_min_size(origin + n.pos.to_vec2(), Vec2::new(NODE_W, NODE_H));
        let is_entry = n.start == cfg.entry;
        let is_focus = focus == Some(n.start);
        let fill = if is_focus {
            Color32::from_rgba_unmultiplied(0xe0, 0xa4, 0x4a, 40)
        } else if is_entry {
            Color32::from_rgba_unmultiplied(0x2f, 0xbf, 0xae, 28)
        } else {
            t.panel
        };
        painter.rect(
            rect,
            6.0,
            fill,
            Stroke::new(1.0_f32, if is_entry { t.proven } else { t.line }),
            egui::StrokeKind::Inside,
        );
        painter.text(
            rect.min + Vec2::new(10.0, 8.0),
            egui::Align2::LEFT_TOP,
            format!("0x{:x}", n.start),
            egui::FontId::monospace(12.0),
            t.ink,
        );
        painter.text(
            rect.min + Vec2::new(10.0, 26.0),
            egui::Align2::LEFT_TOP,
            format!("{} · →0x{:x}", n.terminator, n.end),
            egui::FontId::monospace(10.0),
            t.ink3,
        );
        if resp.clicked() && rect.contains(resp.interact_pointer_pos().unwrap_or(Pos2::ZERO)) {
            clicked = Some(n.start);
        }
    }

    ui.add_space(6.0);
    ui.label(
        egui::RichText::new(format!(
            "{} blocks · {} edges (engine successors only) · click block → listing",
            cfg.blocks.len(),
            cfg.edges.len()
        ))
        .size(11.0)
        .color(t.ink3),
    );
    clicked
}

fn layout(cfg: &CfgInfo) -> Vec<Laid> {
    // BFS layers from entry (Sugiyama-lite).
    use std::collections::{BTreeMap, BTreeSet, VecDeque};

    let succ: BTreeMap<u64, Vec<u64>> = cfg
        .blocks
        .iter()
        .map(|b| (b.start, b.successors.clone()))
        .collect();
    let mut layer: BTreeMap<u64, usize> = BTreeMap::new();
    let mut q = VecDeque::new();
    q.push_back(cfg.entry);
    layer.insert(cfg.entry, 0);
    while let Some(va) = q.pop_front() {
        let layer_n = layer[&va];
        if let Some(ss) = succ.get(&va) {
            for &s in ss {
                if !layer.contains_key(&s) && succ.contains_key(&s) {
                    layer.insert(s, layer_n + 1);
                    q.push_back(s);
                }
            }
        }
    }
    // Unreached blocks (e.g. after incomplete walk) get trailing layers.
    let mut next_orphan = layer.values().copied().max().unwrap_or(0) + 1;
    for b in &cfg.blocks {
        if let std::collections::btree_map::Entry::Vacant(e) = layer.entry(b.start) {
            e.insert(next_orphan);
            next_orphan += 1;
        }
    }

    let mut by_layer: BTreeMap<usize, Vec<&CfgBlock>> = BTreeMap::new();
    for b in &cfg.blocks {
        by_layer
            .entry(layer[&b.start])
            .or_default()
            .push(b);
    }
    for v in by_layer.values_mut() {
        v.sort_by_key(|b| b.start);
    }

    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for (layer_idx, blocks) in by_layer {
        let n = blocks.len() as f32;
        let row_w = n * NODE_W + (n - 1.0).max(0.0) * H_GAP;
        let start_x = (-row_w * 0.5).max(0.0);
        for (i, b) in blocks.iter().enumerate() {
            if !seen.insert(b.start) {
                continue;
            }
            out.push(Laid {
                start: b.start,
                end: b.end,
                terminator: b.terminator.clone(),
                pos: Pos2::new(
                    start_x + i as f32 * (NODE_W + H_GAP),
                    layer_idx as f32 * (NODE_H + V_GAP),
                ),
            });
        }
    }
    // Shift so min x >= 0
    let min_x = out.iter().map(|n| n.pos.x).fold(0.0_f32, f32::min);
    if min_x < 0.0 {
        for n in &mut out {
            n.pos.x -= min_x;
        }
    }
    out
}
