//! Design tokens from `DESIGN_GUI.md` / `design/aletheia-gui-mockup.html`.
//! Dark-first; trust + diff axes kept distinct from syntax.

use egui::{Color32, Stroke, Style, Visuals};

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct Tokens {
    pub ground: Color32,
    pub panel: Color32,
    pub panel2: Color32,
    pub panel3: Color32,
    pub line: Color32,
    pub line_strong: Color32,
    pub ink: Color32,
    pub ink2: Color32,
    pub ink3: Color32,
    pub accent: Color32,
    pub ember: Color32,
    pub proven: Color32,
    pub heuristic: Color32,
    pub asserted: Color32,
    pub sx_addr: Color32,
    pub sx_reg: Color32,
    pub sx_imm: Color32,
    pub sx_str: Color32,
    pub sx_cmt: Color32,
    pub d_unchanged: Color32,
    pub d_moved: Color32,
    pub d_modified: Color32,
    pub d_uncertain: Color32,
    pub d_added: Color32,
    pub d_removed: Color32,
}

impl Tokens {
    pub fn dark() -> Self {
        Self {
            ground: Color32::from_rgb(0x0b, 0x0e, 0x14),
            panel: Color32::from_rgb(0x10, 0x14, 0x1d),
            panel2: Color32::from_rgb(0x16, 0x1c, 0x27),
            panel3: Color32::from_rgb(0x1c, 0x24, 0x31),
            line: Color32::from_rgb(0x23, 0x2b, 0x38),
            line_strong: Color32::from_rgb(0x32, 0x3c, 0x4c),
            ink: Color32::from_rgb(0xcd, 0xd5, 0xe0),
            ink2: Color32::from_rgb(0x8b, 0x96, 0xa6),
            ink3: Color32::from_rgb(0x5c, 0x66, 0x75),
            accent: Color32::from_rgb(0x35, 0xc4, 0xb5),
            ember: Color32::from_rgb(0xf0, 0x70, 0x3a),
            proven: Color32::from_rgb(0x2f, 0xbf, 0xae),
            heuristic: Color32::from_rgb(0xe0, 0xa4, 0x4a),
            asserted: Color32::from_rgb(0x9a, 0x8c, 0xf0),
            sx_addr: Color32::from_rgb(0x6b, 0x76, 0x86),
            sx_reg: Color32::from_rgb(0x7f, 0xb3, 0xd5),
            sx_imm: Color32::from_rgb(0xcf, 0xa0, 0x6a),
            sx_str: Color32::from_rgb(0x8f, 0xbf, 0x7f),
            sx_cmt: Color32::from_rgb(0x5c, 0x66, 0x75),
            d_unchanged: Color32::from_rgb(0x5c, 0x66, 0x75),
            d_moved: Color32::from_rgb(0x4c, 0x8d, 0xff),
            d_modified: Color32::from_rgb(0xe0, 0x80, 0x3a),
            d_uncertain: Color32::from_rgb(0xe0, 0xa4, 0x4a),
            d_added: Color32::from_rgb(0x57, 0xc6, 0x6b),
            d_removed: Color32::from_rgb(0xef, 0x5f, 0x6b),
        }
    }

    pub fn apply(self, ctx: &egui::Context) {
        let mut style = Style::default();
        style.spacing.item_spacing = egui::vec2(6.0, 4.0);
        style.spacing.window_margin = egui::Margin::same(0);
        style.spacing.button_padding = egui::vec2(8.0, 4.0);
        style.visuals = Visuals {
            dark_mode: true,
            override_text_color: Some(self.ink),
            panel_fill: self.panel,
            window_fill: self.panel,
            faint_bg_color: self.panel2,
            extreme_bg_color: self.ground,
            widgets: egui::style::Widgets {
                noninteractive: widget(self.panel2, self.ink2, self.line),
                inactive: widget(self.panel2, self.ink2, self.line),
                hovered: widget(self.panel3, self.ink, self.line_strong),
                active: widget(self.panel3, self.ink, self.accent),
                open: widget(self.panel3, self.ink, self.accent),
            },
            selection: egui::style::Selection {
                bg_fill: Color32::from_rgba_unmultiplied(0x2f, 0xbf, 0xae, 40),
                stroke: Stroke::new(1.0_f32, self.accent),
            },
            hyperlink_color: self.accent,
            window_stroke: Stroke::new(1.0_f32, self.line),
            ..Visuals::dark()
        };
        ctx.set_style(style);
    }
}

fn widget(bg: Color32, fg: Color32, stroke: Color32) -> egui::style::WidgetVisuals {
    egui::style::WidgetVisuals {
        bg_fill: bg,
        weak_bg_fill: bg,
        bg_stroke: Stroke::new(1.0_f32, stroke),
        fg_stroke: Stroke::new(1.0_f32, fg),
        corner_radius: egui::CornerRadius::same(4),
        expansion: 0.0,
    }
}

/// Map protocol `source` label → trust channel.
pub fn trust_for_source(source: &str) -> Trust {
    match source {
        "prologue" | "cfg" | "unknown" => Trust::Heuristic,
        "asserted" => Trust::Asserted,
        _ => Trust::Proven,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Trust {
    Proven,
    Heuristic,
    Asserted,
}

impl Trust {
    pub fn color(self, t: &Tokens) -> Color32 {
        match self {
            Trust::Proven => t.proven,
            Trust::Heuristic => t.heuristic,
            Trust::Asserted => t.asserted,
        }
    }

    pub fn glyph(self) -> &'static str {
        match self {
            Trust::Proven => "",
            Trust::Heuristic => " ?",
            Trust::Asserted => " ✎",
        }
    }
}
