//! Three-region workstation shell (DESIGN_GUI §3) over the protocol client.

use egui::Color32;

use crate::client::{self, Client, DiffInfo, FuncRow, OpenInfo, WhyInfo, XrefsInfo};
use crate::theme::{trust_for_source, Tokens, Trust};

#[derive(Clone, Copy, PartialEq, Eq)]
enum CenterMode {
    Listing,
    Decompile,
    Diff,
    Patch,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Modal {
    None,
    Rename,
    GoTo,
    Palette,
}

pub struct AletheiaApp {
    tokens: Tokens,
    client: Client,
    health_ver: String,

    session: Option<OpenInfo>,
    session_b: Option<OpenInfo>,
    functions: Vec<FuncRow>,
    selected: Option<usize>,
    filter: String,

    center: CenterMode,
    listing: String,
    decompile: String,
    why: WhyInfo,
    xrefs: XrefsInfo,
    diff: DiffInfo,
    patch_report: String,

    modal: Modal,
    rename_buf: String,
    goto_buf: String,
    palette_buf: String,
    status: String,
    busy: bool,
}

impl AletheiaApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let tokens = Tokens::dark();
        tokens.apply(&cc.egui_ctx);
        let client = Client::new();
        let health_ver = client
            .health()
            .map(|h| h.engine_version)
            .unwrap_or_else(|_| "?".into());
        Self {
            tokens,
            client,
            health_ver,
            session: None,
            session_b: None,
            functions: Vec::new(),
            selected: None,
            filter: String::new(),
            center: CenterMode::Listing,
            listing: String::new(),
            decompile: String::new(),
            why: WhyInfo::default(),
            xrefs: XrefsInfo::default(),
            diff: DiffInfo::default(),
            patch_report: String::new(),
            modal: Modal::None,
            rename_buf: String::new(),
            goto_buf: String::new(),
            palette_buf: String::new(),
            status: "Open a binary (⌘O) · Gate G1 surface".into(),
            busy: false,
        }
    }

    fn open_path(&mut self, path: String) {
        self.busy = true;
        match self.client.open(&path) {
            Ok(info) => {
                if info.encrypted {
                    self.status = format!("⚠ encrypted image — {}", info.path);
                } else {
                    self.status = format!("opened {} · {}", info.arch, info.hash);
                }
                let sid = info.session_id.clone();
                self.session = Some(info);
                match self.client.functions(&sid, 8192) {
                    Ok(funcs) => {
                        self.functions = funcs;
                        self.selected = if self.functions.is_empty() {
                            None
                        } else {
                            Some(0)
                        };
                        if let Some(0) = self.selected {
                            self.load_selection();
                        }
                    }
                    Err(e) => self.status = format!("functions: {e}"),
                }
            }
            Err(e) => self.status = format!("open failed: {e}"),
        }
        self.busy = false;
    }

    fn open_diff_path(&mut self, path: String) {
        let Some(a) = self.session.as_ref().map(|s| s.session_id.clone()) else {
            self.status = "open a primary binary before diff".into();
            return;
        };
        match self.client.open(&path) {
            Ok(info) => {
                let b = info.session_id.clone();
                self.session_b = Some(info);
                match self.client.diff(&a, &b) {
                    Ok(d) => {
                        self.diff = d;
                        self.center = CenterMode::Diff;
                        self.status = format!(
                            "diff · +{} −{} ~{} ?{}",
                            self.diff.added, self.diff.removed, self.diff.modified, self.diff.uncertain
                        );
                    }
                    Err(e) => self.status = format!("diff: {e}"),
                }
            }
            Err(e) => self.status = format!("open B failed: {e}"),
        }
    }

    fn selected_func(&self) -> Option<&FuncRow> {
        self.selected.and_then(|i| self.functions.get(i))
    }

    fn load_selection(&mut self) {
        let Some(sess) = self.session.as_ref().map(|s| s.session_id.clone()) else {
            return;
        };
        let Some(func) = self.selected_func().cloned() else {
            return;
        };
        match self.client.listing(&sess, func.va) {
            Ok(t) => self.listing = t,
            Err(e) => self.listing = format!("// listing error: {e}"),
        }
        match self.client.decompile(&sess, func.va) {
            Ok(t) => self.decompile = t,
            Err(e) => self.decompile = format!("// decompile error: {e}"),
        }
        match self.client.why(&sess, func.va) {
            Ok(w) => self.why = w,
            Err(e) => {
                self.why = WhyInfo {
                    text: e,
                    ..WhyInfo::default()
                }
            }
        }
        match self.client.xrefs(&sess, func.va) {
            Ok(x) => self.xrefs = x,
            Err(_) => self.xrefs = XrefsInfo::default(),
        }
        if self.center == CenterMode::Patch {
            self.patch_report = self
                .client
                .patch_preview(&sess, func.va)
                .unwrap_or_else(|e| e);
        }
    }

    fn apply_rename(&mut self) {
        let name = self.rename_buf.trim().to_string();
        if name.is_empty() {
            return;
        }
        let Some(sess) = self.session.as_ref().map(|s| s.session_id.clone()) else {
            return;
        };
        let Some(func) = self.selected_func().cloned() else {
            return;
        };
        match self.client.rename(&sess, func.va, &name) {
            Ok(()) => {
                if let Some(i) = self.selected {
                    if let Some(f) = self.functions.get_mut(i) {
                        f.name = Some(name.clone());
                        f.source = "asserted".into();
                    }
                }
                self.status = format!("renamed → {name}");
                self.modal = Modal::None;
                self.load_selection();
            }
            Err(e) => self.status = format!("rename: {e}"),
        }
    }

    fn goto_address(&mut self) {
        let raw = self.goto_buf.trim().to_string();
        let Some(va) = client::parse_hex(&raw).or_else(|| {
            // name match
            self.functions
                .iter()
                .find(|f| f.name.as_deref() == Some(raw.as_str()))
                .map(|f| f.va)
        }) else {
            self.status = format!("go-to: cannot resolve `{raw}`");
            return;
        };
        if let Some(i) = self.functions.iter().position(|f| f.va == va) {
            self.selected = Some(i);
            self.load_selection();
            self.modal = Modal::None;
            self.status = format!("→ 0x{va:x}");
        } else {
            self.status = format!("no function at 0x{va:x}");
        }
    }

    fn filtered_indices(&self) -> Vec<usize> {
        let q = self.filter.to_ascii_lowercase();
        self.functions
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                if q.is_empty() {
                    return true;
                }
                let name = f.name.as_deref().unwrap_or("");
                name.to_ascii_lowercase().contains(&q)
                    || format!("{:x}", f.va).contains(&q)
                    || f.source.contains(&q)
            })
            .map(|(i, _)| i)
            .collect()
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        let cmd = ctx.input(|i| i.modifiers.command || i.modifiers.mac_cmd);
        if cmd && ctx.input(|i| i.key_pressed(egui::Key::O)) {
            self.pick_open(false);
        }
        if cmd && ctx.input(|i| i.key_pressed(egui::Key::D)) {
            self.pick_open(true);
        }
        if cmd && ctx.input(|i| i.key_pressed(egui::Key::K)) {
            self.modal = Modal::Palette;
            self.palette_buf.clear();
        }
        if self.modal != Modal::None {
            if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.modal = Modal::None;
            }
            return;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::N)) {
            if let Some(f) = self.selected_func() {
                self.rename_buf = f.name.clone().unwrap_or_default();
                self.modal = Modal::Rename;
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::G)) {
            self.goto_buf.clear();
            self.modal = Modal::GoTo;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Y)) {
            self.center = CenterMode::Decompile;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::U)) {
            self.center = CenterMode::Listing;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Questionmark))
            || (ctx.input(|i| i.modifiers.shift) && ctx.input(|i| i.key_pressed(egui::Key::Slash)))
        {
            // refresh why pin
            if let (Some(sess), Some(f)) = (
                self.session.as_ref().map(|s| s.session_id.clone()),
                self.selected_func().cloned(),
            ) {
                if let Ok(w) = self.client.why(&sess, f.va) {
                    self.why = w;
                    self.status = "provenance pinned".into();
                }
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::P)) {
            self.center = CenterMode::Patch;
            if let (Some(sess), Some(f)) = (
                self.session.as_ref().map(|s| s.session_id.clone()),
                self.selected_func().cloned(),
            ) {
                self.patch_report = self
                    .client
                    .patch_preview(&sess, f.va)
                    .unwrap_or_else(|e| e);
            }
        }
    }

    fn pick_open(&mut self, as_diff: bool) {
        if let Some(path) = rfd::FileDialog::new().pick_file() {
            let path = path.display().to_string();
            if as_diff {
                self.open_diff_path(path);
            } else {
                self.open_path(path);
            }
        }
    }
}

impl eframe::App for AletheiaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.tokens.apply(ctx);
        self.handle_keys(ctx);
        let t = self.tokens;

        // ——— Top bar ———
        egui::TopBottomPanel::top("top").exact_height(40.0).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.add_space(10.0);
                // brand ember spark
                let (rect, _) = ui.allocate_exact_size(egui::vec2(11.0, 11.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 5.5, t.ember);
                ui.label(
                    egui::RichText::new("Aletheia")
                        .strong()
                        .size(14.0)
                        .color(t.ink),
                );
                ui.separator();
                if let Some(s) = &self.session {
                    ui.label(
                        egui::RichText::new(short_path(&s.path))
                            .monospace()
                            .size(12.0)
                            .color(t.ink2),
                    );
                    ui.label(
                        egui::RichText::new(format!("· {}", s.arch))
                            .monospace()
                            .size(11.0)
                            .color(t.ink3),
                    );
                    if s.encrypted {
                        ui.colored_label(t.heuristic, "encrypted");
                    }
                } else {
                    ui.label(egui::RichText::new("no binary").color(t.ink3));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(10.0);
                    let stamp = match &self.session {
                        Some(s) => format!("{} · eng {}", s.hash, self.health_ver),
                        None => format!("eng {}", self.health_ver),
                    };
                    egui::Frame::NONE
                        .fill(t.panel)
                        .stroke(egui::Stroke::new(1.0_f32, t.line))
                        .corner_radius(6.0)
                        .inner_margin(egui::Margin::symmetric(9, 4))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let (r, _) =
                                    ui.allocate_exact_size(egui::vec2(7.0, 7.0), egui::Sense::hover());
                                ui.painter().circle_filled(r.center(), 3.5, t.proven);
                                ui.label(
                                    egui::RichText::new(stamp)
                                        .monospace()
                                        .size(11.0)
                                        .color(t.ink2),
                                );
                            });
                        });
                    if ui
                        .add(egui::Button::new(egui::RichText::new("Diff B…").size(12.0)))
                        .on_hover_text("⌘D — open second binary for diff")
                        .clicked()
                    {
                        self.pick_open(true);
                    }
                    if ui
                        .add(egui::Button::new(egui::RichText::new("Open…").size(12.0)))
                        .on_hover_text("⌘O")
                        .clicked()
                    {
                        self.pick_open(false);
                    }
                });
            });
        });

        // ——— Sub bar (mode + keys hint) ———
        egui::TopBottomPanel::top("sub").exact_height(34.0).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.add_space(10.0);
                mode_btn(ui, &mut self.center, CenterMode::Listing, "Listing", "u");
                mode_btn(ui, &mut self.center, CenterMode::Decompile, "Decompile", "y");
                mode_btn(ui, &mut self.center, CenterMode::Diff, "Diff", "⌘D");
                mode_btn(ui, &mut self.center, CenterMode::Patch, "Patch", "p");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(10.0);
                    ui.label(
                        egui::RichText::new("⌘K palette · g go · n rename · ? why · y/u toggle")
                            .size(11.0)
                            .color(t.ink3),
                    );
                });
            });
        });

        // ——— Status ———
        egui::TopBottomPanel::bottom("status").exact_height(28.0).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.add_space(12.0);
                ui.label(egui::RichText::new(&self.status).size(11.0).color(t.ink2));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(12.0);
                    legend_item(ui, t, Trust::Asserted, "asserted");
                    legend_item(ui, t, Trust::Heuristic, "heuristic");
                    legend_item(ui, t, Trust::Proven, "proven");
                });
            });
        });

        // ——— Left navigator ———
        egui::SidePanel::left("nav")
            .exact_width(260.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.set_min_width(200.0);
                egui::Frame::NONE
                    .fill(t.panel2)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("FUNCTIONS")
                                    .size(10.0)
                                    .color(t.ink3)
                                    .strong(),
                            );
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(
                                    egui::RichText::new(format!("{}", self.functions.len()))
                                        .monospace()
                                        .size(11.0)
                                        .color(t.ink3),
                                );
                            });
                        });
                        ui.add(
                            egui::TextEdit::singleline(&mut self.filter)
                                .hint_text("filter…")
                                .desired_width(f32::INFINITY),
                        );
                        ui.add_space(4.0);
                        let indices = self.filtered_indices();
                        egui::ScrollArea::vertical()
                            .id_salt("nav_scroll")
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                for i in indices {
                                    let (va, source, label, trust, selected) = {
                                        let f = &self.functions[i];
                                        let trust = trust_for_source(&f.source);
                                        let label = f
                                            .name
                                            .clone()
                                            .unwrap_or_else(|| format!("sub_{:x}", f.va));
                                        let selected = self.selected == Some(i);
                                        (f.va, f.source.clone(), label, trust, selected)
                                    };
                                    let mut row = egui::RichText::new(format!(
                                        "{}{}{}",
                                        trust_dot_char(trust),
                                        label,
                                        trust.glyph()
                                    ))
                                    .monospace()
                                    .size(12.0)
                                    .color(if selected { t.ink } else { t.ink2 });
                                    if selected {
                                        row = row.strong();
                                    }
                                    let resp = ui.selectable_label(selected, row);
                                    if resp.clicked() {
                                        self.selected = Some(i);
                                        self.load_selection();
                                    }
                                    ui.label(
                                        egui::RichText::new(format!("  0x{va:x} · {source}"))
                                            .monospace()
                                            .size(10.0)
                                            .color(t.ink3),
                                    );
                                }
                            });
                    });
            });

        // ——— Right context ———
        egui::SidePanel::right("ctx")
            .exact_width(300.0)
            .resizable(true)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    section_header(ui, t, "PROVENANCE");
                    let trust = match self.why.trust.as_str() {
                        "proven" => Trust::Proven,
                        "asserted" => Trust::Asserted,
                        _ => Trust::Heuristic,
                    };
                    ui.horizontal(|ui| {
                        trust_badge(ui, t, trust);
                        ui.label(
                            egui::RichText::new(&self.why.source)
                                .monospace()
                                .size(11.0)
                                .color(t.ink2),
                        );
                    });
                    ui.add_space(4.0);
                    egui::Frame::NONE
                        .stroke(egui::Stroke::new(1.0_f32, t.line))
                        .corner_radius(6.0)
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(if self.why.text.is_empty() {
                                        "Select a function · press ?"
                                    } else {
                                        &self.why.text
                                    })
                                    .monospace()
                                    .size(11.0)
                                    .color(t.ink2),
                                )
                                .wrap(),
                            );
                        });

                    ui.add_space(10.0);
                    section_header(ui, t, "XREFS");
                    ui.label(
                        egui::RichText::new(format!("total {}", self.xrefs.total))
                            .size(10.0)
                            .color(t.ink3),
                    );
                    ui.label(
                        egui::RichText::new(format!("from {}", compact_json(&self.xrefs.from)))
                            .monospace()
                            .size(10.5)
                            .color(t.sx_reg),
                    );
                    ui.label(
                        egui::RichText::new(format!("to   {}", compact_json(&self.xrefs.to)))
                            .monospace()
                            .size(10.5)
                            .color(t.sx_reg),
                    );

                    ui.add_space(10.0);
                    section_header(ui, t, "ANNOTATIONS");
                    if let Some(f) = self.selected_func() {
                        ui.label(
                            egui::RichText::new(format!(
                                "name  {}",
                                f.name.as_deref().unwrap_or("—")
                            ))
                            .monospace()
                            .size(11.5)
                            .color(t.asserted),
                        );
                        ui.label(
                            egui::RichText::new("n rename · asserted → annotate::Db")
                                .size(10.0)
                                .color(t.ink3),
                        );
                    }
                });
            });

        // ——— Center ———
        egui::CentralPanel::default().show(ctx, |ui| {
            let body = match self.center {
                CenterMode::Listing => self.listing.as_str(),
                CenterMode::Decompile => self.decompile.as_str(),
                CenterMode::Diff => self.diff.report.as_str(),
                CenterMode::Patch => self.patch_report.as_str(),
            };
            if self.center == CenterMode::Diff {
                ui.horizontal(|ui| {
                    bucket(ui, t, "unchanged", self.diff.unchanged, t.d_unchanged);
                    bucket(ui, t, "moved", self.diff.moved, t.d_moved);
                    bucket(ui, t, "modified", self.diff.modified, t.d_modified);
                    bucket(ui, t, "uncertain", self.diff.uncertain, t.d_uncertain);
                    bucket(ui, t, "added", self.diff.added, t.d_added);
                    bucket(ui, t, "removed", self.diff.removed, t.d_removed);
                });
                ui.add_space(6.0);
                if !self.diff.hunks_blob.is_empty() {
                    ui.label(
                        egui::RichText::new(format!("patch hunks: {}", compact_json(&self.diff.hunks_blob)))
                            .monospace()
                            .size(11.0)
                            .color(t.ink3),
                    );
                }
            }
            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if body.is_empty() {
                        ui.centered_and_justified(|ui| {
                            ui.label(
                                egui::RichText::new(
                                    "Open a fixture (⌘O) → select a function → Listing / Decompile",
                                )
                                .size(14.0)
                                .color(t.ink3),
                            );
                        });
                    } else {
                        // Monospace listing / pseudo — dense RE surface
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(body)
                                    .monospace()
                                    .size(12.5)
                                    .color(t.ink)
                                    .line_height(Some(20.0)),
                            )
                            .selectable(true),
                        );
                    }
                });
        });

        // ——— Modals ———
        match self.modal {
            Modal::Rename => self.show_modal(ctx, "Rename", |s, ui| {
                ui.label("Asserted name → annotate::Db (protocol rename)");
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut s.rename_buf)
                        .font(egui::TextStyle::Monospace)
                        .desired_width(320.0),
                );
                resp.request_focus();
                if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    s.apply_rename();
                }
            }),
            Modal::GoTo => self.show_modal(ctx, "Go to address / symbol", |s, ui| {
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut s.goto_buf)
                        .hint_text("0x1000 or symbol")
                        .font(egui::TextStyle::Monospace)
                        .desired_width(320.0),
                );
                resp.request_focus();
                if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    s.goto_address();
                }
            }),
            Modal::Palette => self.show_modal(ctx, "Command palette", |s, ui| {
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut s.palette_buf)
                        .hint_text("open · diff · listing · decompile · rename · why · patch")
                        .desired_width(400.0),
                );
                resp.request_focus();
                if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    let cmd = s.palette_buf.trim().to_ascii_lowercase();
                    s.modal = Modal::None;
                    match cmd.as_str() {
                        "open" => s.pick_open(false),
                        "diff" => s.pick_open(true),
                        "listing" | "u" => s.center = CenterMode::Listing,
                        "decompile" | "y" => s.center = CenterMode::Decompile,
                        "rename" | "n" => {
                            if let Some(f) = s.selected_func() {
                                s.rename_buf = f.name.clone().unwrap_or_default();
                                s.modal = Modal::Rename;
                            }
                        }
                        "why" | "?" => {
                            if let (Some(sess), Some(f)) = (
                                s.session.as_ref().map(|x| x.session_id.clone()),
                                s.selected_func().cloned(),
                            ) {
                                if let Ok(w) = s.client.why(&sess, f.va) {
                                    s.why = w;
                                }
                            }
                        }
                        "patch" | "p" => {
                            s.center = CenterMode::Patch;
                            if let (Some(sess), Some(f)) = (
                                s.session.as_ref().map(|x| x.session_id.clone()),
                                s.selected_func().cloned(),
                            ) {
                                s.patch_report = s
                                    .client
                                    .patch_preview(&sess, f.va)
                                    .unwrap_or_else(|e| e);
                            }
                        }
                        other => s.status = format!("unknown command: {other}"),
                    }
                }
            }),
            Modal::None => {}
        }
    }
}

impl AletheiaApp {
    fn show_modal(&mut self, ctx: &egui::Context, title: &str, body: impl FnOnce(&mut Self, &mut egui::Ui)) {
        let t = self.tokens;
        let mut open = true;
        egui::Window::new(title)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .frame(
                egui::Frame::NONE
                    .fill(t.panel)
                    .stroke(egui::Stroke::new(1.0_f32, t.line_strong))
                    .corner_radius(10.0)
                    .inner_margin(16.0)
                    .shadow(egui::Shadow {
                        offset: [0, 8],
                        blur: 24,
                        spread: 0,
                        color: Color32::from_black_alpha(180),
                    }),
            )
            .open(&mut open)
            .show(ctx, |ui| {
                body(self, ui);
                ui.add_space(8.0);
                ui.label(egui::RichText::new("Enter · Esc").size(10.0).color(t.ink3));
            });
        if !open {
            self.modal = Modal::None;
        }
    }
}

fn mode_btn(
    ui: &mut egui::Ui,
    cur: &mut CenterMode,
    mode: CenterMode,
    label: &str,
    hint: &str,
) {
    let selected = *cur == mode;
    let text = if selected {
        egui::RichText::new(label).strong().size(12.0)
    } else {
        egui::RichText::new(label).size(12.0)
    };
    if ui
        .add(egui::SelectableLabel::new(selected, text))
        .on_hover_text(hint)
        .clicked()
    {
        *cur = mode;
    }
}

fn section_header(ui: &mut egui::Ui, t: Tokens, title: &str) {
    ui.label(
        egui::RichText::new(title)
            .size(10.0)
            .strong()
            .color(t.ink3),
    );
    ui.add_space(4.0);
}

fn trust_badge(ui: &mut egui::Ui, t: Tokens, trust: Trust) {
    let label = match trust {
        Trust::Proven => "proven",
        Trust::Heuristic => "heuristic ?",
        Trust::Asserted => "asserted ✎",
    };
    let bg = match trust {
        Trust::Proven => Color32::from_rgba_unmultiplied(0x2f, 0xbf, 0xae, 33),
        Trust::Heuristic => Color32::from_rgba_unmultiplied(0xe0, 0xa4, 0x4a, 36),
        Trust::Asserted => Color32::from_rgba_unmultiplied(0x9a, 0x8c, 0xf0, 36),
    };
    egui::Frame::NONE
        .fill(bg)
        .corner_radius(10.0)
        .inner_margin(egui::Margin::symmetric(8, 3))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(label)
                    .size(11.0)
                    .strong()
                    .color(trust.color(&t)),
            );
        });
}

fn legend_item(ui: &mut egui::Ui, t: Tokens, trust: Trust, label: &str) {
    ui.horizontal(|ui| {
        let (r, _) = ui.allocate_exact_size(egui::vec2(9.0, 9.0), egui::Sense::hover());
        match trust {
            Trust::Proven => {
                ui.painter().circle_filled(r.center(), 4.0, t.proven);
            }
            Trust::Heuristic => {
                ui.painter()
                    .circle_stroke(r.center(), 4.0, egui::Stroke::new(1.5_f32, t.heuristic));
            }
            Trust::Asserted => {
                ui.painter()
                    .circle_stroke(r.center(), 4.0, egui::Stroke::new(1.5_f32, t.asserted));
            }
        }
        ui.label(egui::RichText::new(label).size(11.0).color(t.ink2));
    });
}

fn trust_dot_char(trust: Trust) -> &'static str {
    match trust {
        Trust::Proven => "● ",
        Trust::Heuristic => "○ ",
        Trust::Asserted => "◉ ",
    }
}

fn bucket(ui: &mut egui::Ui, t: Tokens, label: &str, n: u64, color: Color32) {
    egui::Frame::NONE
        .fill(t.panel)
        .stroke(egui::Stroke::new(1.0_f32, t.line))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(
                    egui::RichText::new(format!("{n}"))
                        .monospace()
                        .size(18.0)
                        .strong()
                        .color(t.ink),
                );
                ui.label(egui::RichText::new(label).size(10.0).color(color));
            });
        });
}

fn short_path(p: &str) -> String {
    let p = p.replace('\\', "/");
    if let Some(i) = p.rfind('/') {
        p[i + 1..].to_string()
    } else {
        p
    }
}

fn compact_json(s: &str) -> String {
    if s.len() <= 180 {
        s.to_string()
    } else {
        format!("{}…", &s[..180])
    }
}
