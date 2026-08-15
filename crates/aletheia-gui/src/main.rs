//! Aletheia GUI — thin egui client over [`aletheia_mcp`] protocol dispatch.
//!
//! Engine owns all truth. This binary only renders protocol responses and
//! sends assertions (`rename`) / queries (`listing`, `decompile`, `why`, …).

#![allow(clippy::collapsible_if)]

mod app;
mod client;
mod theme;

use app::AletheiaApp;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([960.0, 640.0])
            .with_title("Aletheia"),
        ..Default::default()
    };
    eframe::run_native(
        "Aletheia",
        options,
        Box::new(|cc| Ok(Box::new(AletheiaApp::new(cc)))),
    )
}
