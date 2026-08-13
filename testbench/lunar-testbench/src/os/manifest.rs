use dioxus::prelude::*;

use crate::os::log_window::LogWindow;
use crate::pages::{
    backend_api::BackendApi, dashboard::Dashboard, datasets::Datasets, models::Models,
    pipeline::Pipeline, sandbox::Sandbox, siren_gallery::SirenGallery, training::Training,
    validation::Validation,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AppCategory {
    Backend,
    TestbenchBackend,
    Frontend,
}

impl AppCategory {
    pub const ALL: [Self; 3] = [Self::Backend, Self::TestbenchBackend, Self::Frontend];
    pub fn service(self) -> &'static str {
        match self {
            Self::Backend => "backend",
            Self::TestbenchBackend => "testbench-backend",
            Self::Frontend => "frontend",
        }
    }
    pub fn title(self) -> &'static str {
        match self {
            Self::Backend => "Backend",
            Self::TestbenchBackend => "Testbench Backend",
            Self::Frontend => "Frontend",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowContentMode {
    /// Document-like page with internal vertical scroll.
    Scroll,
    /// iframe/canvas occupies all available area, scroll forbidden.
    Fill,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowSizeSpec {
    pub preferred_width: f64,
    pub preferred_height: f64,
    pub min_width: f64,
    pub min_height: f64,
}

impl WindowSizeSpec {
    /// Fit an initial window rect into the usable client area. When the client
    /// area is smaller than the app minimum, the viewport wins so the window
    /// stays reachable instead of overflowing beyond the screen.
    pub fn fit_to_available_space(self, available_width: f64, available_height: f64) -> (f64, f64) {
        let available_width = available_width.max(0.0);
        let available_height = available_height.max(0.0);
        let width = self
            .preferred_width
            .min(available_width)
            .max(self.min_width.min(available_width));
        let height = self
            .preferred_height
            .min(available_height)
            .max(self.min_height.min(available_height));
        (width, height)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AppDef {
    pub id: &'static str,
    pub title: &'static str,
    pub category: AppCategory,
    pub desktop_order: u8,
    pub required_services: &'static [&'static str],
    pub content_mode: WindowContentMode,
    pub size: WindowSizeSpec,
}

const BACKEND: &[&str] = &["backend"];
const TESTBENCH_BACKEND: &[&str] = &["testbench-backend"];
const SANDBOX_SERVICES: &[&str] = &["backend", "frontend"];

pub const ALL_APPS: &[AppDef] = &[
    AppDef {
        id: "models",
        title: "Models",
        category: AppCategory::Backend,
        desktop_order: 10,
        required_services: BACKEND,
        content_mode: WindowContentMode::Scroll,
        size: WindowSizeSpec {
            preferred_width: 980.0,
            preferred_height: 680.0,
            min_width: 480.0,
            min_height: 360.0,
        },
    },
    AppDef {
        id: "pipeline",
        title: "Pipeline",
        category: AppCategory::Backend,
        desktop_order: 20,
        required_services: BACKEND,
        content_mode: WindowContentMode::Scroll,
        size: WindowSizeSpec {
            preferred_width: 1040.0,
            preferred_height: 720.0,
            min_width: 520.0,
            min_height: 400.0,
        },
    },
    AppDef {
        id: "siren_gallery",
        title: "SIREN Gallery",
        category: AppCategory::Backend,
        desktop_order: 30,
        required_services: BACKEND,
        content_mode: WindowContentMode::Scroll,
        size: WindowSizeSpec {
            preferred_width: 1040.0,
            preferred_height: 720.0,
            min_width: 360.0,
            min_height: 300.0,
        },
    },
    AppDef {
        id: "backend_api",
        title: "Backend API",
        category: AppCategory::Backend,
        desktop_order: 40,
        required_services: BACKEND,
        content_mode: WindowContentMode::Scroll,
        size: WindowSizeSpec {
            preferred_width: 1000.0,
            preferred_height: 680.0,
            min_width: 480.0,
            min_height: 360.0,
        },
    },
    AppDef {
        id: "dashboard",
        title: "Dashboard",
        category: AppCategory::TestbenchBackend,
        desktop_order: 10,
        required_services: TESTBENCH_BACKEND,
        content_mode: WindowContentMode::Scroll,
        size: WindowSizeSpec {
            preferred_width: 1020.0,
            preferred_height: 680.0,
            min_width: 520.0,
            min_height: 420.0,
        },
    },
    AppDef {
        id: "training",
        title: "Training",
        category: AppCategory::TestbenchBackend,
        desktop_order: 20,
        required_services: TESTBENCH_BACKEND,
        content_mode: WindowContentMode::Scroll,
        size: WindowSizeSpec {
            preferred_width: 1080.0,
            preferred_height: 720.0,
            min_width: 560.0,
            min_height: 440.0,
        },
    },
    AppDef {
        id: "validation",
        title: "Validation",
        category: AppCategory::TestbenchBackend,
        desktop_order: 30,
        required_services: TESTBENCH_BACKEND,
        content_mode: WindowContentMode::Scroll,
        size: WindowSizeSpec {
            preferred_width: 1020.0,
            preferred_height: 680.0,
            min_width: 520.0,
            min_height: 420.0,
        },
    },
    AppDef {
        id: "datasets",
        title: "Datasets",
        category: AppCategory::TestbenchBackend,
        desktop_order: 40,
        required_services: TESTBENCH_BACKEND,
        content_mode: WindowContentMode::Scroll,
        size: WindowSizeSpec {
            preferred_width: 960.0,
            preferred_height: 640.0,
            min_width: 480.0,
            min_height: 360.0,
        },
    },
    AppDef {
        id: "sandbox",
        title: "Sandbox",
        category: AppCategory::Frontend,
        desktop_order: 10,
        required_services: SANDBOX_SERVICES,
        content_mode: WindowContentMode::Fill,
        size: WindowSizeSpec {
            preferred_width: 1180.0,
            preferred_height: 760.0,
            min_width: 480.0,
            min_height: 320.0,
        },
    },
];

pub fn app_by_id(app_id: &str) -> Option<&'static AppDef> {
    ALL_APPS.iter().find(|app| app.id == app_id)
}
pub fn apps_for_category(category: AppCategory) -> Vec<&'static AppDef> {
    let mut apps = ALL_APPS
        .iter()
        .filter(|app| app.category == category)
        .collect::<Vec<_>>();
    apps.sort_by_key(|app| app.desktop_order);
    apps
}

#[component]
pub fn AppIcon(app_id: String) -> Element {
    let body = match app_id.as_str() {
        _ if app_id.starts_with("log:") => rsx! {
            path { d: "M4 5h16M4 10h10M4 15h13M4 20h7" }
        },
        "dashboard" => rsx! {
            rect { x: "3", y: "3", width: "7", height: "9", rx: "1.5" }
            rect { x: "14", y: "3", width: "7", height: "5", rx: "1.5" }
            rect { x: "14", y: "12", width: "7", height: "9", rx: "1.5" }
            rect { x: "3", y: "16", width: "7", height: "5", rx: "1.5" }
        },
        "models" => rsx! {
            path { d: "M12 2.5 21 7v10l-9 4.5L3 17V7z" }
            path { d: "M3 7l9 4.5L21 7M12 11.5V21.5" }
        },
        "pipeline" => rsx! {
            circle { cx: "5", cy: "6", r: "2.5" }
            circle { cx: "5", cy: "18", r: "2.5" }
            circle { cx: "19", cy: "12", r: "2.5" }
            path { d: "M7.5 6H12a4 4 0 0 1 4 4v.5M7.5 18H12a4 4 0 0 0 4-4v-.5" }
        },
        "siren_gallery" => rsx! {
            rect { x: "3", y: "4", width: "18", height: "16", rx: "2.5" }
            path { d: "M3 15c3-5 5.5-5 8.5 0s5.5 2 9-2" }
            circle { cx: "8.5", cy: "9", r: "1.5" }
        },
        "training" => rsx! {
            path { d: "M3 20V4M3 20h18" }
            path { d: "M7 16.5c3-1 4.5-8 7-9.5s4 1 5.5 2.5" }
            circle { cx: "19.5", cy: "9.5", r: "1.6" }
        },
        "validation" => rsx! {
            path { d: "M12 2.8 20 6v6c0 5-3.4 8.1-8 9.4C7.4 20.1 4 17 4 12V6z" }
            path { d: "M8.8 12.2 11.2 14.6 15.6 10" }
        },
        "backend_api" => rsx! {
            rect { x: "3", y: "4", width: "18", height: "6", rx: "2" }
            rect { x: "3", y: "14", width: "18", height: "6", rx: "2" }
            path { d: "M7 7h.01M7 17h.01" }
        },
        "datasets" => rsx! {
            path { d: "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3z" }
            path { d: "M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6" }
            path { d: "M20 12c0 1.7-3.6 3-8 3s-8-1.3-8-3" }
        },
        "sandbox" => rsx! {
            path { d: "M9.5 3v6.2L4.6 17.8A2.4 2.4 0 0 0 6.7 21.4h10.6a2.4 2.4 0 0 0 2.1-3.6L14.5 9.2V3" }
            path { d: "M8 3h8M7.6 15h8.8" }
        },
        _ => rsx! {
            circle { cx: "12", cy: "12", r: "8" }
        },
    };

    rsx! {
        svg {
            class: "w-full h-full",
            view_box: "0 0 24 24",
            fill: "none",
            stroke: "currentColor",
            stroke_width: "1.4",
            stroke_linecap: "round",
            stroke_linejoin: "round",
            {body}
        }
    }
}

pub fn app_content(app_id: &str) -> Element {
    if let Some(service) = app_id.strip_prefix("log:") {
        return rsx! { LogWindow { service: service.to_string() } };
    }
    match app_id {
        "dashboard" => rsx! { Dashboard {} },
        "models" => rsx! { Models {} },
        "pipeline" => rsx! { Pipeline {} },
        "siren_gallery" => rsx! { SirenGallery {} },
        "training" => rsx! { Training {} },
        "validation" => rsx! { Validation {} },
        "backend_api" => rsx! { BackendApi {} },
        "datasets" => rsx! { Datasets {} },
        "sandbox" => rsx! { Sandbox {} },
        other => rsx! { div { class: "p-4 text-white/50 text-sm", "Unknown app: {other}" } },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandbox_requires_backend_and_frontend() {
        assert_eq!(
            app_by_id("sandbox").unwrap().required_services,
            &["backend", "frontend"]
        );
    }

    #[test]
    fn category_order_is_explicit() {
        for category in AppCategory::ALL {
            assert!(
                apps_for_category(category)
                    .windows(2)
                    .all(|pair| pair[0].desktop_order <= pair[1].desktop_order)
            );
        }
    }

    #[test]
    fn app_dimensions_are_valid() {
        for app in ALL_APPS {
            assert!(app.size.preferred_width > 0.0);
            assert!(app.size.preferred_height > 0.0);
            assert!(app.size.min_width > 0.0);
            assert!(app.size.min_height > 0.0);
            assert!(app.size.preferred_width >= app.size.min_width);
            assert!(app.size.preferred_height >= app.size.min_height);
        }
    }

    #[test]
    fn small_client_area_overrides_declared_minimum() {
        let sandbox = app_by_id("sandbox").unwrap();
        assert_eq!(sandbox.size.fit_to_available_space(320.0, 240.0), (320.0, 240.0));
        assert_eq!(sandbox.size.fit_to_available_space(1_920.0, 1_080.0), (1_180.0, 760.0));
    }

    #[test]
    fn sandbox_is_the_only_fill_mode_app() {
        assert_eq!(app_by_id("sandbox").unwrap().content_mode, WindowContentMode::Fill);
        assert!(ALL_APPS
            .iter()
            .filter(|app| app.id != "sandbox")
            .all(|app| app.content_mode == WindowContentMode::Scroll));
    }
}
