//! Stage-2 contract: inventory of Lunar-OS application state fields.
#![allow(dead_code)]
//!
//! Every Lunar-OS window instance owns two disjoint groups of state:
//!
//! 1. **Common window state** — identical for every app ([`COMMON_WINDOW_STATE_FIELDS`],
//!    backed by `WindowState` in `state.rs`). On minimize it is captured into the
//!    `WindowSnapshotV1` envelope (geometry) plus envelope metadata.
//! 2. **App-specific values** — listed per app in [`APP_STATE_INVENTORY`]. They are
//!    the payload of `WindowSnapshotV1.app_state`: each app serializes them via its
//!    `AppSnapshot::capture_state` implementation when minimize/restore lands
//!    (Stage 12 migrates apps onto snapshot/restore).
//!
//! The inventory is the single source of truth for what must survive a
//! minimize/restore cycle; `app_specific_fields()` resolves both static app ids
//! and dynamic `log:<service>` windows.

/// Fields of the common window state (`WindowState`, `state.rs`) shared by all apps.
pub const COMMON_WINDOW_STATE_FIELDS: &[&str] = &[
    "id",
    "app_id",
    "title",
    "x",
    "y",
    "width",
    "height",
    "min_width",
    "min_height",
    "z",
    "minimized",
    "maximized",
    "restore_rect",
];

/// A single app-specific value that must be captured into `WindowSnapshotV1.app_state`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppSpecificField {
    /// Stable field key used in the snapshot JSON object.
    pub name: &'static str,
    /// Human-readable type/shape note for contract review.
    pub kind: &'static str,
}

/// App-specific field set for one registered app id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppStateEntry {
    pub app_id: &'static str,
    pub fields: &'static [AppSpecificField],
}

const fn f(name: &'static str, kind: &'static str) -> AppSpecificField {
    AppSpecificField { name, kind }
}

/// Fields captured for dynamic `log:<service>` windows.
pub const LOG_WINDOW_FIELDS: &[AppSpecificField] = &[f("service", "String")];

/// App-specific values per registered Lunar-OS application (contract v1).
pub const APP_STATE_INVENTORY: &[AppStateEntry] = &[
    AppStateEntry { app_id: "dashboard", fields: &[] },
    AppStateEntry {
        app_id: "models",
        fields: &[
            f("tab", "Tab"),
            f("x", "f64"),
            f("y", "f64"),
            f("z", "f64"),
            f("bp_rp", "f64"),
            f("g_mag", "f64"),
            f("result", "Option<serde_json::Value>"),
            f("error", "Option<String>"),
            f("busy", "bool"),
        ],
    },
    AppStateEntry {
        app_id: "pipeline",
        fields: &[
            f("x", "f64"),
            f("y", "f64"),
            f("z", "f64"),
            f("bp_rp", "f64"),
            f("g_mag", "f64"),
            f("texture_size", "u32"),
            f("pipeline_result", "Option<serde_json::Value>"),
            f("png_data_url", "Option<String>"),
            f("png_dims", "(u32, u32)"),
            f("description_result", "Option<serde_json::Value>"),
            f("random_result", "Option<serde_json::Value>"),
            f("busy", "u8"),
        ],
    },
    AppStateEntry {
        app_id: "siren_gallery",
        fields: &[
            f("selected", "Option<GalleryStar>"),
            f("sort", "String"),
            f("refresh_tick", "u32"),
            f("status", "Option<String>"),
            f("busy", "bool"),
            f("name", "String"),
            f("bp_rp", "f32"),
            f("g_mag", "f32"),
            f("temperature", "f32"),
            f("tags", "String"),
        ],
    },
    AppStateEntry {
        app_id: "training",
        fields: &[
            f("selected", "Option<String>"),
            f("model_kind", "ModelKind"),
            f("data_path", "String"),
            f("output_dir", "String"),
            f("epochs", "u32"),
            f("batch_size", "u32"),
            f("lr", "f64"),
            f("physics_weight", "f64"),
            f("val_frac", "f64"),
            f("gpu_index", "u32"),
            f("patience", "u32"),
            f("grad_accum", "u32"),
            f("clip_grad_norm", "f64"),
            f("knn_k", "u32"),
            f("hidden_dim", "u32"),
            f("texture_size", "u32"),
            f("max_stars", "u32"),
            f("resume", "String"),
            f("holdout", "String"),
            f("error", "Option<String>"),
            f("starting", "bool"),
        ],
    },
    AppStateEntry {
        app_id: "validation",
        fields: &[
            f("selected", "Option<String>"),
            f("model_kind", "ModelKind"),
            f("data_path", "String"),
            f("output_dir", "String"),
            f("epochs", "u32"),
            f("batch_size", "u32"),
            f("val_frac", "f64"),
            f("knn_k", "u32"),
            f("hidden_dim", "u32"),
            f("texture_size", "u32"),
            f("max_stars", "u32"),
            f("error", "Option<String>"),
        ],
    },
    AppStateEntry {
        app_id: "backend_api",
        fields: &[
            f("method", "String"),
            f("path", "String"),
            f("body", "String"),
            f("query", "String"),
            f("response", "Option<serde_json::Value>"),
            f("status_code", "Option<u16>"),
            f("raw_response", "Option<String>"),
            f("error", "Option<String>"),
            f("busy", "bool"),
            f("json_err", "Option<(usize, usize, String)>"),
        ],
    },
    AppStateEntry { app_id: "datasets", fields: &[] },
    AppStateEntry {
        app_id: "sandbox",
        fields: &[
            f("selected_scene", "Option<String>"),
            f("retained_iframe_src", "Option<String>"),
            f("snapshot", "Option<LiveSceneSnapshot>"),
            f("selected_star", "Option<u32>"),
        ],
    },
];

/// Resolves the app-specific field set for a window app id, including
/// dynamic `log:<service>` windows. Returns `None` for unknown ids.
pub fn app_specific_fields(app_id: &str) -> Option<&'static [AppSpecificField]> {
    if let Some(_service) = app_id.strip_prefix("log:") {
        return Some(LOG_WINDOW_FIELDS);
    }
    APP_STATE_INVENTORY
        .iter()
        .find(|entry| entry.app_id == app_id)
        .map(|entry| entry.fields)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::os::manifest::{ALL_APPS, app_by_id};

    #[test]
    fn every_registered_app_has_an_inventory_entry() {
        for app in ALL_APPS {
            assert!(
                app_specific_fields(app.id).is_some(),
                "missing state inventory for app '{}'",
                app.id
            );
        }
    }

    #[test]
    fn inventory_covers_no_unregistered_apps() {
        for entry in APP_STATE_INVENTORY {
            assert!(
                app_by_id(entry.app_id).is_some(),
                "inventory lists unknown app '{}'",
                entry.app_id
            );
        }
    }

    #[test]
    fn log_windows_resolve_to_log_fields() {
        let fields = app_specific_fields("log:backend").expect("log:* must resolve");
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "service");
    }

    #[test]
    fn unknown_app_ids_do_not_resolve() {
        assert!(app_specific_fields("no_such_app").is_none());
    }

    #[test]
    fn field_names_are_unique_within_each_app() {
        for entry in APP_STATE_INVENTORY {
            let mut names: Vec<_> = entry.fields.iter().map(|field| field.name).collect();
            names.sort_unstable();
            names.dedup();
            assert_eq!(
                names.len(),
                entry.fields.len(),
                "duplicate field names in app '{}'",
                entry.app_id
            );
        }
    }

    #[test]
    fn common_window_state_fields_are_unique() {
        let mut names: Vec<_> = COMMON_WINDOW_STATE_FIELDS.to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), COMMON_WINDOW_STATE_FIELDS.len());
    }

    #[test]
    fn common_window_state_matches_window_state_struct() {
        use crate::os::state::WindowState;
        let _ = WindowState {
            id: 0,
            app_id: String::new(),
            title: String::new(),
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
            min_width: 0.0,
            min_height: 0.0,
            z: 0,
            minimized: false,
            maximized: false,
            restore_rect: None,
        };
        assert_eq!(COMMON_WINDOW_STATE_FIELDS.len(), 13);
    }
}
