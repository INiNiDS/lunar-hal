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
    "snapshot_payload",
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

/// Current schema version for all snapshot payloads and envelopes.
pub const SCHEMA_VERSION: u32 = 1;

/// App-specific values per registered Lunar-OS application (contract v1).
pub const APP_STATE_INVENTORY: &[AppStateEntry] = &[
    AppStateEntry {
        app_id: "dashboard",
        fields: &[],
    },
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
        app_id: "benchmarks",
        fields: &[
            f("model_kind", "ModelKind"),
            f("iterations", "u32"),
            f("warmup", "u32"),
            f("batch_size", "u32"),
            f("selected_report", "Option<String>"),
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
    AppStateEntry {
        app_id: "datasets",
        fields: &[
            f("coverage_dir", "String"),
            f("active_tab", "String"),
            f("filter", "String"),
        ],
    },
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
            snapshot_payload: None,
        };
        assert_eq!(COMMON_WINDOW_STATE_FIELDS.len(), 14);
    }

    #[test]
    fn completeness_round_trip_for_all_inventory_fields() {
        use crate::os::AppSnapshot;
        use crate::os::log_window::LogWindowSnapshot;
        use crate::pages::backend_api::BackendApiSnapshot;
        use crate::pages::benchmarks::BenchmarksSnapshot;
        use crate::pages::datasets::DatasetsSnapshot;
        use crate::pages::models::{ModelsSnapshot, Tab};
        use crate::pages::pipeline::PipelineSnapshot;
        use crate::pages::siren_gallery::SirenGallerySnapshot;
        use crate::pages::training::TrainingSnapshot;
        use crate::pages::validation::ValidationSnapshot;
        use lunar_structures_testbench::ModelKind;

        // 1. Training
        let orig_training = TrainingSnapshot {
            selected: Some("job-1".into()),
            model_kind: ModelKind::Gnn,
            data_path: "path/data.parquet".into(),
            output_dir: "models/out".into(),
            epochs: 25,
            batch_size: 1024,
            lr: 1e-4,
            physics_weight: 0.5,
            val_frac: 0.2,
            gpu_index: 2,
            patience: 15,
            grad_accum: 4,
            clip_grad_norm: 1.5,
            knn_k: 12,
            hidden_dim: 128,
            texture_size: 32,
            max_stars: 4000,
            resume: "ckpt.pt".into(),
            holdout: "holdout.parquet".into(),
            error: Some("err".into()),
            starting: true,
        };
        let p = orig_training.capture_snapshot();
        let mut hyd_training = TrainingSnapshot::default();
        hyd_training.hydrate_snapshot(&p).unwrap();
        assert_eq!(hyd_training, orig_training);

        // 2. Validation
        let orig_validation = ValidationSnapshot {
            selected: Some("val-1".into()),
            model_kind: ModelKind::Siren,
            data_path: "val_data.parquet".into(),
            output_dir: "val_out".into(),
            epochs: 10,
            batch_size: 512,
            val_frac: 0.3,
            knn_k: 6,
            hidden_dim: 64,
            texture_size: 64,
            max_stars: 2000,
            error: Some("val_err".into()),
        };
        let p = orig_validation.capture_snapshot();
        let mut hyd_validation = ValidationSnapshot::default();
        hyd_validation.hydrate_snapshot(&p).unwrap();
        assert_eq!(hyd_validation, orig_validation);

        // 3. Benchmarks
        let orig_benchmarks = BenchmarksSnapshot {
            model_kind: ModelKind::Pinn,
            iterations: 200,
            warmup: 20,
            batch_size: 128,
            selected_report: Some("rep-1".into()),
        };
        let p = orig_benchmarks.capture_snapshot();
        let mut hyd_benchmarks = BenchmarksSnapshot::default();
        hyd_benchmarks.hydrate_snapshot(&p).unwrap();
        assert_eq!(hyd_benchmarks, orig_benchmarks);

        // 4. Models
        let orig_models = ModelsSnapshot {
            tab: Tab::Siren,
            x: 1.2,
            y: 3.4,
            z: 5.6,
            bp_rp: 0.9,
            g_mag: 12.0,
            result: Some(serde_json::json!({"ok": true})),
            error: Some("none".into()),
            busy: false,
        };
        let p = orig_models.capture_snapshot();
        let mut hyd_models = ModelsSnapshot::default();
        hyd_models.hydrate_snapshot(&p).unwrap();
        assert_eq!(hyd_models, orig_models);

        // 5. Pipeline
        let orig_pipeline = PipelineSnapshot {
            x: 7.8,
            y: 9.0,
            z: 11.2,
            bp_rp: 1.1,
            g_mag: 8.5,
            texture_size: 256,
            pipeline_result: Some(serde_json::json!({"pipeline": 1})),
            png_data_url: Some("data:image/png;base64,123".into()),
            png_dims: (128, 128),
            description_result: Some(serde_json::json!({"desc": "star"})),
            random_result: Some(serde_json::json!({"star_id": 42})),
            busy: 3,
        };
        let p = orig_pipeline.capture_snapshot();
        let mut hyd_pipeline = PipelineSnapshot::default();
        hyd_pipeline.hydrate_snapshot(&p).unwrap();
        assert_eq!(hyd_pipeline, orig_pipeline);

        // 6. Siren Gallery
        let orig_gallery = SirenGallerySnapshot {
            selected: None,
            sort: "created_asc".into(),
            refresh_tick: 4,
            status: Some("loaded".into()),
            busy: true,
            name: "Star A".into(),
            bp_rp: 1.3,
            g_mag: 5.5,
            temperature: 6200.0,
            tags: "cluster, test".into(),
        };
        let p = orig_gallery.capture_snapshot();
        let mut hyd_gallery = SirenGallerySnapshot::default();
        hyd_gallery.hydrate_snapshot(&p).unwrap();
        assert_eq!(hyd_gallery, orig_gallery);

        // 7. Backend API
        let orig_backend = BackendApiSnapshot {
            method: "POST".into(),
            path: "/pinn".into(),
            body: "{\"x_pc\": 1.0}".into(),
            query: "tag=1".into(),
            response: Some(serde_json::json!({"out": 2})),
            status_code: Some(200),
            raw_response: Some("{\"out\": 2}".into()),
            error: None,
            busy: false,
            json_err: None,
        };
        let p = orig_backend.capture_snapshot();
        let mut hyd_backend = BackendApiSnapshot::default();
        hyd_backend.hydrate_snapshot(&p).unwrap();
        assert_eq!(hyd_backend, orig_backend);

        // 8. Datasets
        let orig_datasets = DatasetsSnapshot {
            coverage_dir: "/ai_data/coverage".into(),
            active_tab: "models".into(),
            filter: "clean".into(),
        };
        let p = orig_datasets.capture_snapshot();
        let mut hyd_datasets = DatasetsSnapshot::default();
        hyd_datasets.hydrate_snapshot(&p).unwrap();
        assert_eq!(hyd_datasets, orig_datasets);

        // 9. Log Window
        let orig_log = LogWindowSnapshot {
            service: "backend".into(),
            scroll_offset: 420.5,
        };
        let p = orig_log.capture_snapshot();
        let mut hyd_log = LogWindowSnapshot::default();
        hyd_log.hydrate_snapshot(&p).unwrap();
        assert_eq!(hyd_log, orig_log);

        // 10. Sandbox
        use crate::pages::sandbox::SandboxSnapshot;
        let orig_sandbox = SandboxSnapshot {
            selected_scene: Some("scene-stellar".into()),
            retained_iframe_src: Some("http://127.0.0.1:25255/editor".into()),
            snapshot: None,
            selected_star: Some(77),
            camera_offset: (200.0, -150.0),
            camera_zoom: 3.0,
            selected_star_id: Some(77),
        };
        let p = orig_sandbox.capture_snapshot();
        let mut hyd_sandbox = SandboxSnapshot::default();
        hyd_sandbox.hydrate_snapshot(&p).unwrap();
        assert_eq!(hyd_sandbox, orig_sandbox);
    }

    #[test]
    fn schema_version_mismatch_rejected() {
        use crate::os::snapshot::AppSnapshotEnvelopeV1;
        let mut envelope = AppSnapshotEnvelopeV1::new(
            "win-1",
            "training",
            crate::os::snapshot::WindowGeometry {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
            serde_json::json!({"epochs": 50}),
        );
        assert_eq!(envelope.verify_integrity(), Ok(()));
        envelope.version = 2;
        assert!(envelope.verify_integrity().is_err());
    }
}
