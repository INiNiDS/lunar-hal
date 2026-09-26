use crate::api;
use crate::components::ui::{NumberFieldF64, PageHeader, Tag};
use crate::os::AppSnapshot;
use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tab {
    Pinn,
    Gnn,
    Siren,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ModelsSnapshot {
    pub tab: Tab,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub bp_rp: f64,
    pub g_mag: f64,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
    pub busy: bool,
}

impl Default for ModelsSnapshot {
    fn default() -> Self {
        Self {
            tab: Tab::Pinn,
            x: 0.0,
            y: 0.0,
            z: 100.0,
            bp_rp: 1.5,
            g_mag: 10.0,
            result: None,
            error: None,
            busy: false,
        }
    }
}

impl AppSnapshot for ModelsSnapshot {
    fn capture_snapshot(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    fn hydrate_snapshot(&mut self, payload: &serde_json::Value) -> Result<(), String> {
        let snap: ModelsSnapshot = serde_json::from_value(payload.clone())
            .map_err(|e| format!("Models hydration failed: {e}"))?;
        *self = snap;
        Ok(())
    }
}

#[component]
pub fn Models() -> Element {
    let initial = crate::os::use_window_instance_snapshot::<ModelsSnapshot>().unwrap_or_default();
    let mut tab = use_signal(|| initial.tab);
    let x = use_signal(|| initial.x);
    let y = use_signal(|| initial.y);
    let z = use_signal(|| initial.z);
    let bp_rp = use_signal(|| initial.bp_rp);
    let g_mag = use_signal(|| initial.g_mag);
    let result = use_signal(|| initial.result.clone());
    let error = use_signal(|| initial.error.clone());
    let busy = use_signal(|| initial.busy);

    let mut os = crate::os::use_os_state();
    use_effect(move || {
        let snap = ModelsSnapshot {
            tab: tab(),
            x: x(),
            y: y(),
            z: z(),
            bp_rp: bp_rp(),
            g_mag: g_mag(),
            result: result(),
            error: error(),
            busy: busy(),
        };
        if let Some(inst_id) = crate::os::use_window_instance_id() {
            os.register_instance_snapshot(&inst_id, snap.capture_snapshot());
        }
        os.register_app_snapshot("models", snap.capture_snapshot());
    });

    rsx! {
        PageHeader {
            title: "Run Models".to_string(),
            subtitle: "Send single inference requests to each model. Useful for sanity checks, ad-hoc predictions, and rapid iteration on inputs.".to_string(),
        }
        div { class: "page",
            div { class: "tabs",
                button {
                    class: if tab() == Tab::Pinn { "tab active" } else { "tab" },
                    onclick: move |_| tab.set(Tab::Pinn),
                    Tag { text: "PINN".to_string(), kind: "pinn".to_string() }
                    span { style: "margin-left: 8px;", "Stellar property predictor" }
                }
                button {
                    class: if tab() == Tab::Gnn { "tab active" } else { "tab" },
                    onclick: move |_| tab.set(Tab::Gnn),
                    Tag { text: "GNN".to_string(), kind: "gnn".to_string() }
                    span { style: "margin-left: 8px;", "Velocity inference" }
                }
                button {
                    class: if tab() == Tab::Siren { "tab active" } else { "tab" },
                    onclick: move |_| tab.set(Tab::Siren),
                    Tag { text: "SIREN".to_string(), kind: "siren".to_string() }
                    span { style: "margin-left: 8px;", "Texture synthesis" }
                }
            }
            div { style: if tab() == Tab::Pinn { "display: block;" } else { "display: none;" }, PinnPanel { x, y, z, bp_rp, g_mag, result, error, busy } }
            div { style: if tab() == Tab::Gnn { "display: block;" } else { "display: none;" }, GnnPanel {} }
            div { style: if tab() == Tab::Siren { "display: block;" } else { "display: none;" }, SirenPanel {} }
        }
    }
}

#[component]
fn PinnPanel(
    x: Signal<f64>,
    y: Signal<f64>,
    z: Signal<f64>,
    bp_rp: Signal<f64>,
    g_mag: Signal<f64>,
    mut result: Signal<Option<serde_json::Value>>,
    mut error: Signal<Option<String>>,
    mut busy: Signal<bool>,
) -> Element {
    let run = move |_| {
        busy.set(true);
        error.set(None);
        result.set(None);
        let payload = json!({
            "x_pc": x() as f32, "y_pc": y() as f32, "z_pc": z() as f32,
            "bp_rp": bp_rp() as f32, "g_mag": g_mag() as f32,
        });
        spawn(async move {
            match api::pinn_infer(&payload).await {
                Ok(v) => result.set(Some(v)),
                Err(e) => error.set(Some(e)),
            }
            busy.set(false);
        });
    };

    rsx! {
        div { class: "grid grid-2-eq",
            div { class: "card",
                div { class: "card-title", "Inputs" }
                div { class: "grid",
                    NumberFieldF64 { label: "x_pc".to_string(), value: x, step: 0.1 }
                    NumberFieldF64 { label: "y_pc".to_string(), value: y, step: 0.1 }
                    NumberFieldF64 { label: "z_pc".to_string(), value: z, step: 1.0 }
                    NumberFieldF64 { label: "bp_rp".to_string(), value: bp_rp, step: 0.05 }
                    NumberFieldF64 { label: "g_mag".to_string(), value: g_mag, step: 0.1 }
                }
                div { class: "toolbar", style: "margin-top: 14px;",
                    button {
                        class: "btn btn-primary",
                        disabled: busy(),
                        onclick: run,
                        if busy() {
                            span { class: "spinner" }
                        }
                        span { "Run PINN" }
                    }
                }
                div { class: "field-hint", style: "margin-top: 8px;",
                    "Backend endpoint: POST /pinn"
                }
            }
            div { class: "card",
                div { class: "card-title", "Response" }
                if let Some(e) = error() {
                    div { class: "status-banner status-err", "{e}" }
                }
                if let Some(v) = result() {
                    div { class: "code-block", "{v}" }
                } else {
                    div { class: "empty", "No result yet" }
                }
            }
        }
    }
}

#[component]
fn GnnPanel() -> Element {
    let x = use_signal(|| 0.0_f64);
    let y = use_signal(|| 0.0_f64);
    let z = use_signal(|| 100.0_f64);
    let bp_rp = use_signal(|| 1.5_f64);
    let g_mag = use_signal(|| 10.0_f64);
    let radius = use_signal(|| 25.0_f64);
    let temperature = use_signal(|| 0.7_f64);
    let mut result = use_signal(|| None::<serde_json::Value>);
    let mut error = use_signal(|| None::<String>);
    let mut busy = use_signal(|| false);

    let run = move |_| {
        busy.set(true);
        error.set(None);
        result.set(None);
        let payload = json!({
            "center_x": x() as f32, "center_y": y() as f32, "center_z": z() as f32,
            "bp_rp": bp_rp() as f32, "g_mag": g_mag() as f32,
            "search_radius": radius() as f32, "temperature": temperature() as f32,
        });
        spawn(async move {
            match api::gnn_infer(&payload).await {
                Ok(v) => result.set(Some(v)),
                Err(e) => error.set(Some(e)),
            }
            busy.set(false);
        });
    };

    rsx! {
        div { class: "grid grid-2-eq",
            div { class: "card",
                div { class: "card-title", "Inputs" }
                div { class: "grid",
                    NumberFieldF64 { label: "center_x".to_string(), value: x, step: 0.1 }
                    NumberFieldF64 { label: "center_y".to_string(), value: y, step: 0.1 }
                    NumberFieldF64 { label: "center_z".to_string(), value: z, step: 1.0 }
                    NumberFieldF64 { label: "bp_rp".to_string(), value: bp_rp, step: 0.05 }
                    NumberFieldF64 { label: "g_mag".to_string(), value: g_mag, step: 0.1 }
                    NumberFieldF64 { label: "search_radius".to_string(), value: radius, step: 1.0 }
                    NumberFieldF64 { label: "temperature".to_string(), value: temperature, step: 0.05 }
                }
                div { class: "toolbar", style: "margin-top: 14px;",
                    button {
                        class: "btn btn-primary",
                        disabled: busy(),
                        onclick: run,
                        if busy() {
                            span { class: "spinner" }
                        }
                        span { "Run GNN" }
                    }
                }
                div { class: "field-hint", style: "margin-top: 8px;",
                    "Backend endpoint: POST /gnn"
                }
            }
            div { class: "card",
                div { class: "card-title", "Response" }
                if let Some(e) = error() {
                    div { class: "status-banner status-err", "{e}" }
                }
                if let Some(v) = result() {
                    div { class: "code-block", "{v}" }
                } else {
                    div { class: "empty", "No result yet" }
                }
            }
        }
    }
}

#[component]
fn SirenPanel() -> Element {
    let width = use_signal(|| 128_u32);
    let height = use_signal(|| 128_u32);
    let bp_rp = use_signal(|| 1.5_f64);
    let m_g = use_signal(|| 5.0_f64);
    let teff = use_signal(|| 5778.0_f64);
    let mut result = use_signal(|| None::<serde_json::Value>);
    let mut error = use_signal(|| None::<String>);
    let mut busy = use_signal(|| false);

    let run = move |_| {
        busy.set(true);
        error.set(None);
        result.set(None);
        let payload = json!({
            "width": width(), "height": height(),
            "bp_rp": bp_rp() as f32, "m_g": m_g() as f32, "log_teff": (teff() as f32).log10(),
        });
        spawn(async move {
            match api::siren_texture(&payload).await {
                Ok(v) => result.set(Some(v)),
                Err(e) => error.set(Some(e)),
            }
            busy.set(false);
        });
    };

    rsx! {
        div { class: "grid grid-2-eq",
            div { class: "card",
                div { class: "card-title", "Inputs" }
                div { class: "grid",
                    NumberFieldU32 { label: "width".to_string(), value: width, step: 32 }
                    NumberFieldU32 { label: "height".to_string(), value: height, step: 32 }
                    NumberFieldF64 { label: "bp_rp".to_string(), value: bp_rp, step: 0.05 }
                    NumberFieldF64 { label: "m_g".to_string(), value: m_g, step: 0.1 }
                    NumberFieldF64 { label: "T_eff (K)".to_string(), value: teff, step: 100.0 }
                }
                div { class: "toolbar", style: "margin-top: 14px;",
                    button {
                        class: "btn btn-primary",
                        disabled: busy(),
                        onclick: run,
                        if busy() {
                            span { class: "spinner" }
                        }
                        span { "Render texture" }
                    }
                }
                div { class: "field-hint", style: "margin-top: 8px;",
                    "Backend endpoint: POST /siren/texture"
                }
            }
            div { class: "card",
                div { class: "card-title", "Response" }
                if let Some(e) = error() {
                    div { class: "status-banner status-err", "{e}" }
                }
                if let Some(v) = result() {
                    div { class: "code-block", "{v}" }
                } else {
                    div { class: "empty", "No result yet" }
                }
            }
        }
    }
}

#[component]
fn NumberFieldU32(label: String, value: Signal<u32>, step: u32) -> Element {
    rsx! {
        div { class: "field",
            span { class: "field-label", "{label}" }
            input {
                r#type: "number",
                aria_label: "{label}",
                step: "{step}",
                value: "{value()}",
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<u32>() {
                        value.set(v);
                    }
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_models_snapshot_capture_and_hydration() {
        let original = ModelsSnapshot {
            tab: Tab::Gnn,
            x: 12.5,
            y: -34.0,
            z: 250.0,
            bp_rp: 2.1,
            g_mag: 14.2,
            result: Some(serde_json::json!({ "pred": 42 })),
            error: Some("mock err".into()),
            busy: true,
        };
        let payload = original.capture_snapshot();
        let mut restored = ModelsSnapshot::default();
        restored.hydrate_snapshot(&payload).unwrap();
        assert_eq!(restored, original);
    }
}
