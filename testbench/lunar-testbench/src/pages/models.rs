use crate::api;
use crate::components::ui::{NumberFieldF64, PageHeader, Tag};
use dioxus::prelude::*;
use serde_json::json;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Pinn,
    Gnn,
    Siren,
}

#[component]
pub fn Models() -> Element {
    let mut tab = use_signal(|| Tab::Pinn);
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
            match tab() {
                Tab::Pinn => rsx! { PinnPanel {} },
                Tab::Gnn => rsx! { GnnPanel {} },
                Tab::Siren => rsx! { SirenPanel {} },
            }
        }
    }
}

#[component]
fn PinnPanel() -> Element {
    let x = use_signal(|| 0.0_f64);
    let y = use_signal(|| 0.0_f64);
    let z = use_signal(|| 100.0_f64);
    let bp_rp = use_signal(|| 1.5_f64);
    let g_mag = use_signal(|| 10.0_f64);
    let mut result = use_signal(|| None::<serde_json::Value>);
    let mut error = use_signal(|| None::<String>);
    let mut busy = use_signal(|| false);

    let run = move |_| {
        busy.set(true);
        error.set(None);
        result.set(None);
        let payload = json!({
            "x_pc": x() as f32, "y_pc": y() as f32, "z_pc": z() as f32,
            "bp_rp": bp_rp() as f32, "g_mag": g_mag() as f32,
        });
        spawn(async move {
            match api::pinn_infer(payload).await {
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
            match api::gnn_infer(payload).await {
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
            match api::siren_texture(payload).await {
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
