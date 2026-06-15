use crate::api;
use crate::components::ui::{base64_encode, NumberFieldF64, NumberFieldU32, PageHeader, StatusDot, Tag};
use dioxus::prelude::*;
use serde_json::json;

#[component]
pub fn Pipeline() -> Element {
    let mut x = use_signal(|| 0.0_f64);
    let mut y = use_signal(|| 0.0_f64);
    let mut z = use_signal(|| 100.0_f64);
    let mut bp_rp = use_signal(|| 1.5_f64);
    let mut g_mag = use_signal(|| 10.0_f64);
    let texture_size = use_signal(|| 128_u32);

    let mut pipeline_result = use_signal(|| None::<serde_json::Value>);
    let mut png_data_url = use_signal(|| None::<String>);
    let mut png_dims = use_signal(|| (0_u32, 0_u32));
    let mut description_result = use_signal(|| None::<serde_json::Value>);
    let mut random_result = use_signal(|| None::<serde_json::Value>);

    let mut busy = use_signal(|| 0_u8);
    let mut error = use_signal(|| None::<String>);

    let run_pipeline = move |_| {
        busy.set(1);
        error.set(None);
        let payload = json!({
            "x_pc": x() as f32, "y_pc": y() as f32, "z_pc": z() as f32,
            "bp_rp": bp_rp() as f32, "g_mag": g_mag() as f32,
            "texture_size": texture_size(),
        });
        spawn(async move {
            match api::pipeline(payload).await {
                Ok(v) => pipeline_result.set(Some(v)),
                Err(e) => error.set(Some(e)),
            }
            busy.set(0);
        });
    };

    let run_png = move |_| {
        busy.set(2);
        error.set(None);
        let q = api::PipelinePngQuery {
            x_pc: x() as f32,
            y_pc: y() as f32,
            z_pc: z() as f32,
            bp_rp: bp_rp() as f32,
            g_mag: g_mag() as f32,
            size: texture_size(),
        };
        spawn(async move {
            match api::pipeline_png(q).await {
                Ok((bytes, w, h)) => {
                    let b64 = base64_encode(&bytes);
                    png_data_url.set(Some(format!("data:image/png;base64,{}", b64)));
                    png_dims.set((w, h));
                }
                Err(e) => error.set(Some(e)),
            }
            busy.set(0);
        });
    };

    let run_random = move |_| {
        busy.set(3);
        error.set(None);
        let payload = json!({ "entropy_temperature": 1.0_f32 });
        spawn(async move {
            match api::random_star(payload).await {
                Ok(v) => {
                    if let (Some(bp), Some(gm), Some(xx), Some(yy), Some(zz)) = (
                        v.get("bp_rp").and_then(|x| x.as_f64()),
                        v.get("g_mag").and_then(|x| x.as_f64()),
                        v.get("x_pc").and_then(|x| x.as_f64()),
                        v.get("y_pc").and_then(|x| x.as_f64()),
                        v.get("z_pc").and_then(|x| x.as_f64()),
                    ) {
                        bp_rp.set(bp);
                        g_mag.set(gm);
                        x.set(xx);
                        y.set(yy);
                        z.set(zz);
                    }
                    random_result.set(Some(v));
                }
                Err(e) => error.set(Some(e)),
            }
            busy.set(0);
        });
    };

    let run_description = move |_| {
        busy.set(4);
        error.set(None);
        let payload = json!({
            "pinn_payload": {
                "temperature_k": 5778.0_f32,
                "radius_solar": 1.0_f32,
                "mass_solar": 1.0_f32,
                "luminosity_solar": 1.0_f32,
            },
            "gnn_payload": { "stars": [] },
        });
        spawn(async move {
            match api::description(payload).await {
                Ok(v) => description_result.set(Some(v)),
                Err(e) => error.set(Some(e)),
            }
            busy.set(0);
        });
    };

    rsx! {
        PageHeader {
            title: "Pipeline Composer".to_string(),
            subtitle: "Compose full PINN → SIREN → metadata pipelines. Build and probe the same flows used by the front-end, in isolation.".to_string(),
        }
        div { class: "page",
            div { class: "split",
                div { class: "card",
                    div { class: "card-title", "Coordinates" }
                    div { class: "grid",
                        NumberFieldF64 { label: "x_pc".to_string(), value: x, step: 0.1 }
                        NumberFieldF64 { label: "y_pc".to_string(), value: y, step: 0.1 }
                        NumberFieldF64 { label: "z_pc".to_string(), value: z, step: 1.0 }
                        NumberFieldF64 { label: "bp_rp".to_string(), value: bp_rp, step: 0.05 }
                        NumberFieldF64 { label: "g_mag".to_string(), value: g_mag, step: 0.1 }
                        NumberFieldU32 { label: "texture_size".to_string(), value: texture_size }
                    }
                    if let Some(e) = error() {
                        div { class: "status-banner status-err", "{e}" }
                    }
                    div { class: "section-title", "Pipeline steps" }
                    div { class: "grid",
                        button { class: "btn btn-primary",
                            disabled: busy() != 0,
                            onclick: run_pipeline,
                            if busy() == 1 { span { class: "spinner" } }
                            span { "Run /pipeline (JSON)" }
                        }
                        button { class: "btn",
                            disabled: busy() != 0,
                            onclick: run_png,
                            if busy() == 2 { span { class: "spinner" } }
                            span { "Run /pipeline/png" }
                        }
                        button { class: "btn",
                            disabled: busy() != 0,
                            onclick: run_random,
                            if busy() == 3 { span { class: "spinner" } }
                            span { "Random star (entropy=1.0)" }
                        }
                        button { class: "btn",
                            disabled: busy() != 0,
                            onclick: run_description,
                            if busy() == 4 { span { class: "spinner" } }
                            span { "Run /description" }
                        }
                    }
                }
                div {
                    div { class: "card",
                        div { class: "card-title",
                            StatusDot { status: if png_data_url().is_some() { "ok".to_string() } else { "off".to_string() } }
                            span { "SIREN texture" }
                            Tag { text: "from /pipeline/png".to_string(), kind: "siren".to_string() }
                        }
                        if let Some(url) = png_data_url() {
                            img {
                                src: "{url}",
                                width: "{png_dims().0}",
                                height: "{png_dims().1}",
                                style: "image-rendering: pixelated; max-width: 100%; border-radius: 8px; border: 1px solid var(--border); background: #000;",
                            }
                        } else {
                            div { class: "empty", "Press /pipeline/png" }
                        }
                    }
                    div { class: "card", style: "margin-top: 16px;",
                        div { class: "card-title",
                            StatusDot { status: if pipeline_result().is_some() { "ok".to_string() } else { "off".to_string() } }
                            span { "Pipeline JSON" }
                            Tag { text: "POST /pipeline".to_string(), kind: "pinn".to_string() }
                        }
                        if let Some(v) = pipeline_result() {
                            div { class: "code-block", "{v}" }
                        } else {
                            div { class: "empty", "Press /pipeline" }
                        }
                    }
                    div { class: "card", style: "margin-top: 16px;",
                        div { class: "card-title",
                            StatusDot { status: if random_result().is_some() { "ok".to_string() } else { "off".to_string() } }
                            span { "Random star" }
                            Tag { text: "POST /random_star".to_string(), kind: "gnn".to_string() }
                        }
                        if let Some(v) = random_result() {
                            div { class: "code-block", "{v}" }
                        } else {
                            div { class: "empty", "Press Random star" }
                        }
                    }
                    div { class: "card", style: "margin-top: 16px;",
                        div { class: "card-title",
                            StatusDot { status: if description_result().is_some() { "ok".to_string() } else { "off".to_string() } }
                            span { "Description" }
                            Tag { text: "POST /description".to_string(), kind: "siren".to_string() }
                        }
                        if let Some(v) = description_result() {
                            div { class: "code-block", "{v}" }
                        } else {
                            div { class: "empty", "Press /description" }
                        }
                    }
                }
            }
        }
    }
}


