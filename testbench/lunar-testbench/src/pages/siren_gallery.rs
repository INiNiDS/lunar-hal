use crate::api;
use crate::components::StarShaderCanvas;
use crate::components::ui::{base64_encode, NumberFieldU32, PageHeader, StatusDot};
use dioxus::prelude::*;

#[component]
pub fn SirenGallery() -> Element {
    let width = use_signal(|| 128_u32);
    let height = use_signal(|| 128_u32);
    let mut bp_rp = use_signal(|| 1.5_f32);
    let mut m_g = use_signal(|| 5.0_f32);
    let mut teff = use_signal(|| 5778.0_f32);
    let mut data_url = use_signal(|| None::<String>);
    let mut dims = use_signal(|| (0_u32, 0_u32));
    let mut error = use_signal(|| None::<String>);
    let mut busy = use_signal(|| false);

    let presets: Vec<(f32, f32, f32, String)> = vec![
        (1.20, 4.5, 6500.0, "F-type main seq".to_string()),
        (0.85, 4.8, 5778.0, "Sun-like G".to_string()),
        (1.60, 6.5, 4500.0, "K-type".to_string()),
        (3.10, 8.5, 3200.0, "Cool M-dwarf".to_string()),
        (-0.20, -1.0, 12000.0, "Hot A-star".to_string()),
        (0.30, 1.2, 9500.0, "B-type".to_string()),
        (1.85, 10.0, 3900.0, "Red giant".to_string()),
        (3.20, 12.0, 2900.0, "Cool giant".to_string()),
    ];

    let run = move |_| {
        busy.set(true);
        error.set(None);
        let q = api::SirenPngQuery {
            width: width(),
            height: height(),
            bp_rp: bp_rp(),
            m_g: m_g(),
            temperature_k: teff(),
        };
        spawn(async move {
            match api::siren_png(q).await {
                Ok((bytes, w, h)) => {
                    let b64 = base64_encode(&bytes);
                    data_url.set(Some(format!("data:image/png;base64,{}", b64)));
                    dims.set((w, h));
                }
                Err(e) => error.set(Some(e)),
            }
            busy.set(false);
        });
    };

    let noise_scale = use_signal(|| 2.0_f32);
    let noise_speed = use_signal(|| 1.0_f32);
    let shader_contrast = use_signal(|| 0.35_f32);
    let mut show_shader = use_signal(|| false);

    rsx! {
        div { class: "page",
            // PageHeader перенесён внутрь контейнера ".page" для ровного центрирования с сеткой
            PageHeader {
                title: "SIREN Gallery".to_string(),
                subtitle: "Render stellar surface textures via the SIREN network. Tweak Bp-Rp, M_G, and effective temperature to explore the latent space.".to_string(),
            }

            div { class: "split",
                div { class: "card",
                    div { class: "card-title", "Renderer parameters" }
                    div { class: "grid",
                        NumberFieldU32 { label: "width".to_string(), value: width }
                        NumberFieldU32 { label: "height".to_string(), value: height }
                        NumberFieldF32 { label: "bp_rp".to_string(), value: bp_rp, step: 0.05 }
                        NumberFieldF32 { label: "M_G".to_string(), value: m_g, step: 0.1 }
                        NumberFieldF32 { label: "T_eff (K)".to_string(), value: teff, step: 100.0 }
                    }
                    div { class: "section-title", "Quick presets" }
                    div { class: "row",
                        for p in presets.clone().into_iter() {
                            {
                                let b = p.0;
                                let m = p.1;
                                let t = p.2;
                                let name = p.3;
                                rsx! {
                                    button {
                                        class: "btn btn-sm",
                                        onclick: move |_| {
                                            bp_rp.set(b);
                                            m_g.set(m);
                                            teff.set(t);
                                        },
                                        "{name}"
                                    }
                                }
                            }
                        }
                    }
                    div { class: "toolbar", style: "margin-top: 16px;",
                        button {
                            class: "btn btn-primary",
                            disabled: busy(),
                            onclick: run,
                            if busy() { span { class: "spinner" } }
                            span { "Render" }
                        }
                    }
                    if let Some(e) = error() {
                        div { class: "status-banner status-err", style: "margin-top: 12px;", "{e}" }
                    }
                }
                div { class: "card",
                    div { class: "card-title",
                        StatusDot { status: if data_url().is_some() { "ok".to_string() } else { "off".to_string() } }
                        span { "Latest render" }
                        if data_url().is_some() {
                            span { class: "mono", style: "margin-left: auto; color: var(--text-3);",
                                "{dims().0}x{dims().1}"
                            }
                        }
                    }
                    if let Some(url) = data_url() {
                        img {
                            src: "{url}",
                            width: "{dims().0}",
                            height: "{dims().1}",
                            style: "image-rendering: pixelated; max-width: 100%; border-radius: 8px; border: 1px solid var(--border); background: #000;",
                        }
                    } else {
                        div { class: "empty", "No render yet" }
                    }
                }
            }

            div { class: "card", style: "margin-top: 16px;",
                div { class: "card-title",
                    StatusDot { status: if show_shader() { "ok" } else { "off" } }
                    span { "GPU Star Shader" }
                    span { class: "mono", style: "margin-left: auto; font-size: 11px; color: var(--text-3);", "Rust + web-sys &middot; WebGL2" }
                }
                div { style: "display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px;",
                    p { style: "margin: 0; font-size: 13px; color: var(--text-2); flex: 1;",
                        "Live GPU-accelerated star shader. Pure Rust via web-sys — no JavaScript."
                    }
                    button {
                        class: "btn btn-primary",
                        onclick: move |_| show_shader.set(!show_shader()),
                        if show_shader() { "Hide Shader" } else { "Show Shader" }
                    }
                }
                if show_shader() {
                    div { class: "grid", style: "margin-bottom: 12px;",
                        NumberFieldF32 { label: "noise_scale".to_string(), value: noise_scale, step: 0.1 }
                        NumberFieldF32 { label: "noise_speed".to_string(), value: noise_speed, step: 0.1 }
                        NumberFieldF32 { label: "contrast".to_string(), value: shader_contrast, step: 0.05 }
                    }
                    div { style: "width: 100%; height: 520px; border-radius: 8px; overflow: hidden;",
                        StarShaderCanvas {
                            width: 520,
                            height: 520,
                            teff: teff() as f64,
                            bp_rp: bp_rp() as f64,
                            noise_scale: noise_scale() as f64,
                            noise_speed: noise_speed() as f64,
                            contrast: shader_contrast() as f64,
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn NumberFieldF32(label: String, value: Signal<f32>, step: f32) -> Element {
    rsx! {
        div { class: "field",
            span { class: "field-label", "{label}" }
            input {
                r#type: "number",
                step: "{step}",
                value: "{value()}",
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<f32>() {
                        value.set(v);
                    }
                },
            }
        }
    }
}