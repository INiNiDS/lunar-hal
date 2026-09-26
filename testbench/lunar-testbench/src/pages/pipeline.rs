use crate::api;
use crate::components::ui::{
    NumberFieldF64, NumberFieldU32, PageHeader, StatusDot, Tag, base64_encode,
};
use crate::os::state::use_window_lifecycle;
use crate::os::{
    AppSnapshot, WindowLifecycle, use_os_state, use_window_instance_id,
    use_window_instance_snapshot,
};
use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PipelineSnapshot {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub bp_rp: f64,
    pub g_mag: f64,
    pub texture_size: u32,
    pub pipeline_result: Option<serde_json::Value>,
    pub png_data_url: Option<String>,
    pub png_dims: (u32, u32),
    pub description_result: Option<serde_json::Value>,
    pub random_result: Option<serde_json::Value>,
    pub busy: u8,
}

impl Default for PipelineSnapshot {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 100.0,
            bp_rp: 1.5,
            g_mag: 10.0,
            texture_size: 128,
            pipeline_result: None,
            png_data_url: None,
            png_dims: (0, 0),
            description_result: None,
            random_result: None,
            busy: 0,
        }
    }
}

impl AppSnapshot for PipelineSnapshot {
    fn capture_snapshot(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    fn hydrate_snapshot(&mut self, payload: &serde_json::Value) -> Result<(), String> {
        let snap: PipelineSnapshot = serde_json::from_value(payload.clone())
            .map_err(|e| format!("Pipeline hydration failed: {e}"))?;
        *self = snap;
        Ok(())
    }
}

#[component]
pub fn Pipeline() -> Element {
    let initial = use_window_instance_snapshot::<PipelineSnapshot>().unwrap_or_default();
    let mut x = use_signal(|| initial.x);
    let mut y = use_signal(|| initial.y);
    let mut z = use_signal(|| initial.z);
    let mut bp_rp = use_signal(|| initial.bp_rp);
    let mut g_mag = use_signal(|| initial.g_mag);
    let texture_size = use_signal(|| initial.texture_size);

    let mut pipeline_result = use_signal(|| initial.pipeline_result.clone());
    let mut png_data_url = use_signal(|| initial.png_data_url.clone());
    let mut png_dims = use_signal(|| initial.png_dims);
    let mut description_result = use_signal(|| initial.description_result.clone());
    let mut random_result = use_signal(|| initial.random_result.clone());

    let mut busy = use_signal(|| initial.busy);
    let mut error = use_signal(|| None::<String>);
    let mut pipeline_result_stale = use_signal(|| false);

    let mut os = use_os_state();
    use_effect(move || {
        let snap = PipelineSnapshot {
            x: x(),
            y: y(),
            z: z(),
            bp_rp: bp_rp(),
            g_mag: g_mag(),
            texture_size: texture_size(),
            pipeline_result: pipeline_result(),
            png_data_url: png_data_url(),
            png_dims: png_dims(),
            description_result: description_result(),
            random_result: random_result(),
            busy: busy(),
        };
        if let Some(inst_id) = use_window_instance_id() {
            os.register_instance_snapshot(&inst_id, snap.capture_snapshot());
        }
        os.register_app_snapshot("pipeline", snap.capture_snapshot());
    });

    let run_pipeline = move |_| {
        busy.set(1);
        error.set(None);
        let payload = json!({
            "x_pc": x() as f32, "y_pc": y() as f32, "z_pc": z() as f32,
            "bp_rp": bp_rp() as f32, "g_mag": g_mag() as f32,
            "texture_size": texture_size(),
        });
        spawn(async move {
            match api::pipeline(&payload).await {
                Ok(v) => {
                    pipeline_result.set(Some(v));
                    pipeline_result_stale.set(false);
                }
                Err(e) => {
                    if pipeline_result().is_some() {
                        pipeline_result_stale.set(true);
                    }
                    error.set(Some(e));
                }
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
            match api::pipeline_png(&q).await {
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
            match api::random_star(&payload).await {
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
            match api::description(&payload).await {
                Ok(v) => description_result.set(Some(v)),
                Err(e) => error.set(Some(e)),
            }
            busy.set(0);
        });
    };

    let lifecycle = use_window_lifecycle();
    use_effect(move || {
        if lifecycle
            .map(|signal| *signal.read() == WindowLifecycle::Blocked)
            .unwrap_or(false)
            && pipeline_result().is_some()
            && !*pipeline_result_stale.peek()
        {
            pipeline_result_stale.set(true);
        }
    });

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
                                class: "aspect-preview-image",
                                style: "image-rendering: pixelated;",
                            }
                        } else {
                            div { class: "empty", "Press /pipeline/png" }
                        }
                    }
                    div { class: "card", style: "margin-top: 16px;",
                        div { class: "card-title",
                            StatusDot { status: if pipeline_result().is_some() { "ok".to_string() } else { "off".to_string() } }
                            span { "Pipeline JSON" }
                            if pipeline_result_stale() || (error().is_some() && pipeline_result().is_some()) {
                                Tag { text: "STALE".to_string(), kind: "warn".to_string() }
                            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_snapshot_capture_and_hydration() {
        let original = PipelineSnapshot {
            x: 10.0,
            y: 20.0,
            z: 30.0,
            bp_rp: 1.25,
            g_mag: 9.8,
            texture_size: 256,
            pipeline_result: Some(serde_json::json!({ "status": "ok" })),
            png_data_url: Some("data:image/png;base64,AAAA".into()),
            png_dims: (256, 256),
            description_result: Some(serde_json::json!({ "desc": "solar" })),
            random_result: Some(serde_json::json!({ "star": 99 })),
            busy: 2,
        };
        let payload = original.capture_snapshot();
        let mut restored = PipelineSnapshot::default();
        restored.hydrate_snapshot(&payload).unwrap();
        assert_eq!(restored, original);
    }
}
