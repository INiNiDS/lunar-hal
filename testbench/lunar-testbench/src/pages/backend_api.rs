use crate::api;
use crate::components::ui::{PageHeader, StatusDot, Tag};
use dioxus::prelude::*;

fn fmt_json(s: &str) -> Result<String, String> {
    let v: serde_json::Value = serde_json::from_str(s).map_err(|e| format!("{e}"))?;
    serde_json::to_string_pretty(&v).map_err(|e| format!("{e}"))
}

fn minify_json(s: &str) -> Result<String, String> {
    let v: serde_json::Value = serde_json::from_str(s).map_err(|e| format!("{e}"))?;
    serde_json::to_string(&v).map_err(|e| format!("{e}"))
}

fn json_err_linecol(s: &str) -> Option<(usize, usize, String)> {
    match serde_json::from_str::<serde_json::Value>(s) {
        Ok(_) => None,
        Err(e) => {
            let line = s[..e.column()].matches('\n').count() + 1;
            let last_newl = s[..e.column()].rfind('\n').map(|p| p + 1).unwrap_or(0);
            let col = e.column() - last_newl + 1;
            Some((line, col, e.to_string()))
        }
    }
}

#[component]
pub fn BackendApi() -> Element {
    let mut method = use_signal(|| "GET".to_string());
    let mut path = use_signal(|| "/".to_string());
    let mut body = use_signal(|| "{}".to_string());
    let mut query = use_signal(|| String::new());
    let mut response = use_signal(|| None::<serde_json::Value>);
    let mut status_code = use_signal(|| None::<u16>);
    let mut raw_response = use_signal(|| None::<String>);
    let mut error = use_signal(|| None::<String>);
    let mut busy = use_signal(|| false);
    let mut json_err = use_signal(|| None::<(usize, usize, String)>);
    let placeholder_text = r#"{"key": "value"}"#;

    let json_info = move || {
        if method() != "POST" { return None; }
        let raw = body();
        match serde_json::from_str::<serde_json::Value>(&raw) {
            Ok(v) => {
                let type_label = match &v {
                    serde_json::Value::Object(_) => "object",
                    serde_json::Value::Array(_) => "array",
                    serde_json::Value::String(_) => "string",
                    serde_json::Value::Number(_) => "number",
                    serde_json::Value::Bool(_) => "bool",
                    serde_json::Value::Null => "null",
                };
                let count = match &v {
                    serde_json::Value::Object(m) => m.len(),
                    serde_json::Value::Array(a) => a.len(),
                    _ => 0,
                };
                Some((type_label, count))
            }
            Err(_) => None,
        }
    };

    let send = move |_| {
        busy.set(true);
        error.set(None);
        response.set(None);
        raw_response.set(None);
        status_code.set(None);
        json_err.set(None);

        let parsed_body: Option<serde_json::Value> = if method().to_uppercase() == "GET" {
            None
        } else {
            match serde_json::from_str(&body()) {
                Ok(v) => Some(v),
                Err(e) => {
                    let err_info = json_err_linecol(&body()).unwrap_or((0, 0, format!("{e}")));
                    json_err.set(Some(err_info.clone()));
                    error.set(Some(format!("JSON error at line {}, col {}: {e}", err_info.0, err_info.1)));
                    busy.set(false);
                    return;
                }
            }
        };

        let req_path = path();
        let req_method = method();
        let req_query = if query().is_empty() { None } else { Some(query()) };
        spawn(async move {
            match api::backend_proxy(req_path, req_method, parsed_body, req_query).await {
                Ok(v) => {
                    status_code.set(Some(200));
                    let pretty = serde_json::to_string_pretty(&v).unwrap_or_default();
                    raw_response.set(Some(pretty));
                    response.set(Some(v));
                }
                Err(msg) => {
                    if let Some(rest) = msg.split_whitespace().nth(1) {
                        if let Ok(code) = rest.parse::<u16>() {
                            status_code.set(Some(code));
                        }
                    }
                    error.set(Some(msg));
                }
            }
            busy.set(false);
        });
    };

    rsx! {
        PageHeader {
            title: "Backend API".to_string(),
            subtitle: "Ad-hoc request builder with JSON editor. Probe any endpoint exposed by lunar-backend directly from the browser.".to_string(),
        }
        div { class: "page",
            div { class: "alert-banner",
                div { class: "icon", "↗" }
                div {
                    div { class: "mono", "Base URL: http://127.0.0.1:25255 (lunar-backend)" }
                    div { class: "field-hint", style: "margin-top: 4px;",
                        "Browser talks to the backend directly. Make sure lunar-backend is running on 127.0.0.1:25255."
                    }
                    div { class: "field-hint", style: "margin-top: 2px;",
                        "Pipeline &amp; system snapshot moved to testbench-backend (127.0.0.1:25256)."
                    }
                }
            }
            div { class: "grid grid-2-eq",
                div { class: "card",
                    div { class: "card-title", "Request" }
                    div { class: "grid",
                        div { class: "field",
                            span { class: "field-label", "Method" }
                            div { class: "row",
                                for m in ["GET", "POST"].iter() {
                                    button {
                                        class: if method() == *m { "btn btn-primary btn-sm" } else { "btn btn-sm" },
                                        onclick: move |_| method.set(m.to_string()),
                                        "{m}"
                                    }
                                }
                            }
                        }
                        div { class: "field",
                            span { class: "field-label", "Path" }
                            input {
                                value: "{path()}",
                                oninput: move |e| path.set(e.value()),
                                placeholder: "/pinn",
                            }
                        }
                        div { class: "field",
                            span { class: "field-label", "Query string" }
                            input {
                                value: "{query()}",
                                oninput: move |e| query.set(e.value()),
                                placeholder: "width=128&height=128",
                            }
                        }
                        if method() == "POST" {
                            div { class: "field",
                                div { class: "row", style: "justify-content: space-between;",
                                    span { class: "field-label", "JSON body" }
                                    div { class: "row", style: "gap: 4px;",
                                        button {
                                            class: "btn btn-sm",
                                            onclick: move |_| {
                                                match fmt_json(&body()) {
                                                    Ok(f) => body.set(f),
                                                    Err(e) => {
                                                        let err_info = json_err_linecol(&body()).unwrap_or((0, 0, e.clone()));
                                                        json_err.set(Some(err_info.clone()));
                                                        error.set(Some(format!("Invalid JSON at line {}, col {}: {e}", err_info.0, err_info.1)));
                                                    }
                                                }
                                            },
                                            title: "Format JSON",
                                            "{{}} ⟶"
                                        }
                                        button {
                                            class: "btn btn-sm",
                                            onclick: move |_| {
                                                match minify_json(&body()) {
                                                    Ok(m) => body.set(m),
                                                    Err(e) => {
                                                        let err_info = json_err_linecol(&body()).unwrap_or((0, 0, e));
                                                        json_err.set(Some(err_info));
                                                    }
                                                }
                                            },
                                            title: "Minify JSON",
                                            "{{}} ⟵"
                                        }

                                    }
                                }
                                textarea {
                                    class: "textarea-mono json-editor",
                                    value: "{body()}",
                                    oninput: move |e| {
                                        body.set(e.value());
                                        json_err.set(None);
                                    },
                                    placeholder: placeholder_text,
                                    rows: "14",
                                }
                            }
                            if let Some((line, col, _)) = json_err() {
                                div { class: "status-banner status-warn", style: "margin-top: 4px;",
                                    span { "⚠ JSON error at line {line}, column {col}" }
                                }
                            }
                            div { class: "json-status",
                                span { "lines: {body().lines().count()}" }
                                span { " chars: {body().len()}" }
                                if let Some((tl, n)) = json_info() {
                                    span { "type: {tl}" }
                                    span { "{n} keys" }
                                }
                            }
                        }
                    }
                    div { class: "toolbar", style: "margin-top: 14px;",
                        button {
                            class: "btn btn-primary",
                            disabled: busy(),
                            onclick: send,
                            if busy() { span { class: "spinner" } }
                            span { "Send" }
                        }
                    }
                }
                div { class: "card",
                    div { class: "card-title",
                        StatusDot { status: status_code().map(|c| if c < 400 { "ok" } else { "err" }.to_string()).unwrap_or_else(|| "off".to_string()) }
                        span { "Response" }
                        if let Some(c) = status_code() {
                            Tag {
                                text: c.to_string(),
                                kind: if c < 300 { "ok".to_string() } else if c < 400 { "warn".to_string() } else { "err".to_string() },
                            }
                        }
                        if raw_response().is_some() {
                            button {
                                class: "btn btn-sm",
                                style: "margin-left: auto;",
                                onclick: move |_| {
                                    if let Some(raw) = raw_response() {
                                        match fmt_json(&raw) {
                                            Ok(f) => raw_response.set(Some(f)),
                                            Err(_) => {}
                                        }
                                    }
                                },
                                title: "Format JSON",
                                "{{}} ⟶"
                            }
                            button {
                                class: "btn btn-sm",
                                onclick: move |_| {
                                    if let Some(raw) = raw_response() {
                                        match minify_json(&raw) {
                                            Ok(m) => raw_response.set(Some(m)),
                                            Err(_) => {}
                                        }
                                    }
                                },
                                title: "Minify JSON",
                                "{{}} ⟵"
                            }
                            button {
                                class: "btn btn-sm",
                                onclick: move |_| {
                                    if let Some(raw) = raw_response() {
                                        body.set(raw.clone());
                                        json_err.set(None);
                                        error.set(None);
                                    }
                                },
                                title: "Copy to body",
                                "⬇ body"
                            }
                        }
                    }
                    if let Some(e) = error() {
                        div { class: "status-banner status-err", "{e}" }
                    }
                    if let Some(raw) = raw_response() {
                        div { class: "code-block", "{raw}" }
                    } else if response().is_none() {
                        div { class: "empty", "No response yet" }
                    }
                }
            }
            div { class: "section-title", "Quick templates" }
            div { class: "row",
                for t in [
                    ("POST", "/pinn", r#"{"x_pc":0,"y_pc":0,"z_pc":100,"bp_rp":1.5,"g_mag":10}"#),
                    ("POST", "/gnn", r#"{"center_x":0,"center_y":0,"center_z":100,"bp_rp":1.5,"g_mag":10,"search_radius":25,"temperature":0.7}"#),
                    ("POST", "/random_star", r#"{"entropy_temperature":0.5}"#),
                    ("POST", "/siren/texture", r#"{"width":128,"height":128,"bp_rp":1.5,"m_g":5.0,"log_teff":3.76}"#),
                    ("GET", "/siren/png", ""),
                ].iter() {
                    button {
                        class: "btn btn-sm",
                        onclick: move |_| {
                            method.set(t.0.to_string());
                            path.set(t.1.to_string());
                            body.set(t.2.to_string());
                            json_err.set(None);
                            error.set(None);
                        },
                        "{t.0} {t.1}"
                    }
                }
            }
        }
    }
}
