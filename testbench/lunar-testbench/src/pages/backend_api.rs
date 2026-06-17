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

fn json_type_label(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Object(_) => "object",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Null => "null",
    }
}

fn json_type_count(v: &serde_json::Value) -> usize {
    match v {
        serde_json::Value::Object(m) => m.len(),
        serde_json::Value::Array(a) => a.len(),
        _ => 0,
    }
}

struct PreparedRequest {
    method: String,
    path: String,
    body: Option<serde_json::Value>,
    query: Option<String>,
    json_err: Option<(usize, usize, String)>,
    error: Option<String>,
}

fn prepare_request(
    method: &str,
    path: &str,
    body: &str,
    query: &str,
) -> PreparedRequest {
    let parsed_body: Option<serde_json::Value> = if method.to_uppercase() == "GET" {
        None
    } else {
        match serde_json::from_str(body) {
            Ok(v) => Some(v),
            Err(e) => {
                let err_info = json_err_linecol(body).unwrap_or((0, 0, format!("{e}")));
                return PreparedRequest {
                    method: method.to_string(),
                    path: path.to_string(),
                    body: None,
                    query: if query.is_empty() { None } else { Some(query.to_string()) },
                    json_err: Some(err_info.clone()),
                    error: Some(format!("JSON error at line {}, col {}: {e}", err_info.0, err_info.1)),
                };
            }
        }
    };

    PreparedRequest {
        method: method.to_string(),
        path: path.to_string(),
        body: parsed_body,
        query: if query.is_empty() { None } else { Some(query.to_string()) },
        json_err: None,
        error: None,
    }
}

fn apply_request_to_signals(
    prep: &PreparedRequest,
    status_code: &mut Signal<Option<u16>>,
    response: &mut Signal<Option<serde_json::Value>>,
    raw_response: &mut Signal<Option<String>>,
    json_err: &mut Signal<Option<(usize, usize, String)>>,
    error: &mut Signal<Option<String>>,
) {
    status_code.set(None);
    response.set(None);
    raw_response.set(None);
    json_err.set(None);
    error.set(None);

    if let Some(err_info) = &prep.json_err {
        json_err.set(Some(err_info.clone()));
    }
    if let Some(err) = &prep.error {
        error.set(Some(err.clone()));
    }
}

fn extract_status_code_from_err(msg: &str) -> Option<u16> {
    msg.split_whitespace().nth(1).and_then(|rest| rest.parse::<u16>().ok())
}

#[component]
fn RequestBodyEditor(
    mut body: Signal<String>,
    mut json_err: Signal<Option<(usize, usize, String)>>,
    mut error: Signal<Option<String>>,
) -> Element {
    let placeholder_text = r#"{"key": "value"}"#;

    rsx! {
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
    }
}

#[component]
fn MethodSelector(mut method: Signal<String>) -> Element {
    rsx! {
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
    }
}

#[component]
fn PathField(mut path: Signal<String>) -> Element {
    rsx! {
        div { class: "field",
            span { class: "field-label", "Path" }
            input {
                value: "{path()}",
                oninput: move |e| path.set(e.value()),
                placeholder: "/pinn",
            }
        }
    }
}

#[component]
fn QueryField(mut query: Signal<String>) -> Element {
    rsx! {
        div { class: "field",
            span { class: "field-label", "Query string" }
            input {
                value: "{query()}",
                oninput: move |e| query.set(e.value()),
                placeholder: "width=128&height=128",
            }
        }
    }
}

#[component]
fn JsonStatusBar(
    body: String,
    json_info: Option<(&'static str, usize)>,
    json_err: Option<(usize, usize, String)>,
) -> Element {
    rsx! {
        if let Some((line, col, _)) = json_err {
            div { class: "status-banner status-warn", style: "margin-top: 4px;",
                span { "⚠ JSON error at line {line}, column {col}" }
            }
        }
        div { class: "json-status",
            span { "lines: {body.lines().count()}" }
            span { " chars: {body.len()}" }
            if let Some((tl, n)) = json_info {
                span { "type: {tl}" }
                span { "{n} keys" }
            }
        }
    }
}

#[component]
fn RequestPanel(
    mut method: Signal<String>,
    mut path: Signal<String>,
    mut body: Signal<String>,
    mut query: Signal<String>,
    mut json_err: Signal<Option<(usize, usize, String)>>,
    mut error: Signal<Option<String>>,
    busy: bool,
    on_send: EventHandler<()>,
) -> Element {
    let json_info = if method() == "POST" {
        serde_json::from_str::<serde_json::Value>(&body())
            .ok()
            .map(|v| (json_type_label(&v), json_type_count(&v)))
    } else {
        None
    };

    rsx! {
        div { class: "card",
            div { class: "card-title", "Request" }
            div { class: "grid",
                MethodSelector { method }
                PathField { path }
                QueryField { query }
                if method() == "POST" {
                    RequestBodyEditor { body, json_err, error }
                    JsonStatusBar {
                        body: body(),
                        json_info,
                        json_err: json_err(),
                    }
                }
            }
            div { class: "toolbar", style: "margin-top: 14px;",
                button {
                    class: "btn btn-primary",
                    disabled: busy,
                    onclick: move |_| on_send.call(()),
                    if busy { span { class: "spinner" } }
                    span { "Send" }
                }
            }
        }
    }
}

#[component]
fn ResponsePanel(
    status_code: Option<u16>,
    error: Option<String>,
    raw_response: Option<String>,
    response_loaded: bool,
    mut raw_response_signal: Signal<Option<String>>,
    mut body: Signal<String>,
    mut json_err: Signal<Option<(usize, usize, String)>>,
    mut error_signal: Signal<Option<String>>,
) -> Element {
    rsx! {
        div { class: "card",
            div { class: "card-title",
                StatusDot { status: status_code.map(|c| if c < 400 { "ok" } else { "err" }.to_string()).unwrap_or_else(|| "off".to_string()) }
                span { "Response" }
                if let Some(c) = status_code {
                    Tag {
                        text: c.to_string(),
                        kind: if c < 300 { "ok".to_string() } else if c < 400 { "warn".to_string() } else { "err".to_string() },
                    }
                }
                if raw_response.is_some() {
                    button {
                        class: "btn btn-sm",
                        style: "margin-left: auto;",
                        onclick: move |_| {
                            if let Some(raw) = raw_response_signal() {
                                if let Ok(f) = fmt_json(&raw) {
                                    raw_response_signal.set(Some(f));
                                }
                            }
                        },
                        title: "Format JSON",
                        "{{}} ⟶"
                    }
                    button {
                        class: "btn btn-sm",
                        onclick: move |_| {
                            if let Some(raw) = raw_response_signal() {
                                if let Ok(m) = minify_json(&raw) {
                                    raw_response_signal.set(Some(m));
                                }
                            }
                        },
                        title: "Minify JSON",
                        "{{}} ⟵"
                    }
                    button {
                        class: "btn btn-sm",
                        onclick: move |_| {
                            if let Some(raw) = raw_response_signal() {
                                body.set(raw);
                                json_err.set(None);
                                error_signal.set(None);
                            }
                        },
                        title: "Copy to body",
                        "⬇ body"
                    }
                }
            }
            if let Some(e) = error {
                div { class: "status-banner status-err", "{e}" }
            }
            if let Some(raw) = raw_response {
                div { class: "code-block", "{raw}" }
            } else if !response_loaded {
                div { class: "empty", "No response yet" }
            }
        }
    }
}

const QUICK_TEMPLATES: &[(&str, &str, &str)] = &[
    ("POST", "/pinn", r#"{"x_pc":0,"y_pc":0,"z_pc":100,"bp_rp":1.5,"g_mag":10}"#),
    ("POST", "/gnn", r#"{"center_x":0,"center_y":0,"center_z":100,"bp_rp":1.5,"g_mag":10,"search_radius":25,"temperature":0.7}"#),
    ("POST", "/random_star", r#"{"entropy_temperature":0.5}"#),
    ("POST", "/siren/texture", r#"{"width":128,"height":128,"bp_rp":1.5,"m_g":5.0,"log_teff":3.76}"#),
    ("GET", "/siren/png", ""),
];

#[component]
fn TemplateBar(
    mut method: Signal<String>,
    mut path: Signal<String>,
    mut body: Signal<String>,
    mut json_err: Signal<Option<(usize, usize, String)>>,
    mut error: Signal<Option<String>>,
) -> Element {
    rsx! {
        div { class: "row",
            for t in QUICK_TEMPLATES.iter() {
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

#[component]
pub fn BackendApi() -> Element {
    let method = use_signal(|| "GET".to_string());
    let path = use_signal(|| "/".to_string());
    let body = use_signal(|| "{}".to_string());
    let query = use_signal(|| String::new());
    let mut response = use_signal(|| None::<serde_json::Value>);
    let mut status_code = use_signal(|| None::<u16>);
    let mut raw_response = use_signal(|| None::<String>);
    let mut error = use_signal(|| None::<String>);
    let mut busy = use_signal(|| false);
    let mut json_err = use_signal(|| None::<(usize, usize, String)>);

    let send = move |_: ()| {
        let prep = prepare_request(&method(), &path(), &body(), &query());
        apply_request_to_signals(
            &prep,
            &mut status_code,
            &mut response,
            &mut raw_response,
            &mut json_err,
            &mut error,
        );

        if prep.error.is_some() {
            busy.set(false);
            return;
        }

        busy.set(true);
        error.set(None);

        let req_path = prep.path.clone();
        let req_method = prep.method.clone();
        let req_query = prep.query.clone();
        let req_body = prep.body.clone();
        spawn(async move {
            match api::backend_proxy(&req_path, &req_method, req_body.as_ref(), req_query.as_deref()).await {
                Ok(v) => {
                    status_code.set(Some(200));
                    let pretty = serde_json::to_string_pretty(&v).unwrap_or_default();
                    raw_response.set(Some(pretty));
                    response.set(Some(v));
                }
                Err(msg) => {
                    if let Some(code) = extract_status_code_from_err(&msg) {
                        status_code.set(Some(code));
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
                RequestPanel {
                    method,
                    path,
                    body,
                    query,
                    json_err,
                    error,
                    busy: busy(),
                    on_send: send,
                }
                ResponsePanel {
                    status_code: status_code(),
                    error: error(),
                    raw_response: raw_response(),
                    response_loaded: response().is_some(),
                    raw_response_signal: raw_response,
                    body,
                    json_err,
                    error_signal: error,
                }
            }
            div { class: "section-title", "Quick templates" }
            TemplateBar {
                method,
                path,
                body,
                json_err,
                error,
            }
        }
    }
}
