use dioxus::prelude::*;

pub fn bytes_human(n: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{} {}", n, UNITS[0])
    } else {
        format!("{:.2} {}", v, UNITS[i])
    }
}

pub fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

pub fn fmt_ms(ms: i64) -> String {
    if ms < 0 {
        return "—".to_string();
    }
    let secs = ms / 1000;
    if secs < 60 {
        return format!("{secs}s");
    }
    let mins = secs / 60;
    let rem = secs % 60;
    if mins < 60 {
        return format!("{mins}m {rem:02}s");
    }
    let hrs = mins / 60;
    let rem_m = mins % 60;
    format!("{hrs}h {rem_m:02}m")
}

pub fn fmt_age(mtime_ms: u64) -> String {
    if mtime_ms == 0 {
        return "—".to_string();
    }
    let now = now_ms();
    if (now as u64) < mtime_ms {
        return "—".to_string();
    }
    fmt_ms(now - mtime_ms as i64)
}

#[component]
pub fn StatusDot(status: String) -> Element {
    let cls = match status.as_str() {
        "running" | "busy" => "led-dot led-busy",
        "completed" | "ok" | "ready" => "led-dot led-on",
        "failed" | "err" | "unreachable" => "led-dot led-err",
        _ => "led-dot led-off",
    };
    rsx! { span { class: "{cls}" } }
}

#[component]
pub fn PageHeader(title: String, subtitle: String) -> Element {
    rsx! {
        div { class: "page-header",
            div {
                div { class: "page-title", "{title}" }
                div { class: "page-subtitle", "{subtitle}" }
            }
        }
    }
}

#[component]
pub fn Tag(text: String, kind: String) -> Element {
    let cls = match kind.as_str() {
        "pinn" => "tag tag-pinn",
        "gnn" => "tag tag-gnn",
        "siren" => "tag tag-siren",
        "ok" => "tag tag-ok",
        "warn" => "tag tag-warn",
        "err" => "tag tag-err",
        _ => "tag tag-mute",
    };
    rsx! { span { class: "{cls}", "{text}" }
    }
}

#[component]
pub fn LossChart(
    metrics: Vec<lunar_structures_testbench::EpochMetric>,
    width: f64,
    height: f64,
) -> Element {
    if metrics.is_empty() {
        return rsx! {
            div { class: "empty", "No epoch data yet" }
        };
    }
    let train: Vec<f64> = metrics.iter().map(|m| m.train_loss).collect();
    let val: Vec<f64> = metrics.iter().map(|m| m.val_loss).collect();
    let phys: Vec<f64> = metrics.iter().map(|m| m.phys_loss.unwrap_or(0.0)).collect();

    let all_max = train
        .iter()
        .chain(val.iter())
        .chain(phys.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let all_min = train
        .iter()
        .chain(val.iter())
        .chain(phys.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let (y_min, y_max) = if all_max == all_min {
        (all_min - 0.5, all_max + 0.5)
    } else {
        let pad = (all_max - all_min) * 0.08;
        ((all_min - pad).max(0.0), all_max + pad)
    };

    let pad_l = 50.0_f64;
    let pad_r = 12.0_f64;
    let pad_t = 12.0_f64;
    let pad_b = 24.0_f64;
    let plot_w = (width - pad_l - pad_r).max(20.0);
    let plot_h = (height - pad_t - pad_b).max(20.0);

    let n = metrics.len().max(1);
    let x_step = if n > 1 { plot_w / (n - 1) as f64 } else { 0.0 };

    let to_xy = |i: usize, v: f64| -> (f64, f64) {
        let x = pad_l + (i as f64) * x_step;
        let y_norm = if y_max > y_min {
            (v - y_min) / (y_max - y_min)
        } else {
            0.5
        };
        let y = pad_t + plot_h * (1.0 - y_norm);
        (x, y)
    };

    let path_for = |vals: &[f64]| -> String {
        if vals.is_empty() {
            return String::new();
        }
        let mut s = String::new();
        for (i, v) in vals.iter().enumerate() {
            let (x, y) = to_xy(i, *v);
            if i == 0 {
                s.push_str(&format!("M{:.2},{:.2}", x, y));
            } else {
                s.push_str(&format!(" L{:.2},{:.2}", x, y));
            }
        }
        s
    };

    let p_train = path_for(&train);
    let p_val = path_for(&val);
    let p_phys = path_for(&phys);

    let last_train = train.last().copied().unwrap_or(0.0);
    let last_val = val.last().copied().unwrap_or(0.0);
    let last_phys = phys.last().copied().unwrap_or(0.0);
    let last_train_str = format!("{:.5}", last_train);
    let last_val_str = format!("{:.5}", last_val);
    let last_phys_str = format!("{:.5}", last_phys);

    let last_x_for = |vals: &[f64]| -> f64 {
        if vals.is_empty() {
            0.0
        } else {
            let idx = vals.len() - 1;
            pad_l + (idx as f64) * x_step
        }
    };
    let lt_x = last_x_for(&train);
    let lv_x = last_x_for(&val);
    let lp_x = last_x_for(&phys);
    let (lt_y, _) = to_xy(train.len() - 1, last_train);
    let (lv_y, _) = to_xy(val.len() - 1, last_val);
    let (lp_y, _) = to_xy(phys.len() - 1, last_phys);

    let y_ticks: Vec<f64> = (0..=4)
        .map(|i| y_min + (y_max - y_min) * (i as f64) / 4.0)
        .collect();
    let x_ticks: Vec<usize> = if n <= 6 {
        (0..n).collect()
    } else {
        (0..6).map(|i| i * (n - 1) / 5).collect()
    };

    rsx! {
        div { class: "card",
            div { class: "row", style: "justify-content: space-between; margin-bottom: 8px;",
                div { class: "row",
                    span { class: "led-dot led-on" }
                    span { class: "mono", style: "color: var(--accent);", "train" }
                    span { class: "mono", style: "color: var(--text-3); margin-left: 8px;", "{last_train_str}" }
                }
                div { class: "row",
                    span { class: "led-dot led-on" }
                    span { class: "mono", style: "color: var(--gnn);", "val" }
                    span { class: "mono", style: "color: var(--text-3); margin-left: 8px;", "{last_val_str}" }
                }
                div { class: "row",
                    span { class: "led-dot led-on" }
                    span { class: "mono", style: "color: var(--pinn);", "phys" }
                    span { class: "mono", style: "color: var(--text-3); margin-left: 8px;", "{last_phys_str}" }
                }
            }
            svg { width: "{width}", height: "{height}", view_box: "0 0 {width} {height}",
                for t in y_ticks.clone() {
                    {
                        let y_norm = if y_max > y_min { (t - y_min) / (y_max - y_min) } else { 0.5 };
                        let y = pad_t + plot_h * (1.0 - y_norm);
                        let t_str = format!("{:.4}", t);
                        rsx! {
                            line {
                                x1: "{pad_l}", y1: "{y}", x2: "{width - pad_r}", y2: "{y}",
                                stroke: "var(--border)", "stroke-width": "1", "stroke-dasharray": "2 4"
                            }
                            text {
                                x: "{pad_l - 6.0}", y: "{y + 3.0}",
                                fill: "var(--text-4)", "font-size": "9", "text-anchor": "end",
                                "font-family": "var(--mono)",
                                "{t_str}"
                            }
                        }
                    }
                }
                for idx in x_ticks.clone() {
                    {
                        let x = pad_l + (idx as f64) * x_step;
                        let epoch = metrics.get(idx).map(|m| m.epoch).unwrap_or(0);
                        rsx! {
                            line {
                                x1: "{x}", y1: "{pad_t}", x2: "{x}", y2: "{pad_t + plot_h}",
                                stroke: "var(--border)", "stroke-width": "1", "stroke-dasharray": "2 4"
                            }
                            text {
                                x: "{x}", y: "{height - 6.0}",
                                fill: "var(--text-4)", "font-size": "9", "text-anchor": "middle",
                                "font-family": "var(--mono)",
                                "{epoch}"
                            }
                        }
                    }
                }
                path { d: "{p_train}", fill: "none", stroke: "var(--accent)", "stroke-width": "1.5" }
                path { d: "{p_val}", fill: "none", stroke: "var(--gnn)", "stroke-width": "1.5" }
                path { d: "{p_phys}", fill: "none", stroke: "var(--pinn)", "stroke-width": "1.5", "stroke-dasharray": "3 3" }
                circle { cx: "{lt_x}", cy: "{lt_y}", r: "3", fill: "var(--accent)" }
                circle { cx: "{lv_x}", cy: "{lv_y}", r: "3", fill: "var(--gnn)" }
                circle { cx: "{lp_x}", cy: "{lp_y}", r: "3", fill: "var(--pinn)" }
            }
        }
    }
}

#[component]
pub fn ProgressBar(value: f32, kind: String) -> Element {
    let cls = format!("progress-bar {kind}");
    let pct = (value * 100.0).clamp(0.0, 100.0) as u32;
    rsx! {
        div { class: "progress-track",
            div { class: "{cls}", style: "width: {pct}%" }
        }
    }
}

pub fn base64_encode(bytes: &[u8]) -> String {
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let b0 = bytes[i];
        let b1 = bytes[i + 1];
        let b2 = bytes[i + 2];
        out.push(CHARSET[(b0 >> 2) as usize] as char);
        out.push(CHARSET[((b0 & 0x03) << 4 | b1 >> 4) as usize] as char);
        out.push(CHARSET[((b1 & 0x0F) << 2 | b2 >> 6) as usize] as char);
        out.push(CHARSET[(b2 & 0x3F) as usize] as char);
        i += 3;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        let b0 = bytes[i];
        out.push(CHARSET[(b0 >> 2) as usize] as char);
        out.push(CHARSET[((b0 & 0x03) << 4) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let b0 = bytes[i];
        let b1 = bytes[i + 1];
        out.push(CHARSET[(b0 >> 2) as usize] as char);
        out.push(CHARSET[((b0 & 0x03) << 4 | b1 >> 4) as usize] as char);
        out.push(CHARSET[((b1 & 0x0F) << 2) as usize] as char);
        out.push('=');
    }
    out
}

#[component]
pub fn NumberFieldU32(label: String, value: Signal<u32>) -> Element {
    rsx! {
        div { class: "field",
            span { class: "field-label", "{label}" }
            input {
                r#type: "number",
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

#[component]
pub fn NumberFieldF64(label: String, value: Signal<f64>, step: f64) -> Element {
    rsx! {
        div { class: "field",
            span { class: "field-label", "{label}" }
            input {
                r#type: "number",
                step: "{step}",
                value: "{value()}",
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<f64>() {
                        value.set(v);
                    }
                },
            }
        }
    }
}

#[component]
pub fn TextField(label: String, value: Signal<String>) -> Element {
    rsx! {
        div { class: "field",
            span { class: "field-label", "{label}" }
            input {
                value: "{value()}",
                oninput: move |e| value.set(e.value()),
            }
        }
    }
}

pub async fn tokio_time_sleep(ms: u32) {
    use gloo_timers::future::TimeoutFuture;
    TimeoutFuture::new(ms).await;
}
