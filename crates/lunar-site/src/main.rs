use dioxus::prelude::*;
use gloo_timers::callback::Timeout;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::{Clamped, JsCast};
use web_sys::{
    CanvasRenderingContext2d, HtmlCanvasElement, ImageData, MouseEvent, TouchEvent, Window,
};

const MAIN_CSS: Asset = asset!("/assets/main.css");

const INSTALL_CMD: &str =
    "curl https://raw.githubusercontent.com/INiNiDS/lunar-hal/refs/heads/main/install.sh | sh";
const GH_URL: &str = "https://github.com/ininids/lunar-hal";

const PIXEL_SIZE: f64 = 4.0;
const BAYER: [[u8; 4]; 4] = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
];
const FG: [u8; 3] = [200, 220, 255];
const BG: [u8; 3] = [6, 8, 12];

const SPEED_PHASE: f64 = 20.0;
const FADE_DURATION: f64 = 50.0;

fn main() {
    launch(App);
}

#[component]
fn App() -> Element {
    let mut copied = use_signal(|| false);
    let mut active = use_signal(|| false);

    use_effect(move || {
        start_dither();
    });

    use_effect(move || {
        bind_mouse_active(active);
    });

    let on_click = move |_| {
        write_clipboard(INSTALL_CMD);
        copied.set(true);
        Timeout::new(1500, move || copied.set(false)).forget();
        open_url(GH_URL);
    };

    let btn_class = if active() {
        "install-btn active"
    } else {
        "install-btn"
    };

    rsx! {
        div {
            id: "app-root",
            onmousemove: move |_| {
                if !active() { active.set(true); }
            },
            ontouchstart: move |_| {
                if !active() { active.set(true); }
            },

            document::Link { rel: "stylesheet", href: MAIN_CSS }
            canvas { id: "dither-canvas" }
            div { id: "lunar-hal-text", "LUNAR-HAL" }
            div { id: "overlay",
                div { id: "center-content",
                    button {
                        id: "install-btn",
                        class: "{btn_class}",
                        r#type: "button",
                        onclick: on_click,
                        span { class: "prompt", "$ " }
                        code { class: "install-cmd", "{INSTALL_CMD}" }
                        span { class: "cursor" }
                        if copied() {
                            span { class: "copied-tag", "copied" }
                        }
                    }
                }
            }
        }
    }
}

fn write_clipboard(text: &str) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let navigator = window.navigator();
    if let Ok(clip) = js_sys::Reflect::get(&navigator, &JsValue::from_str("clipboard")) {
        if !clip.is_undefined() && !clip.is_null() {
            let _ = navigator.clipboard().write_text(text);
        }
    }
}

fn open_url(url: &str) {
    if let Some(window) = web_sys::window() {
        let _ = window.open_with_url_and_target(url, "_blank");
    }
}

type FrameCallback = Rc<RefCell<Option<Closure<dyn FnMut()>>>>;

struct DitherState {
    mouse_x: f64,
    mouse_y: f64,
    time: f64,
    real_time: f64,
    buffer: Vec<u8>,
}

fn bind_mouse_active(mut active: Signal<bool>) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(document) = window.document() else {
        return;
    };

    let cb = Closure::<dyn FnMut(MouseEvent)>::new(move |_: MouseEvent| {
        if !*active.peek() {
            active.set(true);
        }
    });
    let _ = document.add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref());
    cb.forget();

    let cb_t = Closure::<dyn FnMut(TouchEvent)>::new(move |_: TouchEvent| {
        if !*active.peek() {
            active.set(true);
        }
    });
    let _ = document.add_event_listener_with_callback(
        "touchstart",
        cb_t.as_ref().unchecked_ref(),
    );
    cb_t.forget();
}

fn start_dither() {
    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(document) = window.document() else {
        return;
    };

    let Some(canvas) = document
        .get_element_by_id("dither-canvas")
        .and_then(|el| el.dyn_into::<HtmlCanvasElement>().ok())
    else {
        return;
    };

    let Some(ctx) = canvas
        .get_context("2d")
        .ok()
        .flatten()
        .and_then(|o| o.dyn_into::<CanvasRenderingContext2d>().ok())
    else {
        return;
    };

    let state = Rc::new(RefCell::new(DitherState {
        mouse_x: -9999.0,
        mouse_y: -9999.0,
        time: 0.0,
        real_time: 0.0,
        buffer: Vec::new(),
    }));

    resize_canvas(&canvas, &window);

    {
        let canvas = canvas.clone();
        let win = window.clone();
        let cb = Closure::<dyn Fn()>::new(move || {
            resize_canvas(&canvas, &win);
        });
        let _ = window.add_event_listener_with_callback("resize", cb.as_ref().unchecked_ref());
        cb.forget();
    }

    {
        let state = state.clone();
        let cb = Closure::<dyn Fn(MouseEvent)>::new(move |e: MouseEvent| {
            let mut st = state.borrow_mut();
            st.mouse_x = e.client_x() as f64;
            st.mouse_y = e.client_y() as f64;
        });
        let _ = document.add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref());
        cb.forget();
    }

    {
        let state = state.clone();
        let cb = Closure::<dyn Fn(TouchEvent)>::new(move |e: TouchEvent| {
            if let Some(t) = e.touches().get(0) {
                let mut st = state.borrow_mut();
                st.mouse_x = t.client_x() as f64;
                st.mouse_y = t.client_y() as f64;
            }
        });
        let _ = document.add_event_listener_with_callback("touchmove", cb.as_ref().unchecked_ref());
        cb.forget();
    }

    let f: FrameCallback = Rc::new(RefCell::new(None));
    let g = f.clone();
    let canvas_loop = canvas.clone();
    let ctx_loop = ctx.clone();
    let state_loop = state.clone();
    let last_time = Rc::new(RefCell::new(0.0_f64));

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let now = js_sys::Date::now() / 1000.0;
        let prev = *last_time.borrow();
        let dt = if prev == 0.0 { 0.016 } else { (now - prev).min(0.1) };
        *last_time.borrow_mut() = now;

        {
            let mut st = state_loop.borrow_mut();
            st.real_time += dt;
            let speed = if st.real_time < SPEED_PHASE {
                1.0 + 0.75 * (st.real_time * std::f64::consts::PI / 3.5).sin()
            } else {
                1.0
            };
            st.time += dt * speed;
        }
        draw(&canvas_loop, &ctx_loop, &state_loop);

        if let Some(w) = web_sys::window() {
            if let Some(cb) = f.borrow().as_ref() {
                let _ = w.request_animation_frame(cb.as_ref().unchecked_ref());
            }
        }
    }) as Box<dyn FnMut()>));

    let borrow = g.borrow();
    if let Some(cb) = borrow.as_ref() {
        let _ = window.request_animation_frame(cb.as_ref().unchecked_ref());
    }
    drop(borrow);
}

fn resize_canvas(canvas: &HtmlCanvasElement, window: &Window) {
    let w = window
        .inner_width()
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let h = window
        .inner_height()
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    canvas.set_width((w / PIXEL_SIZE).ceil().max(1.0) as u32);
    canvas.set_height((h / PIXEL_SIZE).ceil().max(1.0) as u32);
}
fn draw(
    canvas: &HtmlCanvasElement,
    ctx: &CanvasRenderingContext2d,
    state: &Rc<RefCell<DitherState>>,
) {
    let (mx, my, t, rt) = {
        let st = state.borrow();
        (
            st.mouse_x / PIXEL_SIZE,
            st.mouse_y / PIXEL_SIZE,
            st.time,
            st.real_time,
        )
    };

    let w = canvas.width() as usize;
    let h = canvas.height() as usize;
    if w == 0 || h == 0 {
        return;
    }

    let fade_raw = (rt / FADE_DURATION).clamp(0.0, 1.0);
    let fade = fade_raw * fade_raw * (3.0 - 2.0 * fade_raw);

    if fade >= 0.999 {
        ctx.set_fill_style_str("#06080c");
        ctx.fill_rect(0.0, 0.0, w as f64, h as f64);
        return;
    }

    let needed = w * h * 4;
    {
        let mut st = state.borrow_mut();
        if st.buffer.len() != needed {
            st.buffer.resize(needed, 0);
        }
    }

    let mut st = state.borrow_mut();
    let data: &mut [u8] = &mut st.buffer;
    let fade_inv = 1.0 - fade;

    for y in 0..h {
        let row_off = y * w * 4;
        let by = y & 3;
        let fy = y as f64;
        for x in 0..w {
            let fx = x as f64;

            let s1 = (fx * 0.05 + t * 0.6).sin();
            let s2 = (fy * 0.04 - t * 0.5).sin();
            let s3 = ((fx + fy) * 0.035 + t * 0.4).cos();
            let s4 = ((fx - fy) * 0.025 + t * 0.3).sin();
            let s5 = (fx * 0.17 - fy * 0.13 + t * 1.1).sin();

            let cx = (fx * 0.3).floor();
            let cy = (fy * 0.3).floor();
            let phase =
                ((cx * 12.9898 + cy * 78.233).sin() * 43758.5453).fract() * std::f64::consts::TAU;
            let flicker = (t * 2.4 + phase).sin() * 0.18;

            let mut value = (s1 + s2 + s3 + s4 + s5) * 0.26 + 0.5 + flicker;
            value = value.clamp(0.0, 1.0);

            if mx > -1000.0 {
                let dx = fx - mx;
                let dy = fy - my;
                let dist = (dx * dx + dy * dy).sqrt();
                let boost = (1.0 - dist / 60.0).clamp(0.0, 1.0) * 0.35;
                value = (value + boost).clamp(0.0, 1.0);
            }

            let threshold = BAYER[by][x & 3] as f64 / 15.0;
            let color = if value > threshold { FG } else { BG };

            let idx = row_off + x * 4;
            data[idx] = (color[0] as f64 * fade_inv + BG[0] as f64 * fade) as u8;
            data[idx + 1] = (color[1] as f64 * fade_inv + BG[1] as f64 * fade) as u8;
            data[idx + 2] = (color[2] as f64 * fade_inv + BG[2] as f64 * fade) as u8;
            data[idx + 3] = 255;
        }
    }

    let data_imm: &[u8] = data;
    if let Ok(image_data) =
        ImageData::new_with_u8_clamped_array_and_sh(Clamped(data_imm), w as u32, h as u32)
    {
        let _ = ctx.put_image_data(&image_data, 0.0, 0.0);
    }
}
