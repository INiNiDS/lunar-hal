use dioxus::prelude::*;
#[cfg(all(feature = "web", target_family = "wasm"))]
use std::cell::RefCell;
#[cfg(all(feature = "web", target_family = "wasm"))]
use std::rc::Rc;

#[cfg(all(feature = "web", target_family = "wasm"))]
use wasm_bindgen::{JsCast, JsValue, prelude::Closure};
#[cfg(all(feature = "web", target_family = "wasm"))]
use web_sys::WebGl2RenderingContext as Gl2;

#[cfg(all(feature = "web", target_family = "wasm"))]
macro_rules! gl_err {
    ($($arg:tt)*) => {
        web_sys::console::error_1(&JsValue::from(format!($($arg)*)))
    };
}

const VERT_SRC: &str = r#"#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
    vUv = aPos;
    gl_Position = vec4(aPos, 0.0, 1.0);
}"#;

// Preview shaders must never allocate an unbounded backing store on a
// fractional/high-DPI browser layout. This upper bound is about 8 MiB of RGBA
// pixels at the largest supported preview size.
#[cfg(all(feature = "web", target_family = "wasm"))]
const MAX_BACKING_DIMENSION: f64 = 2_048.0;
#[cfg(all(feature = "web", target_family = "wasm"))]
const MAX_BACKING_PIXELS: f64 = 2_097_152.0;
#[cfg(all(feature = "web", target_family = "wasm"))]
const PREVIEW_FRAME_INTERVAL_MS: f64 = 1_000.0 / 30.0;

const FRAG_SRC: &str = r#"#version 300 es
precision highp float;
in vec2 vUv;
uniform float uTime;
uniform float uTeff;
uniform float uBpRp;
uniform float uScale;
uniform float uSpeed;
uniform float uContrast;
out vec4 fragColor;

float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float noise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(i+vec3(0,0,0)),hash(i+vec3(1,0,0)),f.x),mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x),f.y),mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x),f.y),f.z);
}
float fbm(vec3 p) {
    float v = 0.0, a = 0.5, f = 1.0;
    for (int i = 0; i < 5; i++) { v += a * noise3D(p * f); f *= 2.0; a *= 0.5; }
    return v;
}
vec3 starBaseColor(float teff, float bp_rp) {
    vec3 c;
    if (teff > 30000.0) { c = vec3(0.62, 0.69, 1.0); }
    else if (teff > 10000.0) { float f = (teff-10000.0)/20000.0; c = vec3(0.70-0.08*f,0.77-0.08*f,0.95+0.05*f); }
    else if (teff > 7500.0) { float f = (teff-7500.0)/2500.0; c = vec3(0.82-0.12*f,0.85-0.08*f,0.95); }
    else if (teff > 6000.0) { float f = (teff-6000.0)/1500.0; c = vec3(0.95-0.13*f,0.93-0.08*f,0.90+0.05*f); }
    else if (teff > 5200.0) { float f = (teff-5200.0)/800.0; c = vec3(1.0,1.0-0.07*f,0.82+0.08*f); }
    else if (teff > 3700.0) { float f = (teff-3700.0)/1500.0; c = vec3(1.0,0.85+0.08*f,0.65+0.25*f); }
    else { c = vec3(1.0, 0.55, 0.35); }
    float t = (bp_rp-0.5)/4.0;
    return clamp(vec3(c.r-t*0.05,c.g-t*0.03,c.b+t*0.05),0.0,1.0);
}
void main() {
    vec2 uv = vUv;
    float r = length(uv);
    float disk = smoothstep(1.005, 0.992, r);
    if (disk < 0.001) { fragColor = vec4(0.0); return; }
    float mu = sqrt(max(1.0-r*r,0.0));
    float limb = 1.0-0.6*(1.0-mu);
    vec3 base = starBaseColor(uTeff, uBpRp);
    float n = fbm(vec3(uv*uScale,uTime*uSpeed));
    float n2 = fbm(vec3(uv*uScale*1.7,uTime*uSpeed+100.0));
    float plasma = (n-0.5)*uContrast*2.0;
    vec3 color = base*(1.0+plasma)+plasma*0.3*vec3(0.2,-0.1,-0.3);
    color *= limb;
    float gran = fbm(vec3(uv*uScale*3.0,uTime*uSpeed+50.0))-0.5;
    color += base*gran*0.04;
    float glow = exp(-(1.0-r)*12.0)*0.08;
    float corona = smoothstep(0.85,1.05,r)*exp(-(r-0.85)*8.0)*0.15;
    vec3 coronaColor = mix(base,vec3(1.0,0.7,0.3),0.5);
    color += glow*base*(0.5+0.5*n2)+corona*coronaColor*(0.5+0.5*n);
    float hot = smoothstep(0.55,0.75,n)*0.08; color += hot*base*1.5;
    float cold = smoothstep(0.65,0.45,n)*0.06; color -= cold*base*0.5;
    fragColor = vec4(max(color*disk,vec3(0.0)),1.0);
}"#;

/// JS bootstrap used on the desktop build, where the Rust side has no
/// `web-sys` and therefore has to drive the canvas from inside the
/// WebView via `document::eval`.
#[cfg(not(all(feature = "web", target_family = "wasm")))]
const DESKTOP_JS: &str = r#"
(function() {
    const canvas = document.getElementById('__ID__');
    if (!canvas) return;
    if (canvas.__starReady) return;
    canvas.__starReady = true;
    const gl = canvas.getContext('webgl2');
    if (!gl) { console.error('StarShader: WebGL2 not supported'); return; }

    const compile = (type, src) => {
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
            console.error('StarShader shader error:', gl.getShaderInfoLog(sh));
        }
        return sh;
    };
    const vs = compile(gl.VERTEX_SHADER, `__VERT__`);
    const fs = compile(gl.FRAGMENT_SHADER, `__FRAG__`);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        console.error('StarShader link error:', gl.getProgramInfoLog(prog));
        return;
    }
    gl.useProgram(prog);

    const verts = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    const aPos = gl.getAttribLocation(prog, 'aPos');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    const uTime = gl.getUniformLocation(prog, 'uTime');
    const uTeff = gl.getUniformLocation(prog, 'uTeff');
    const uBpRp = gl.getUniformLocation(prog, 'uBpRp');
    const uScale = gl.getUniformLocation(prog, 'uScale');
    const uSpeed = gl.getUniformLocation(prog, 'uSpeed');
    const uContrast = gl.getUniformLocation(prog, 'uContrast');

    const state = { teff: __TEFF__, bp_rp: __BP_RP__, scale: __SCALE__, speed: __SPEED__, contrast: __CONTRAST__, t: 0 };
    canvas.__starUpdate = (teff, bp_rp, scale, speed, contrast) => {
        state.teff = teff; state.bp_rp = bp_rp;
        state.scale = scale; state.speed = speed; state.contrast = contrast;
    };

    const maxDimension = 2048;
    const maxPixels = 2097152;
    const resize = () => {
        const rect = canvas.getBoundingClientRect();
        const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
        let width = Math.max(1, Math.min(maxDimension, Math.round(rect.width * dpr)));
        let height = Math.max(1, Math.min(maxDimension, Math.round(rect.height * dpr)));
        const pixels = width * height;
        if (pixels > maxPixels) {
            const scale = Math.sqrt(maxPixels / pixels);
            width = Math.max(1, Math.floor(width * scale));
            height = Math.max(1, Math.floor(height * scale));
        }
        if (canvas.width !== width) canvas.width = width;
        if (canvas.height !== height) canvas.height = height;
        gl.viewport(0, 0, width, height);
    };
    const resizeObserver = typeof ResizeObserver === 'undefined'
        ? null
        : new ResizeObserver(resize);
    if (resizeObserver) resizeObserver.observe(canvas);
    else window.addEventListener('resize', resize);
    resize();
    canvas.__starResizeCleanup = () => {
        if (resizeObserver) resizeObserver.disconnect();
        else window.removeEventListener('resize', resize);
        if (canvas.__starRaf) cancelAnimationFrame(canvas.__starRaf);
    };
    gl.clearColor(0.02, 0.02, 0.04, 1.0);

    let lastFrame = 0;
    const tick = (now) => {
        if (!canvas.isConnected) {
            canvas.__starResizeCleanup();
            return;
        }
        if (!document.hidden && now - lastFrame >= 1000 / 30) {
            const elapsed = lastFrame ? Math.min(0.1, (now - lastFrame) / 1000) : 1 / 30;
            lastFrame = now;
            state.t += elapsed;
            gl.uniform1f(uTime, state.t);
            gl.uniform1f(uTeff, state.teff);
            gl.uniform1f(uBpRp, state.bp_rp);
            gl.uniform1f(uScale, state.scale);
            gl.uniform1f(uSpeed, state.speed);
            gl.uniform1f(uContrast, state.contrast);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        }
        canvas.__starRaf = requestAnimationFrame(tick);
    };
    canvas.__starRaf = requestAnimationFrame(tick);
})();
"#;

#[cfg(not(all(feature = "web", target_family = "wasm")))]
fn escape_for_template(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('`', "\\`")
        .replace("${", "\\${")
}

#[cfg(not(all(feature = "web", target_family = "wasm")))]
fn build_desktop_js(
    id: &str,
    teff: f32,
    bp_rp: f32,
    scale: f32,
    speed: f32,
    contrast: f32,
) -> String {
    let v = escape_for_template(VERT_SRC);
    let f = escape_for_template(FRAG_SRC);
    DESKTOP_JS
        .replace("__ID__", id)
        .replace("__VERT__", &v)
        .replace("__FRAG__", &f)
        .replace("__TEFF__", &format!("{teff}"))
        .replace("__BP_RP__", &format!("{bp_rp}"))
        .replace("__SCALE__", &format!("{scale}"))
        .replace("__SPEED__", &format!("{speed}"))
        .replace("__CONTRAST__", &format!("{contrast}"))
}

#[cfg(all(feature = "web", target_family = "wasm"))]
struct StarProps {
    teff: f32,
    bp_rp: f32,
    scale: f32,
    speed: f32,
    contrast: f32,
}

#[cfg(all(feature = "web", target_family = "wasm"))]
struct GlState {
    _slot: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>,
    _resize_observer: web_sys::ResizeObserver,
    _resize_callback: Closure<dyn FnMut()>,
    _ctx: Gl2,
    _u_teff: web_sys::WebGlUniformLocation,
    _u_bp_rp: web_sys::WebGlUniformLocation,
    _u_scale: web_sys::WebGlUniformLocation,
    _u_speed: web_sys::WebGlUniformLocation,
    _u_contrast: web_sys::WebGlUniformLocation,
}

#[cfg(all(feature = "web", target_family = "wasm"))]
thread_local! {
    static GL_STATE: RefCell<Option<GlState>> = const { RefCell::new(None) };
}

#[cfg(all(feature = "web", target_family = "wasm"))]
fn resize_canvas_backing(canvas: &web_sys::HtmlCanvasElement, ctx: &Gl2) {
    let rect = canvas.get_bounding_client_rect();
    let dpr = web_sys::window()
        .map(|window| window.device_pixel_ratio().clamp(1.0, 2.0))
        .unwrap_or(1.0);
    let mut width = (rect.width() * dpr).round().clamp(1.0, MAX_BACKING_DIMENSION);
    let mut height = (rect.height() * dpr).round().clamp(1.0, MAX_BACKING_DIMENSION);
    let pixels = width * height;
    if pixels > MAX_BACKING_PIXELS {
        let scale = (MAX_BACKING_PIXELS / pixels).sqrt();
        width = (width * scale).floor().max(1.0);
        height = (height * scale).floor().max(1.0);
    }
    let width = width as u32;
    let height = height as u32;

    if canvas.width() != width {
        canvas.set_width(width);
    }
    if canvas.height() != height {
        canvas.set_height(height);
    }
    ctx.viewport(0, 0, width as i32, height as i32);
}

#[component]
pub fn StarShaderCanvas(
    width: u32,
    height: u32,
    teff: f64,
    bp_rp: f64,
    noise_scale: f64,
    noise_speed: f64,
    contrast: f64,
) -> Element {
    let canvas_id = use_signal(|| {
        #[cfg(all(feature = "web", target_family = "wasm"))]
        {
            format!("star-gl-{:016x}", js_sys::Math::random().to_bits())
        }
        #[cfg(not(all(feature = "web", target_family = "wasm")))]
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("star-gl-{n}")
        }
    });

    let teff_f = teff as f32;
    let bp_rp_f = bp_rp as f32;
    let scale_f = noise_scale as f32;
    let speed_f = noise_speed as f32;
    let contrast_f = contrast as f32;

    #[cfg(all(feature = "web", target_family = "wasm"))]
    let props: Signal<Rc<RefCell<StarProps>>> = use_signal(|| {
        Rc::new(RefCell::new(StarProps {
            teff: teff_f,
            bp_rp: bp_rp_f,
            scale: scale_f,
            speed: speed_f,
            contrast: contrast_f,
        }))
    });

    #[cfg(all(feature = "web", target_family = "wasm"))]
    {
        let p = &*props.read();
        let mut mp = p.borrow_mut();
        mp.teff = teff_f;
        mp.bp_rp = bp_rp_f;
        mp.scale = scale_f;
        mp.speed = speed_f;
        mp.contrast = contrast_f;
    }

    #[cfg(all(feature = "web", target_family = "wasm"))]
    {
        let id = canvas_id.read().clone();
        let p2 = props.clone();
        use_effect(move || {
            let window = match web_sys::window() {
                Some(w) => w,
                None => {
                    gl_err!("StarShader: no window");
                    return;
                }
            };
            let doc = match window.document() {
                Some(d) => d,
                None => {
                    gl_err!("StarShader: no document");
                    return;
                }
            };
            let canvas = match doc.get_element_by_id(&id) {
                Some(el) => match el.dyn_into::<web_sys::HtmlCanvasElement>() {
                    Ok(c) => c,
                    Err(_) => {
                        gl_err!("StarShader: not a canvas");
                        return;
                    }
                },
                None => {
                    gl_err!("StarShader: canvas not found");
                    return;
                }
            };
            let ctx = match canvas.get_context("webgl2") {
                Ok(Some(c)) => match c.dyn_into::<Gl2>() {
                    Ok(gl) => gl,
                    Err(_) => {
                        gl_err!("StarShader: ctx");
                        return;
                    }
                },
                Ok(None) => {
                    gl_err!("StarShader: WebGL2 unsupported");
                    return;
                }
                Err(e) => {
                    gl_err!("StarShader: get_context {:?}", e);
                    return;
                }
            };

            let vs = match ctx.create_shader(Gl2::VERTEX_SHADER) {
                Some(s) => s,
                None => {
                    gl_err!("StarShader: vs create");
                    return;
                }
            };
            ctx.shader_source(&vs, VERT_SRC);
            ctx.compile_shader(&vs);
            if !ctx
                .get_shader_parameter(&vs, Gl2::COMPILE_STATUS)
                .as_bool()
                .unwrap_or(false)
            {
                gl_err!(
                    "StarShader: vs compile {}",
                    ctx.get_shader_info_log(&vs).unwrap_or_default()
                );
                return;
            }
            let fs = match ctx.create_shader(Gl2::FRAGMENT_SHADER) {
                Some(s) => s,
                None => {
                    gl_err!("StarShader: fs create");
                    return;
                }
            };
            ctx.shader_source(&fs, FRAG_SRC);
            ctx.compile_shader(&fs);
            if !ctx
                .get_shader_parameter(&fs, Gl2::COMPILE_STATUS)
                .as_bool()
                .unwrap_or(false)
            {
                gl_err!(
                    "StarShader: fs compile {}",
                    ctx.get_shader_info_log(&fs).unwrap_or_default()
                );
                return;
            }
            let prog = match ctx.create_program() {
                Some(p) => p,
                None => {
                    gl_err!("StarShader: prog");
                    return;
                }
            };
            ctx.attach_shader(&prog, &vs);
            ctx.attach_shader(&prog, &fs);
            ctx.link_program(&prog);
            if !ctx
                .get_program_parameter(&prog, Gl2::LINK_STATUS)
                .as_bool()
                .unwrap_or(false)
            {
                gl_err!(
                    "StarShader: link {}",
                    ctx.get_program_info_log(&prog).unwrap_or_default()
                );
                return;
            }
            ctx.use_program(Some(&prog));

            let verts: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
            let buf = match ctx.create_buffer() {
                Some(b) => b,
                None => {
                    gl_err!("StarShader: buf");
                    return;
                }
            };
            ctx.bind_buffer(Gl2::ARRAY_BUFFER, Some(&buf));
            ctx.buffer_data_with_array_buffer_view(
                Gl2::ARRAY_BUFFER,
                unsafe { js_sys::Float32Array::view(&verts) }.as_ref(),
                Gl2::STATIC_DRAW,
            );
            let a_pos = ctx.get_attrib_location(&prog, "aPos") as u32;
            ctx.enable_vertex_attrib_array(a_pos);
            ctx.vertex_attrib_pointer_with_i32(a_pos, 2, Gl2::FLOAT, false, 0, 0);

            resize_canvas_backing(&canvas, &ctx);
            let resize_canvas = canvas.clone();
            let resize_context = ctx.clone();
            let resize_callback = Closure::<dyn FnMut()>::new(move || {
                resize_canvas_backing(&resize_canvas, &resize_context);
            });
            let resize_observer = match web_sys::ResizeObserver::new(
                resize_callback.as_ref().unchecked_ref(),
            ) {
                Ok(observer) => observer,
                Err(error) => {
                    gl_err!("StarShader: ResizeObserver {:?}", error);
                    return;
                }
            };
            resize_observer.observe(canvas.as_ref());
            ctx.clear_color(0.02, 0.02, 0.04, 1.0);

            let (
                Some(u_time),
                Some(u_teff),
                Some(u_bp_rp),
                Some(u_scale),
                Some(u_speed),
                Some(u_contrast),
            ) = (
                ctx.get_uniform_location(&prog, "uTime"),
                ctx.get_uniform_location(&prog, "uTeff"),
                ctx.get_uniform_location(&prog, "uBpRp"),
                ctx.get_uniform_location(&prog, "uScale"),
                ctx.get_uniform_location(&prog, "uSpeed"),
                ctx.get_uniform_location(&prog, "uContrast"),
            )
            else {
                gl_err!("StarShader: uniforms");
                return;
            };

            let slot: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
            let slot2 = slot.clone();
            let w2 = window.clone();
            let ctx2 = ctx.clone();
            let p3 = p2.clone();
            let document_for_frame = doc.clone();
            let canvas_id_for_frame = id.clone();
            let mut time = 0.0f32;
            let mut last_frame = 0.0f64;

            let u_teff_c = u_teff.clone();
            let u_bp_rp_c = u_bp_rp.clone();
            let u_scale_c = u_scale.clone();
            let u_speed_c = u_speed.clone();
            let u_contrast_c = u_contrast.clone();
            *slot.borrow_mut() = Some(Closure::new(move |timestamp: f64| {
                // Stop the self-scheduling loop when Dioxus has removed this
                // component. This is essential for browser tabs and minimized
                // WebOS windows, where an orphaned canvas otherwise burns CPU.
                if document_for_frame
                    .get_element_by_id(&canvas_id_for_frame)
                    .is_none()
                {
                    return;
                }

                if !document_for_frame.hidden()
                    && timestamp - last_frame >= PREVIEW_FRAME_INTERVAL_MS
                {
                    let elapsed = if last_frame > 0.0 {
                        ((timestamp - last_frame) / 1_000.0).min(0.1)
                    } else {
                        1.0 / 30.0
                    };
                    last_frame = timestamp;
                    time += elapsed as f32;
                    if let Ok(pr) = p3.try_read() {
                        let sp = pr.borrow();
                        ctx2.uniform1f(Some(&u_teff_c), sp.teff);
                        ctx2.uniform1f(Some(&u_bp_rp_c), sp.bp_rp);
                        ctx2.uniform1f(Some(&u_scale_c), sp.scale);
                        ctx2.uniform1f(Some(&u_speed_c), sp.speed);
                        ctx2.uniform1f(Some(&u_contrast_c), sp.contrast);
                        ctx2.uniform1f(Some(&u_time), time);
                        ctx2.clear(Gl2::COLOR_BUFFER_BIT);
                        ctx2.draw_arrays(Gl2::TRIANGLE_STRIP, 0, 4);
                    }
                }
                if let Some(c) = &*slot2.borrow() {
                    let f: &js_sys::Function = c.as_ref().unchecked_ref();
                    let _ = w2.request_animation_frame(f);
                }
            }));

            {
                let guard = slot.borrow();
                if let Some(c) = &*guard {
                    let f: &js_sys::Function = c.as_ref().unchecked_ref();
                    let _ = window.request_animation_frame(f);
                }
            }

            GL_STATE.with(|s| {
                *s.borrow_mut() = Some(GlState {
                    _slot: slot,
                    _resize_observer: resize_observer,
                    _resize_callback: resize_callback,
                    _ctx: ctx,
                    _u_teff: u_teff,
                    _u_bp_rp: u_bp_rp,
                    _u_scale: u_scale,
                    _u_speed: u_speed,
                    _u_contrast: u_contrast,
                });
            });
        });
    }

    #[cfg(not(all(feature = "web", target_family = "wasm")))]
    {
        let id = canvas_id.read().clone();
        use_effect(move || {
            let js = build_desktop_js(&id, teff_f, bp_rp_f, scale_f, speed_f, contrast_f);
            let _ = dioxus::document::eval(&js);
        });
        let teff_v = teff_f;
        let bp_rp_v = bp_rp_f;
        let scale_v = scale_f;
        let speed_v = speed_f;
        let contrast_v = contrast_f;
        let id_for_update = canvas_id.read().clone();
        use_effect(move || {
            let teff_v = teff_v;
            let bp_rp_v = bp_rp_v;
            let scale_v = scale_v;
            let speed_v = speed_v;
            let contrast_v = contrast_v;
            let id = id_for_update.clone();
            let _ = teff_v;
            let _ = bp_rp_v;
            let _ = scale_v;
            let _ = speed_v;
            let _ = contrast_v;
            let js = format!(
                "(function(){{const c=document.getElementById('{}');if(c&&c.__starUpdate){{c.__starUpdate({},{},{},{},{});}}}})();",
                id, teff_v, bp_rp_v, scale_v, speed_v, contrast_v
            );
            let _ = dioxus::document::eval(&js);
        });
    }

    rsx! {
        canvas {
            id: "{canvas_id()}",
            width: "{width}",
            height: "{height}",
            style: "display: block; width: 100%; height: 100%; border-radius: 8px; background: #05050a;",
        }
    }
}
