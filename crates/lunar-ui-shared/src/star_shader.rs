use std::cell::RefCell;
use std::rc::Rc;
use dioxus::prelude::*;
use wasm_bindgen::{JsCast, JsValue, prelude::Closure};
use web_sys::WebGl2RenderingContext as Gl2;

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

struct StarProps {
    teff: f32, bp_rp: f32, scale: f32, speed: f32, contrast: f32,
}

#[allow(dead_code)]
struct GlState {
    _slot: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>,
    ctx: Gl2,
    u_teff: web_sys::WebGlUniformLocation,
    u_bp_rp: web_sys::WebGlUniformLocation,
    u_scale: web_sys::WebGlUniformLocation,
    u_speed: web_sys::WebGlUniformLocation,
    u_contrast: web_sys::WebGlUniformLocation,
}

thread_local! {
    static GL_STATE: RefCell<Option<GlState>> = const { RefCell::new(None) };
}

#[component]
pub fn StarShaderCanvas(
    width: u32, height: u32,
    teff: f64, bp_rp: f64,
    noise_scale: f64, noise_speed: f64, contrast: f64,
) -> Element {
    let canvas_id = use_signal(|| {
        format!("star-gl-{:016x}", js_sys::Math::random().to_bits())
    });
    let props: Signal<Rc<RefCell<StarProps>>> = use_signal(|| {
        Rc::new(RefCell::new(StarProps {
            teff: teff as f32, bp_rp: bp_rp as f32,
            scale: noise_scale as f32, speed: noise_speed as f32,
            contrast: contrast as f32,
        }))
    });

    {
        let p = &*props.read();
        let mut mp = p.borrow_mut();
        mp.teff = teff as f32;
        mp.bp_rp = bp_rp as f32;
        mp.scale = noise_scale as f32;
        mp.speed = noise_speed as f32;
        mp.contrast = contrast as f32;
    }

    {
        let id = canvas_id.read().clone();
        let p2 = props.clone();
        let w = width;
        let h = height;
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
                        gl_err!("StarShader: element is not a canvas");
                        return;
                    }
                },
                None => {
                    gl_err!("StarShader: canvas element not found");
                    return;
                }
            };
            canvas.set_width(w);
            canvas.set_height(h);

            let ctx = match canvas.get_context("webgl2") {
                Ok(Some(c)) => match c.dyn_into::<Gl2>() {
                    Ok(gl) => gl,
                    Err(_) => {
                        gl_err!("StarShader: context is not WebGl2RenderingContext");
                        return;
                    }
                },
                Ok(None) => {
                    gl_err!("StarShader: WebGL2 not supported");
                    return;
                }
                Err(e) => {
                    gl_err!("StarShader: get_context error: {:?}", e);
                    return;
                }
            };

            let vs = match ctx.create_shader(Gl2::VERTEX_SHADER) {
                Some(s) => s,
                None => {
                    gl_err!("StarShader: failed to create vertex shader");
                    return;
                }
            };
            ctx.shader_source(&vs, VERT_SRC);
            ctx.compile_shader(&vs);
            if !ctx.get_shader_parameter(&vs, Gl2::COMPILE_STATUS).as_bool().unwrap_or(false) {
                let log = ctx.get_shader_info_log(&vs).unwrap_or_default();
                gl_err!("StarShader: vertex shader compile error: {}", log);
                return;
            }

            let fs = match ctx.create_shader(Gl2::FRAGMENT_SHADER) {
                Some(s) => s,
                None => {
                    gl_err!("StarShader: failed to create fragment shader");
                    return;
                }
            };
            ctx.shader_source(&fs, FRAG_SRC);
            ctx.compile_shader(&fs);
            if !ctx.get_shader_parameter(&fs, Gl2::COMPILE_STATUS).as_bool().unwrap_or(false) {
                let log = ctx.get_shader_info_log(&fs).unwrap_or_default();
                gl_err!("StarShader: fragment shader compile error: {}", log);
                return;
            }

            let prog = match ctx.create_program() {
                Some(p) => p,
                None => {
                    gl_err!("StarShader: failed to create program");
                    return;
                }
            };
            ctx.attach_shader(&prog, &vs);
            ctx.attach_shader(&prog, &fs);
            ctx.link_program(&prog);
            if !ctx.get_program_parameter(&prog, Gl2::LINK_STATUS).as_bool().unwrap_or(false) {
                let log = ctx.get_program_info_log(&prog).unwrap_or_default();
                gl_err!("StarShader: program link error: {}", log);
                return;
            }
            ctx.use_program(Some(&prog));

            let verts: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
            let buf = match ctx.create_buffer() {
                Some(b) => b,
                None => {
                    gl_err!("StarShader: failed to create buffer");
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

            ctx.viewport(0, 0, w as i32, h as i32);
            ctx.clear_color(0.02, 0.02, 0.04, 1.0);

            let (Some(u_time), Some(u_teff), Some(u_bp_rp), Some(u_scale), Some(u_speed), Some(u_contrast)) =
                (ctx.get_uniform_location(&prog, "uTime"),
                 ctx.get_uniform_location(&prog, "uTeff"),
                 ctx.get_uniform_location(&prog, "uBpRp"),
                 ctx.get_uniform_location(&prog, "uScale"),
                 ctx.get_uniform_location(&prog, "uSpeed"),
                 ctx.get_uniform_location(&prog, "uContrast"))
            else {
                gl_err!("StarShader: failed to get uniform locations");
                return;
            };

            let slot: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
            let slot2 = slot.clone();
            let w2 = window.clone();
            let ctx2 = ctx.clone();
            let p3 = p2.clone();
            let mut time = 0.0f32;

            let u_teff_c = u_teff.clone();
            let u_bp_rp_c = u_bp_rp.clone();
            let u_scale_c = u_scale.clone();
            let u_speed_c = u_speed.clone();
            let u_contrast_c = u_contrast.clone();
            *slot.borrow_mut() = Some(Closure::new(move |_ts: f64| {
                time += 1.0 / 60.0;
                let pr = &*p3.read();
                let sp = pr.borrow();
                ctx2.uniform1f(Some(&u_teff_c), sp.teff);
                ctx2.uniform1f(Some(&u_bp_rp_c), sp.bp_rp);
                ctx2.uniform1f(Some(&u_scale_c), sp.scale);
                ctx2.uniform1f(Some(&u_speed_c), sp.speed);
                ctx2.uniform1f(Some(&u_contrast_c), sp.contrast);
                ctx2.uniform1f(Some(&u_time), time);
                ctx2.clear(Gl2::COLOR_BUFFER_BIT);
                ctx2.draw_arrays(Gl2::TRIANGLE_STRIP, 0, 4);
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
                    ctx,
                    u_teff, u_bp_rp,
                    u_scale, u_speed, u_contrast,
                });
            });
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
