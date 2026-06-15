use crate::api;
use crate::assets::FONT_SANS;
use crate::components::editor::sidebar::StarSidebar;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use dioxus::prelude::*;
use lunar_structures::{LoreMetadata, PipelineRequest, ResponseStar, StarLore};
use std::io::Cursor;
use nah::high_complexity;

fn prng(mut seed: u32) -> f32 {
    seed = seed.wrapping_mul(0x45d9f3b);
    seed = (seed ^ (seed >> 16)).wrapping_mul(0x45d9f3b);
    seed ^= seed >> 16;
    (seed as f32) / (u32::MAX as f32)
}

fn generate_starfield(seed_offset: u32, count: usize) -> String {
    (0..count)
        .map(|i| {
            let idx = seed_offset + i as u32;
            let x = (prng(idx * 3) * 8000.0) as i32 - 4000;
            let y = (prng(idx * 3 + 1) * 8000.0) as i32 - 4000;
            let opacity = 0.1 + prng(idx * 3 + 2) * 0.7;
            format!("{}px {}px rgba(255,255,255,{:.2})", x, y, opacity)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn generate_distant_stars() -> String {
    (0..3000)
        .map(|i| {
            let idx = 20000 + i as u32;
            let x = (prng(idx * 3) * 12000.0) as i32 - 6000;
            let y = (prng(idx * 3 + 1) * 12000.0) as i32 - 6000;
            let opacity = 0.05 + prng(idx * 3 + 2) * 0.25;
            format!("{}px {}px rgba(255,255,255,{:.2})", x, y, opacity)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn get_star_color(temp: f32) -> &'static str {
    if temp > 25000.0 { "#93c5fd" }
    else if temp > 10000.0 { "#bfdbfe" }
    else if temp > 7500.0 { "#ffffff" }
    else if temp > 6000.0 { "#fef08a" }
    else if temp > 5200.0 { "#fde047" }
    else if temp > 3700.0 { "#fdba74" }
    else { "#fca5a5" }
}

#[high_complexity]
#[component]
pub fn EditorPage() -> Element {
    let mut temperature = use_signal(|| 0.7_f32);
    let mut bp_rp = use_signal(|| 1.0_f32);
    let mut g_mag = use_signal(|| 10.0_f32);

    let mut sidebar_open = use_signal(|| false);
    let mut selected_star = use_signal(|| Option::<ResponseStar>::None);

    let mut offset = use_signal(|| (0.0_f32, 0.0_f32));
    let mut zoom = use_signal(|| 1.0_f32);
    let mut is_dragging = use_signal(|| false);
    let mut last_mouse = use_signal(|| (0.0_f32, 0.0_f32));

    let starfield_small = use_memo(move || generate_starfield(0, 2000));
    let starfield_medium = use_memo(move || generate_starfield(10000, 400));
    let starfield_distant = use_memo(move || generate_distant_stars());

    let mut sector_center = use_signal(|| None::<(f32, f32, f32)>);
    let mut last_temp = use_signal(|| 0.7_f32);

    let all_data = use_resource(move || {
        let t = temperature();
        let center = sector_center();
        async move {
            let (cx, cy, cz, b, g) = if let Some((ax, ay, az)) = center {
                (ax, ay, az, bp_rp(), g_mag())
            } else {
                let r = api::fetch_random_star(t).await.ok()?;
                bp_rp.set(r.bp_rp);
                g_mag.set(r.g_mag);
                (r.x_pc, r.y_pc, r.z_pc, r.bp_rp, r.g_mag)
            };
            let gnn_resp = api::fetch_gnn_by_temp(cx, cy, cz, 250.0, t, b, g).await.ok()?;
            Some((cx, cy, cz, gnn_resp.stars))
        }
    });

    if *last_temp.peek() != temperature() {
        last_temp.set(temperature());
        sector_center.set(None);
    }
    if sector_center.peek().is_none() && all_data.read().as_ref().and_then(|d| d.as_ref()).is_some() {
        if let Some(Some((cx, cy, cz, _))) = &*all_data.read() {
            sector_center.set(Some((*cx, *cy, *cz)));
        }
    }

    let (center_x, center_y, center_z, stars) = match &*all_data.read() {
        Some(Some((cx, cy, cz, s))) => (*cx, *cy, *cz, s.clone()),
        _ => (0.0, 0.0, 0.0, Vec::new()),
    };

    let pipeline_data = use_resource(move || {
        let star_opt = selected_star();
        let b = bp_rp();
        let g = g_mag();

        async move {
            if let Some(star) = star_opt {
                let req = PipelineRequest {
                    x_pc: star.x,
                    y_pc: star.y,
                    z_pc: star.z,
                    bp_rp: b,
                    g_mag: g,
                    texture_size: 256,
                };

                if let Ok(resp) = api::fetch_pipeline(req).await {
                    return Some(resp);
                }
            }
            None
        }
    });

    let (pinn_data, lore_data, siren_texture_b64) = match &*pipeline_data.read() {
        Some(Some(pipeline)) => {
            let p = pipeline.pinn.clone();
            let m = &pipeline.metadata;
            let l = StarLore {
                designated_name: m.designated_name.clone(),
                category: m.category.clone(),
                visual_profile: m.spectral_class.clone(),
                system_lore: m.description.clone(),
                metadata: LoreMetadata {
                    simulation_engine: "Lunar Pipeline v2.0".to_string(),
                    data_source: "PINN + GNN + SIREN".to_string(),
                    complexity_level: "High".to_string(),
                },
            };

            let mut png_data = Cursor::new(Vec::new());
            let b64 = if let Some(img) = image::RgbImage::from_raw(pipeline.siren.width, pipeline.siren.height, pipeline.siren.pixels.clone()) {
                if img.write_to(&mut png_data, image::ImageFormat::Png).is_ok() {
                    Some(STANDARD.encode(png_data.into_inner()))
                } else {
                    None
                }
            } else {
                None
            };

            (Some(p), Some(l), b64)
        },
        _ => (None, None, None),
    };

    rsx! {
        div {
            class: "h-screen w-screen bg-[#050505] overflow-hidden relative select-none [background-image:radial-gradient(ellipse_at_center,_rgba(255,255,255,0.015)_0%,_transparent_70%)] cursor-grab active:cursor-grabbing",
            style: "font-family: {FONT_SANS};",

            onmousedown: move |e| {
                is_dragging.set(true);
                last_mouse.set((e.client_coordinates().x as f32, e.client_coordinates().y as f32));
            },
            onmousemove: move |e| {
                if is_dragging() {
                    let (lx, ly) = last_mouse();
                    let nx = e.client_coordinates().x as f32;
                    let ny = e.client_coordinates().y as f32;
                    let (ox, oy) = offset();
                    offset.set((ox + (nx - lx), oy + (ny - ly)));
                    last_mouse.set((nx, ny));
                }
            },
            onmouseup: move |_| {
                is_dragging.set(false);
                if let Some((ax, ay, _)) = sector_center() {
                    if let Some(Some((cx, cy, _, _))) = &*all_data.read() {
                        let vx = cx - offset().0 / (zoom() * 15.0);
                        let vy = cy - offset().1 / (zoom() * 15.0);
                        let dx = vx - ax;
                        let dy = vy - ay;
                        if (dx * dx + dy * dy).sqrt() > 150.0 {
                            sector_center.set(Some((vx, vy, 0.0)));
                        }
                    }
                }
            },
            onmouseleave: move |_| is_dragging.set(false),
            onwheel: move |e| {
                let dy = e.delta().strip_units().y;
                let factor = if dy > 0.0 { 1.0 / 1.15 } else { 1.15 };
                zoom.set((zoom() * factor).clamp(0.05, 15.0));
            },

            div {
                class: "absolute inset-0 pointer-events-none flex items-center justify-center transition-transform duration-75",
                style: "transform: translate({offset().0}px, {offset().1}px) scale({zoom()});",

                div {
                    class: "absolute w-[8000px] h-[8000px] opacity-15",
                    style: "background-image: linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px); background-size: 100px 100px; background-position: center;"
                }

                div {
                    class: "absolute bg-transparent",
                    style: "width: 1px; height: 1px; box-shadow: {starfield_small()};"
                }
                div {
                    class: "absolute bg-transparent rounded-full",
                    style: "width: 2px; height: 2px; box-shadow: {starfield_medium()};"
                }

                div {
                    class: "absolute bg-transparent",
                    style: "width: 1px; height: 1px; box-shadow: {starfield_distant()};"
                }

                div {
                    class: "absolute w-8 h-8 border border-white/20 rounded-full flex items-center justify-center",
                    div { class: "w-1 h-1 bg-white/40 rounded-full" }
                }
            }

            div {
                class: "absolute inset-0 pointer-events-none",
                style: "transform: translate({offset().0}px, {offset().1}px);",
                for star in stars.clone() {
                    {
                        let px = (star.x - center_x) * 15.0 * zoom();
                        let py = (star.y - center_y) * 15.0 * zoom();
                        let size = ((star.radius * 3.0) * zoom()).clamp(2.0, 40.0);
                        let color = get_star_color(star.temperature_k);
                        let is_selected = selected_star().map(|s| s.id == star.id).unwrap_or(false);

                        let halo_size = (size * 4.0).max(12.0);
                        let selected_style = if is_selected {
                            format!("border: 1.5px solid {color}; box-shadow: 0 0 {size}px {color}, 0 0 {halo_size}px {color}60, 0 0 {size}px {color} inset;")
                        } else {
                            String::new()
                        };

                        let delay = (star.id as f32 * 1.7).fract() * 5.0;
                        rsx! {
                            div {
                                key: "{star.id}",
                                class: "absolute rounded-full -translate-x-1/2 -translate-y-1/2 pointer-events-auto cursor-pointer",
                                style: "
                                    left: {px}px;
                                    top: {py}px;
                                    width: {size}px;
                                    height: {size}px;
                                    background: {color};
                                    box-shadow: 0 0 {size}px {color}, 0 0 {size * 2.5}px {color}50;
                                    animation: star-twinkle 4s ease-in-out {delay}s infinite;
                                    {selected_style}
                                ",
                                onclick: move |e| {
                                    e.stop_propagation();
                                    selected_star.set(Some(star.clone()));
                                    sidebar_open.set(true);
                                }
                            }
                        }
                    }
                }
            }

            if sidebar_open() {
                StarSidebar {
                    temperature,
                    on_temperature_change: move |val| temperature.set(val),
                    bp_rp,
                    g_mag,
                    selected: selected_star().is_some(),
                    pinn_data,
                    lore_data,
                    siren_texture_b64,
                }
            } else {
                div {
                    class: "absolute left-4 top-4 px-5 py-2.5 bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl text-white/70 text-[10px] font-bold uppercase tracking-[0.2em] cursor-pointer hover:bg-white/10 hover:text-white transition-colors shadow-lg pointer-events-auto",
                    onclick: move |_| sidebar_open.set(true),
                    "Open Scanner"
                }
            }

            div {
                class: "absolute top-4 right-4 flex items-center gap-3 bg-black/40 backdrop-blur-xl border border-white/10 px-4 py-2 rounded-xl text-white/60 text-[10px] uppercase tracking-widest",
                if all_data.read().is_none() {
                    div { class: "w-3 h-3 border-2 border-white/20 border-t-white/80 rounded-full animate-spin" }
                    span { "Scanning Sector..." }
                } else if stars.is_empty() {
                    span { class: "text-red-400", "No stars detected" }
                } else {
                    span { "Sector: [{center_x:.0}, {center_y:.0}, {center_z:.0}]" }
                    div { class: "w-px h-3 bg-white/20 mx-1" }
                    span { "{stars.len()} Stars" }
                }
            }

            div {
                class: "absolute right-6 bottom-6 flex flex-col gap-2 bg-black/40 backdrop-blur-xl border border-white/10 p-2 rounded-xl shadow-lg pointer-events-auto",
                button {
                    class: "w-8 h-8 rounded-lg text-white/70 hover:text-white hover:bg-white/10 flex items-center justify-center font-bold text-lg",
                    onclick: move |_| zoom.set((zoom() * 1.3).min(15.0)),
                    "+"
                }
                button {
                    class: "w-8 h-8 rounded-lg text-white/70 hover:text-white hover:bg-white/10 flex items-center justify-center font-bold text-lg",
                    onclick: move |_| zoom.set((zoom() / 1.3).max(0.05)),
                    "-"
                }
                div { class: "w-full h-px bg-white/10 my-1" }
                button {
                    class: "w-8 h-8 rounded-lg text-white/70 hover:text-white hover:bg-white/10 flex items-center justify-center text-[10px] font-bold tracking-wider",
                    onclick: move |_| { offset.set((0.0, 0.0)); zoom.set(1.0); },
                    "RECENTER"
                }
            }
        }
    }
}