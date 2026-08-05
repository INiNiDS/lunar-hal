use crate::components::editor::enemy::Enemy;
use crate::game_state::use_game_version;
use dioxus::prelude::*;
use lunar_stellar_core::enemy::Enemy as EnemyData;
use lunar_stellar_core::{CHUNK_SIZE_PC, Game, PX_PER_PC, Projectile, SectorKey, chunk_center};
use lunar_structures::ResponseStar;
use std::collections::HashSet;

const FIELD_HALF: i32 = 18000;

fn teff_to_rgb(teff: f32) -> (f32, f32, f32) {
    if teff > 30000.0 {
        (0.62, 0.69, 1.0)
    } else if teff > 10000.0 {
        let f = (teff - 10000.0) / 20000.0;
        (0.70 - 0.08 * f, 0.77 - 0.08 * f, 0.95 + 0.05 * f)
    } else if teff > 7500.0 {
        let f = (teff - 7500.0) / 2500.0;
        (0.82 - 0.12 * f, 0.85 - 0.08 * f, 0.95)
    } else if teff > 6000.0 {
        let f = (teff - 6000.0) / 1500.0;
        (0.95 - 0.13 * f, 0.93 - 0.08 * f, 0.90 + 0.05 * f)
    } else if teff > 5200.0 {
        let f = (teff - 5200.0) / 800.0;
        (1.0, 1.0 - 0.07 * f, 0.82 + 0.08 * f)
    } else if teff > 3700.0 {
        let f = (teff - 3700.0) / 1500.0;
        (1.0, 0.85 + 0.08 * f, 0.65 + 0.25 * f)
    } else {
        (1.0, 0.55, 0.35)
    }
}

fn teff_to_rgb8(teff: f32) -> (u8, u8, u8) {
    let (r, g, b) = teff_to_rgb(teff);
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

fn hp_color(ratio: f32) -> &'static str {
    if ratio > 0.6 {
        "#22c55e"
    } else if ratio > 0.25 {
        "#eab308"
    } else {
        "#ef4444"
    }
}
fn render_star(
    key_prefix: &str,
    star: &ResponseStar,
    center_x: f32,
    center_y: f32,
    is_selected: bool,
    on_select: EventHandler<ResponseStar>,
    on_hover: Option<EventHandler<u32>>,
) -> Element {
    let px = (star.x - center_x) * PX_PER_PC;
    let py = (star.y - center_y) * PX_PER_PC;
    let size = (star.radius * 3.0).clamp(3.0, 44.0);
    let (r, g, b) = teff_to_rgb8(star.temperature_k);
    let inner = format!("rgba({},{},{},1.0)", r, g, b);
    let mid = format!("rgba({},{},{},0.4)", r, g, b);

    let border_style = if is_selected {
        format!("border: 1.5px solid {inner};")
    } else {
        String::new()
    };

    let delay = (star.id as f32 * 1.7).fract() * 5.0;
    let star_cloned = star.clone();
    let on_hover = on_hover.clone();
    let star_id = star.id;

    let hp_ratio = (star.hp / 100.0).clamp(0.0, 1.0);
    let hp_c = hp_color(hp_ratio);

    rsx! {
        div {
            key: "{key_prefix}-{star.id}-container",
            class: "absolute pointer-events-none",
            style: "
                left: {px}px;
                top: {py}px;
                width: {size}px;
                height: {size}px;
                transform: translate(-50%, -50%);
            ",

            div {
                class: "absolute inset-0 rounded-full pointer-events-auto cursor-pointer",
                style: "
                    background: radial-gradient(circle, {inner} 0%, {mid} 40%, transparent 80%);
                    animation: star-twinkle 4s ease-in-out {delay}s infinite;
                    {border_style}
                ",
                onclick: move |e| {
                    e.stop_propagation();
                    on_select.call(star_cloned.clone());
                },
                onmouseenter: move |_| {
                    if let Some(ref h) = on_hover {
                        h.call(star_id);
                    }
                }
            }

            if star.hp < 100.0 {
                div {
                    class: "absolute pointer-events-none",
                    style: "
                        left: 50%;
                        top: {size + 4.0}px;
                        width: {size}px;
                        height: 3px;
                        transform: translateX(-50%);
                        background: rgba(255,255,255,0.06);
                        border-radius: 2px;
                        overflow: hidden;
                    ",
                    div {
                        style: "
                            height: 100%;
                            width: {hp_ratio * 100.0}%;
                            background: {hp_c};
                            border-radius: 2px;
                            transition: width 0.3s ease;
                        ",
                    }
                }
            }
        }
    }
}

fn render_loading_chunk(chunk: SectorKey, center_x: f32, center_y: f32) -> Element {
    let (cx, cy) = chunk_center(chunk);
    let px = (cx - center_x) * PX_PER_PC;
    let py = (cy - center_y) * PX_PER_PC;
    let size_px = CHUNK_SIZE_PC * PX_PER_PC;
    let (k0, k1) = chunk;
    rsx! {
        div {
            key: "loading-{k0}-{k1}",
            class: "absolute pointer-events-none",
            style: "
                left: {px}px;
                top: {py}px;
                width: {size_px}px;
                height: {size_px}px;
                transform: translate(-50%, -50%);
                border: 1px dashed rgba(120, 180, 255, 0.35);
                border-radius: 4px;
                animation: chunk-pulse 1.5s ease-in-out infinite;
            ",
        }
    }
}

#[cfg(feature = "web")]
fn measure_viewport_size() -> Option<(f32, f32)> {
    let window = web_sys::window();
    let document = window.as_ref().and_then(|w| w.document());
    let el = document.and_then(|d| d.query_selector(".starmap-root").ok().flatten());
    el.map(|el| {
        let rect = el.get_bounding_client_rect();
        let w = (rect.width() as f32).clamp(100.0, 4000.0);
        let h = (rect.height() as f32).clamp(100.0, 4000.0);
        (w, h)
    })
}

async fn delay_tick() {
    #[cfg(feature = "web")]
    {
        use gloo_timers::future::TimeoutFuture;
        TimeoutFuture::new(80).await;
    }
    #[cfg(not(feature = "web"))]
    {
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    }
}

fn get_viewport_dimensions() -> Option<(f32, f32)> {
    #[cfg(feature = "web")]
    {
        measure_viewport_size()
    }
    #[cfg(not(feature = "web"))]
    {
        Some((1280.0_f32, 800.0_f32))
    }
}

fn use_viewport_measurement() -> Signal<(f32, f32)> {
    let mut viewport = use_signal(|| (0.0_f32, 0.0_f32));

    use_future(move || async move {
        let mut last = (0.0_f32, 0.0_f32);

        for _ in 0..6 {
            delay_tick().await;

            let Some((w, h)) = get_viewport_dimensions() else {
                continue;
            };

            if w <= 0.0 || h <= 0.0 {
                continue;
            }

            if (w, h) != last {
                last = (w, h);
                viewport.set((w, h));
            }

            if w >= 800.0 && h >= 400.0 {
                break;
            }
        }
    });

    viewport
}

fn use_sync_sector_loading(game: Signal<Game>, version: Signal<u64>, viewport: Signal<(f32, f32)>) {
    use_resource(move || async move {
        let _ = version();
        let vp = *viewport.read();
        if vp.0 <= 0.0 || vp.1 <= 0.0 {
            return;
        }
        let pending = game.read().sectors_to_fetch(vp);
        if pending.is_empty() {
            return;
        }
        let temperature = game.read().temperature();
        let bp_rp = game.read().bp_rp();
        let g_mag = game.read().g_mag();
        for (chunk, center) in pending {
            spawn(async move {
                let g: Game = game.read().clone();
                let _ = g
                    .fetch_sector(chunk, center, temperature, bp_rp, g_mag)
                    .await;
            });
        }
    });
}

fn use_sync_sector_eviction(
    game: Signal<Game>,
    version: Signal<u64>,
    viewport: Signal<(f32, f32)>,
) {
    use_resource(move || async move {
        let _ = version();
        let vp = *viewport.read();
        if vp.0 <= 0.0 || vp.1 <= 0.0 {
            return;
        }
        game.read().evict_excess_sectors();
    });
}

// Reduced background star count to lower GPU load
fn use_starfield_backgrounds() -> (Memo<String>, Memo<String>, Memo<String>) {
    let starfield_small = use_memo(move || starfield(0, 150, FIELD_HALF));
    let starfield_medium = use_memo(move || starfield(10000, 75, FIELD_HALF));
    let starfield_distant = use_memo(move || starfield(20000, 100, FIELD_HALF));
    (starfield_small, starfield_medium, starfield_distant)
}

#[derive(Clone)]
struct InteractionState {
    last_mouse: Signal<(f32, f32)>,
    mouse_world: Signal<(f32, f32)>,
}

fn use_star_map_interactions() -> InteractionState {
    let last_mouse = use_signal(|| (0.0_f32, 0.0_f32));
    let mouse_world = use_signal(|| (0.0_f32, 0.0_f32));

    InteractionState {
        last_mouse,
        mouse_world,
    }
}

#[component]
pub fn StarMap(
    game: Signal<Game>,
    world_stars: Vec<ResponseStar>,
    center_x: f32,
    center_y: f32,
    selected_id: Option<u32>,
    on_select: EventHandler<ResponseStar>,
) -> Element {
    let viewport = use_viewport_measurement();
    let version = use_game_version();

    use_sync_sector_loading(game, version, viewport);
    use_sync_sector_eviction(game, version, viewport);

    let mut interact = use_star_map_interactions();

    let g_attn = game;
    let int_attn = interact.clone();

    use_future(move || async move {
        loop {
            delay_tick().await;
            let g = g_attn.read().clone();
            let snap = g.snapshot();
            let (mx, my) = (int_attn.mouse_world)();
            if !snap.sector_stars.is_empty() {
                g.tick_attention(0.08, Some((mx, my)), &snap.sector_stars);
            }

            let _payload = g.update(0.08);
            g.remove_dead_enemies();
        }
    });

    let snap = game.read().snapshot();
    let offset = snap.camera.offset;
    let zoom = snap.camera.zoom;
    let dragging = snap.camera.dragging;

    let sector_stars: Vec<ResponseStar> = snap.sector_stars.clone();
    let loading: HashSet<SectorKey> = snap.sector_loading.clone();
    let enemies: Vec<EnemyData> = snap.enemies.clone();
    let projectiles: Vec<Projectile> = snap.projectiles.clone();

    let (starfield_small, starfield_medium, starfield_distant) = use_starfield_backgrounds();

    let handle_zoom = move |factor: f32| {
        let vp = *viewport.read();
        let g = game.read().clone();
        g.zoom_camera(vp, factor);
    };

    let game_for_hover = game;
    let on_star_hover = EventHandler::new(move |star_id: u32| {
        game_for_hover.read().look_at_star(star_id);
    });

    rsx! {
        style {
            "@keyframes bullet-pulse {{ 0% {{ transform: scale(0.85); }} 100% {{ transform: scale(1.3); }} }}"
        }

        div {
            class: "starmap-root absolute inset-0 cursor-grab active:cursor-grabbing",
            onmousedown: move |e| {
                let g = game.read().clone();
                g.set_dragging(true);
                interact.last_mouse.set((e.client_coordinates().x as f32, e.client_coordinates().y as f32));
            },
            onmousemove: move |e| {
                let nx = e.client_coordinates().x as f32;
                let ny = e.client_coordinates().y as f32;
                let vp = *viewport.read();

                if dragging {
                    let (lx, ly) = (interact.last_mouse)();
                    let g = game.read().clone();
                    g.pan_camera((nx - lx, ny - ly));
                    interact.last_mouse.set((nx, ny));
                }

                let mw = mouse_to_world(nx, ny, vp, offset, zoom, center_x, center_y);
                let old_mw = (interact.mouse_world)();

                let dist_sq = (mw.0 - old_mw.0).powi(2) + (mw.1 - old_mw.1).powi(2);
                if dist_sq > 4.0 {
                    interact.mouse_world.set(mw);
                }

                if !dragging {
                    interact.last_mouse.set((nx, ny));
                }
            },
            onmouseup: move |_| {
                let g = game.read().clone();
                g.set_dragging(false);
            },
            onmouseleave: move |_| {
                let g = game.read().clone();
                g.set_dragging(false);
            },
            onwheel: move |e| {
                let dy = e.delta().strip_units().y;
                let factor = if dy > 0.0 { 1.0 / 1.15 } else { 1.15 };
                let mx = e.client_coordinates().x as f32;
                let my = e.client_coordinates().y as f32;
                let vp = *viewport.read();
                let g = game.read().clone();
                g.zoom_camera_at(vp, (mx, my), factor);
            },

            div {
                class: "absolute inset-0 pointer-events-none flex items-center justify-center overflow-hidden",

                div {
                    class: "absolute inset-0 opacity-15",
                    style: "
                        background-image: linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px);
                        background-size: {200.0 * zoom}px {200.0 * zoom}px;
                        background-position: {offset.0}px {offset.1}px;
                    "
                }

                div {
                    class: "absolute pointer-events-none transition-transform duration-75",
                    style: "transform: translate({offset.0}px, {offset.1}px) scale({zoom});",

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
            }

            div {
                class: "absolute inset-0 pointer-events-none",
                style: "transform: translate({offset.0}px, {offset.1}px) scale({zoom});",

                for star in sector_stars {
                    {
                        let is_sel = selected_id.map(|id| id == star.id).unwrap_or(false);
                        render_star("sector", &star, center_x, center_y, is_sel, on_select, Some(on_star_hover))
                    }
                }

                for star in world_stars {
                    {
                        let is_sel = selected_id.map(|id| id == star.id).unwrap_or(false);
                        render_star("world", &star, center_x, center_y, is_sel, on_select, Some(on_star_hover))
                    }
                }

                for key in loading.iter() {
                    {
                        render_loading_chunk(*key, center_x, center_y)
                    }
                }

                for enemy in enemies {
                    {
                        let e = enemy;
                        let game_c = game;
                        rsx! {
                            Enemy {
                                enemy: e.clone(),
                                center_x,
                                center_y,
                                px_per_pc: PX_PER_PC,
                                on_click: EventHandler::new(move |_| {
                                    game_c.read().damage_enemy(e.id, 1.0);
                                }),
                            }
                        }
                    }
                }

                for proj in projectiles {
                    {
                        let p = proj;
                        let px = (p.coordinates.0 - center_x) * PX_PER_PC;
                        let py = (p.coordinates.1 - center_y) * PX_PER_PC;
                        let size = (p.radius * 2.0).max(12.0);
                        let pid = p.id;
                        let game_c = game;
                        rsx! {
                            div {
                                key: "projectile-{pid}",
                                class: "absolute pointer-events-auto cursor-pointer flex items-center justify-center",
                                style: "
                                    left: {px}px;
                                    top: {py}px;
                                    width: {size}px;
                                    height: {size}px;
                                    transform: translate(-50%, -50%);
                                    z-index: 50;
                                ",
                                onclick: move |e| {
                                    e.stop_propagation();
                                    game_c.read().click_projectile(pid);
                                },
                                div {
                                    style: "
                                        width: 100%;
                                        height: 100%;
                                        background-color: #f97316;
                                        border-radius: 50%;
                                        box-shadow: 0 0 10px #f97316, 0 0 20px #ef4444;
                                        animation: bullet-pulse 0.3s ease-in-out infinite alternate;
                                    ",
                                }
                            }
                        }
                    }
                }
            }

            div {
                class: "absolute right-6 bottom-6 flex flex-col gap-2 bg-black/40 backdrop-blur-xl border border-white/10 p-2 rounded-xl shadow-lg pointer-events-auto",
                button {
                    class: "w-8 h-8 rounded-lg text-white/70 hover:text-white hover:bg-white/10 flex items-center justify-center font-bold text-lg",
                    onclick: move |_| handle_zoom(1.3),
                    "+"
                }
                button {
                    class: "w-8 h-8 rounded-lg text-white/70 hover:text-white hover:bg-white/10 flex items-center justify-center font-bold text-lg",
                    onclick: move |_| handle_zoom(1.0 / 1.3),
                    "-"
                }
                div { class: "w-full h-px bg-white/10 my-1" }
                button {
                    class: "w-8 h-8 rounded-lg text-white/70 hover:text-white hover:bg-white/10 flex items-center justify-center text-[10px] font-bold tracking-wider",
                    onclick: move |_| {
                        let g = game.read().clone();
                        g.recenter_camera();
                    },
                    "RECENTER"
                }
            }
        }
    }
}

fn starfield(seed_offset: u32, count: usize, half_extent: i32) -> String {
    use std::fmt::Write;
    let span = (half_extent * 2) as f32;
    let mut out = String::new();
    for i in 0..count {
        let idx = seed_offset.wrapping_add(i as u32);
        let x = (prng(idx.wrapping_mul(3)) * span) as i32 - half_extent;
        let y = (prng(idx.wrapping_mul(3).wrapping_add(1)) * span) as i32 - half_extent;
        let opacity = 0.1 + prng(idx.wrapping_mul(3).wrapping_add(2)) * 0.7;
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{}px {}px rgba(255,255,255,{:.2})", x, y, opacity);
    }
    out
}

fn prng(mut seed: u32) -> f32 {
    seed = seed.wrapping_mul(0x45d9f3b);
    seed = (seed ^ (seed >> 16)).wrapping_mul(0x45d9f3b);
    seed ^= seed >> 16;
    (seed as f32) / (u32::MAX as f32)
}

fn mouse_to_world(
    mx: f32,
    my: f32,
    vp: (f32, f32),
    offset: (f32, f32),
    zoom: f32,
    center_x: f32,
    center_y: f32,
) -> (f32, f32) {
    let cx = vp.0 * 0.5;
    let cy = vp.1 * 0.5;
    let wx = (mx - cx - offset.0) / zoom / PX_PER_PC + center_x;
    let wy = (my - cy - offset.1) / zoom / PX_PER_PC + center_y;
    (wx, wy)
}
