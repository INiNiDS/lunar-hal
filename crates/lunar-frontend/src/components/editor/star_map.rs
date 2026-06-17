use dioxus::prelude::*;
use lunar_structures::ResponseStar;
use std::collections::{HashMap, HashSet};

#[cfg(feature = "web")]
fn now_ms() -> f64 {
    js_sys::Date::now()
}

#[cfg(not(feature = "web"))]
fn now_ms() -> f64 {
    use std::sync::LazyLock;
    use std::time::Instant;
    static ORIGIN: LazyLock<Instant> = LazyLock::new(Instant::now);
    ORIGIN.elapsed().as_secs_f64() * 1000.0
}

#[cfg(feature = "web")]
async fn sleep_ms(ms: u32) {
    gloo_timers::future::TimeoutFuture::new(ms).await;
}

#[cfg(not(feature = "web"))]
async fn sleep_ms(ms: u32) {
    tokio::time::sleep(std::time::Duration::from_millis(ms as u64)).await;
}

const FIELD_HALF: i32 = 18000;
const PX_PER_PC: f32 = 15.0;
const CHUNK_SIZE_PC: f32 = 400.0;
const INNER_EXCLUSION_PC: f32 = 450.0;
const MAX_CACHED_CHUNKS: usize = 64;
const PREFETCH_PAD_CHUNKS: i32 = 1;
const MAX_CONCURRENT_FETCHES: usize = 3;
const FETCH_COOLDOWN_MS: u32 = 120;

fn prng(mut seed: u32) -> f32 {
    seed = seed.wrapping_mul(0x45d9f3b);
    seed = (seed ^ (seed >> 16)).wrapping_mul(0x45d9f3b);
    seed ^= seed >> 16;
    (seed as f32) / (u32::MAX as f32)
}

fn generate_starfield(seed_offset: u32, count: usize, half_extent: i32) -> String {
    let span = (half_extent * 2) as f32;
    (0..count)
        .map(|i| {
            let idx = seed_offset + i as u32;
            let x = (prng(idx * 3) * span) as i32 - half_extent;
            let y = (prng(idx * 3 + 1) * span) as i32 - half_extent;
            let opacity = 0.1 + prng(idx * 3 + 2) * 0.7;
            format!("{}px {}px rgba(255,255,255,{:.2})", x, y, opacity)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn generate_distant_stars() -> String {
    generate_starfield(20000, 2000, FIELD_HALF)
}

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

fn visible_chunks(
    cam_offset: (f32, f32),
    cam_zoom: f32,
    viewport: (f32, f32),
    world_center: (f32, f32),
) -> Vec<(i32, i32)> {
    if viewport.0 <= 0.0 || viewport.1 <= 0.0 || cam_zoom <= 0.0 {
        return Vec::new();
    }

    let center_x = world_center.0 - cam_offset.0 / (cam_zoom * PX_PER_PC);
    let center_y = world_center.1 - cam_offset.1 / (cam_zoom * PX_PER_PC);
    let half_w = (viewport.0 * 0.5) / (cam_zoom * PX_PER_PC);
    let half_h = (viewport.1 * 0.5) / (cam_zoom * PX_PER_PC);

    let min_cx = ((center_x - half_w) / CHUNK_SIZE_PC).floor() as i32 - PREFETCH_PAD_CHUNKS;
    let max_cx = ((center_x + half_w) / CHUNK_SIZE_PC).floor() as i32 + PREFETCH_PAD_CHUNKS;
    let min_cy = ((center_y - half_h) / CHUNK_SIZE_PC).floor() as i32 - PREFETCH_PAD_CHUNKS;
    let max_cy = ((center_y + half_h) / CHUNK_SIZE_PC).floor() as i32 + PREFETCH_PAD_CHUNKS;

    let mut chunks = Vec::new();
    for cx in min_cx..=max_cx {
        for cy in min_cy..=max_cy {
            chunks.push((cx, cy));
        }
    }
    chunks
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

fn render_star(
    key_prefix: &str,
    star: &ResponseStar,
    center_x: f32,
    center_y: f32,
    is_selected: bool,
    on_select: EventHandler<ResponseStar>,
) -> Element {
    let px = (star.x - center_x) * PX_PER_PC;
    let py = (star.y - center_y) * PX_PER_PC;
    let size = (star.radius * 3.0).clamp(3.0, 44.0);
    let (r, g, b) = teff_to_rgb8(star.temperature_k);
    let inner = format!("rgba({},{},{},1.0)", r, g, b);
    let mid = format!("rgba({},{},{},0.45)", r, g, b);
    let mid_size = size * 4.0;

    let border_style = if is_selected {
        format!("border: 1.5px solid {inner}; box-shadow: 0 0 {size}px {inner}, 0 0 {mid_size}px {mid};")
    } else {
        format!("box-shadow: 0 0 {size}px {inner}, 0 0 {mid_size}px {mid};")
    };

    let delay = (star.id as f32 * 1.7).fract() * 5.0;

    let star_cloned = star.clone();

    rsx! {
        div {
            key: "{key_prefix}-{star.id}",
            class: "absolute rounded-full pointer-events-auto cursor-pointer",
            style: "
                left: {px}px;
                top: {py}px;
                width: {size}px;
                height: {size}px;
                background: radial-gradient(circle, {inner} 0%, rgba({r},{g},{b},0.6) 50%, transparent 100%);
                transform: translate(-50%, -50%);
                animation: star-twinkle 4s ease-in-out {delay}s infinite;
                {border_style}
            ",
            onclick: move |e| {
                e.stop_propagation();
                on_select.call(star_cloned.clone());
            }
        }
    }
}

fn chunk_center(chunk: (i32, i32)) -> (f32, f32) {
    (
        (chunk.0 as f32 + 0.5) * CHUNK_SIZE_PC,
        (chunk.1 as f32 + 0.5) * CHUNK_SIZE_PC,
    )
}

fn chunk_distance_sq(chunk: (i32, i32), target: (f32, f32)) -> f32 {
    let (cx, cy) = chunk_center(chunk);
    let dx = cx - target.0;
    let dy = cy - target.1;
    dx * dx + dy * dy
}

fn is_excluded(chunk: (i32, i32), center: (f32, f32)) -> bool {
    chunk_distance_sq(chunk, center) < INNER_EXCLUSION_PC * INNER_EXCLUSION_PC
}

fn evict_excess_cache(
    cache: &mut HashMap<(i32, i32), Vec<ResponseStar>>,
    cam_pos: (f32, f32),
) {
    if cache.len() <= MAX_CACHED_CHUNKS {
        return;
    }

    let mut keys: Vec<(i32, i32)> = cache.keys().copied().collect();

    keys.sort_unstable_by(|&a, &b| {
        let dist_a = chunk_distance_sq(a, cam_pos);
        let dist_b = chunk_distance_sq(b, cam_pos);
        dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
    });

    let to_remove = keys.len() - MAX_CACHED_CHUNKS;
    for key in keys.iter().rev().take(to_remove) {
        cache.remove(key);
    }
}


fn fetch_neighbor_sectors(
    vp: (f32, f32),
    off: (f32, f32),
    zm: f32,
    center_coords: (f32, f32, f32),
    temp: f32,
    bprp: f32,
    gm: f32,
    mut sector_cache: Signal<HashMap<(i32, i32), Vec<ResponseStar>>>,
    mut sector_loading: Signal<HashSet<(i32, i32)>>,
    mut last_fetch_at: Signal<f64>,
) {
    let now = now_ms();
    if now - *last_fetch_at.read() < FETCH_COOLDOWN_MS as f64 {
        return;
    }

    let chunks = visible_chunks(off, zm, vp, (center_coords.0, center_coords.1));
    let in_flight = sector_loading.peek().len();
    let mut budget = MAX_CONCURRENT_FETCHES.saturating_sub(in_flight);
    let mut dispatched = 0;

    for chunk in chunks {
        if budget == 0 {
            break;
        }
        if is_excluded(chunk, (center_coords.0, center_coords.1)) {
            continue;
        }
        if sector_cache.peek().contains_key(&chunk) || sector_loading.peek().contains(&chunk) {
            continue;
        }

        sector_loading.write().insert(chunk);
        let (sector_cx, sector_cy) = chunk_center(chunk);
        let sector_cz = center_coords.2;

        spawn(async move {
            if let Ok(resp) = crate::api::fetch_sector_stars(
                sector_cx, sector_cy, sector_cz, temp, bprp, gm,
            ).await {
                sector_cache.write().insert(chunk, resp.stars);
            }
            sector_loading.write().remove(&chunk);
        });

        budget -= 1;
        dispatched += 1;
    }

    if dispatched > 0 {
        last_fetch_at.set(now);
    }

    let cam_world_x = center_coords.0 - off.0 / (zm * PX_PER_PC);
    let cam_world_y = center_coords.1 - off.1 / (zm * PX_PER_PC);
    evict_excess_cache(&mut sector_cache.write(), (cam_world_x, cam_world_y));
}

#[component]
pub fn StarMap(
    stars: Vec<ResponseStar>,
    center_x: f32,
    center_y: f32,
    center_z: f32,
    temperature: f32,
    bp_rp: f32,
    g_mag: f32,
    world_id: Option<String>,
    selected_star: Signal<Option<ResponseStar>>,
    cam_offset: Signal<(f32, f32)>,
    cam_zoom: Signal<f32>,
    cam_dragging: Signal<bool>,
    on_select: EventHandler<ResponseStar>,
) -> Element {
    let mut last_mouse = use_signal(|| (0.0_f32, 0.0_f32));
    let mut viewport = use_signal(|| (0.0_f32, 0.0_f32));

    let starfield_small = use_memo(move || generate_starfield(0, 3000, FIELD_HALF));
    let starfield_medium = use_memo(move || generate_starfield(10000, 1200, FIELD_HALF));
    let starfield_distant = use_memo(move || generate_distant_stars());

    let mut sector_cache = use_signal(HashMap::<(i32, i32), Vec<ResponseStar>>::new);
    let mut sector_loading = use_signal(HashSet::<(i32, i32)>::new);
    let mut prev_world_id = use_signal(|| Option::<String>::None);
    let last_fetch_at = use_signal(|| 0.0_f64);

    {
        use_effect(move || {
            if let Some(id) = &world_id {
                let prev = prev_world_id();
                if prev.as_deref() != Some(id.as_str()) {
                    prev_world_id.set(Some(id.clone()));
                    sector_cache.write().clear();
                    sector_loading.write().clear();
                }
            } else {
                prev_world_id.set(None);
                sector_cache.write().clear();
                sector_loading.write().clear();
            }
        });
    }

    {
        use_effect(move || {
            let vp = *viewport.read();
            let off = *cam_offset.read();
            let zm = *cam_zoom.read();
            if vp.0 <= 0.0 || vp.1 <= 0.0 || zm <= 0.0 {
                return;
            }

            fetch_neighbor_sectors(
                vp,
                off,
                zm,
                (center_x, center_y, center_z),
                temperature,
                bp_rp,
                g_mag,
                sector_cache,
                sector_loading,
                last_fetch_at,
            );
        });
    }

    let offset = *cam_offset.read();
    let zoom = *cam_zoom.read();
    let sector_snapshot: Vec<ResponseStar> = sector_cache
        .read()
        .values()
        .flatten()
        .cloned()
        .collect();

    let mut handle_zoom = move |factor: f32| {
        let (vp_w, vp_h) = *viewport.read();
        let (cx, cy) = (vp_w * 0.5, vp_h * 0.5);
        let cur_zoom = *cam_zoom.read();
        let cur_off = *cam_offset.read();
        let new_zoom = (cur_zoom * factor).clamp(0.05, 15.0);
        let wx = (cx - cur_off.0) / cur_zoom;
        let wy = (cy - cur_off.1) / cur_zoom;
        let new_off_x = cx - wx * new_zoom;
        let new_off_y = cy - wy * new_zoom;
        cam_zoom.set(new_zoom);
        cam_offset.set((new_off_x, new_off_y));
    };

    rsx! {
        div {
            class: "starmap-root absolute inset-0 cursor-grab active:cursor-grabbing",
            onmounted: move |_| {
                spawn(async move {
                    let mut last = (0.0_f32, 0.0_f32);
                    for _ in 0..6 {
                        sleep_ms(80).await;
                        #[cfg(feature = "web")]
                        let measured = measure_viewport_size();
                        #[cfg(not(feature = "web"))]
                        let measured = Some((1280.0_f32, 800.0_f32));

                        if let Some((w, h)) = measured {
                            if w > 0.0 && h > 0.0 {
                                if (w, h) != last {
                                    last = (w, h);
                                    #[cfg(feature = "web")]
                                    web_sys::console::log_1(
                                        &format!("[starmap] set viewport {}x{}", w, h).into(),
                                    );
                                    viewport.set((w, h));
                                }
                                if w >= 800.0 && h >= 400.0 {
                                    break;
                                }
                            }
                        }
                    }
                });
            },
            onmousedown: move |e| {
                cam_dragging.set(true);
                last_mouse.set((e.client_coordinates().x as f32, e.client_coordinates().y as f32));
            },
            onmousemove: move |e| {
                if *cam_dragging.read() {
                    let (lx, ly) = last_mouse();
                    let nx = e.client_coordinates().x as f32;
                    let ny = e.client_coordinates().y as f32;
                    let (ox, oy) = *cam_offset.read();
                    cam_offset.set((ox + (nx - lx), oy + (ny - ly)));
                    last_mouse.set((nx, ny));
                }
            },
            onmouseup: move |_| {
                cam_dragging.set(false);
            },
            onmouseleave: move |_| {
                cam_dragging.set(false);
            },
            onwheel: move |e| {
                let dy = e.delta().strip_units().y;
                let factor = if dy > 0.0 { 1.0 / 1.15 } else { 1.15 };
                let mx = e.client_coordinates().x as f32;
                let my = e.client_coordinates().y as f32;
                let (vp_w, vp_h) = *viewport.read();
                let (cx, cy) = (vp_w * 0.5, vp_h * 0.5);
                let cur_zoom = *cam_zoom.read();
                let cur_off = *cam_offset.read();
                let wx = (mx - cx - cur_off.0) / cur_zoom;
                let wy = (my - cy - cur_off.1) / cur_zoom;
                let nz = (cur_zoom * factor).clamp(0.05, 15.0);
                let new_off_x = mx - cx - wx * nz;
                let new_off_y = my - cy - wy * nz;
                cam_zoom.set(nz);
                cam_offset.set((new_off_x, new_off_y));
            },

            div {
                class: "absolute inset-0 pointer-events-none flex items-center justify-center transition-transform duration-75",
                style: "transform: translate({offset.0}px, {offset.1}px) scale({zoom});",

                div {
                    class: "absolute opacity-15",
                    style: "width: {36000}px; height: {36000}px; left: {-18000}px; top: {-18000}px; background-image: linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px); background-size: 200px 200px; background-position: center;"
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
                style: "transform: translate({offset.0}px, {offset.1}px) scale({zoom});",

                for star in sector_snapshot {
                    {
                        let is_sel = selected_star().map(|s| s.id == star.id).unwrap_or(false);
                        render_star("sector", &star, center_x, center_y, is_sel, on_select)
                    }
                }

                for star in stars {
                    {
                        let is_sel = selected_star().map(|s| s.id == star.id).unwrap_or(false);
                        render_star("world", &star, center_x, center_y, is_sel, on_select)
                    }
                }

                for key in sector_loading.read().iter() {
                    {
                        let chunk_world_x = (key.0 as f32 + 0.5) * CHUNK_SIZE_PC;
                        let chunk_world_y = (key.1 as f32 + 0.5) * CHUNK_SIZE_PC;
                        let px = (chunk_world_x - center_x) * PX_PER_PC;
                        let py = (chunk_world_y - center_y) * PX_PER_PC;
                        let size_px = CHUNK_SIZE_PC * PX_PER_PC;
                        let k0 = key.0;
                        let k1 = key.1;
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
                    onclick: move |_| { cam_offset.set((0.0, 0.0)); cam_zoom.set(1.0); },
                    "RECENTER"
                }
            }
        }
    }
}