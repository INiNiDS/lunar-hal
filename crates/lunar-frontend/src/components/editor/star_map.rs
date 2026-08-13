use crate::stellar_state::use_stellar_scene_version;
use dioxus::prelude::*;
use lunar_stellar_core::sector::FETCH_COOLDOWN_MS;
use lunar_stellar_core::{CHUNK_SIZE_PC, PX_PER_PC, SectorKey, StellarScene, chunk_center};
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

fn render_star(
    key_prefix: &str,
    star: &ResponseStar,
    center_x: f32,
    center_y: f32,
    is_selected: bool,
    animate: bool,
    on_select: EventHandler<ResponseStar>,
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
    let animation_style = if animate {
        format!("animation: star-twinkle 4s ease-in-out {delay}s infinite;")
    } else {
        String::new()
    };
    let star_cloned = star.clone();

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
                    {animation_style}
                    {border_style}
                ",
                onclick: move |e| {
                    e.stop_propagation();
                    on_select.call(star_cloned.clone());
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

/// A long-lived `ResizeObserver` reports the size of the actual map container.
/// It works in browser, desktop WebView and Android WebView renderers through
/// Dioxus' supported `document::eval` bridge, instead of guessing from the
/// browser viewport or returning a native `1280x800` fallback.
const ELEMENT_SIZE_OBSERVER: &str = r#"
(() => {
  const element = document.querySelector('.starmap-root');
  if (!element || typeof ResizeObserver === 'undefined') return;

  let frame = 0;
  let lastWidth = 0;
  let lastHeight = 0;
  const publish = () => {
    frame = 0;
    const rect = element.getBoundingClientRect();
    // Sub-pixel changes can otherwise feed a ResizeObserver/render loop in
    // Firefox. The map does not need a backing extent larger than this.
    const width = Math.max(1, Math.min(4096, Math.round(rect.width)));
    const height = Math.max(1, Math.min(4096, Math.round(rect.height)));
    if (width === lastWidth && height === lastHeight) return;
    lastWidth = width;
    lastHeight = height;
    dioxus.send([width, height]);
  };
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(publish);
  };
  const observer = new ResizeObserver(schedule);
  observer.observe(element);
  schedule();
  // Keep the evaluator alive while the component is mounted. When its task is
  // cancelled by Dioxus, the bridge closes and this loop disconnects observer.
  (async () => {
    try { await dioxus.recv(); }
    finally {
      observer.disconnect();
      if (frame) cancelAnimationFrame(frame);
    }
  })();
})();
"#;

fn use_viewport_measurement() -> Signal<(f32, f32)> {
    let viewport = use_signal(|| (0.0_f32, 0.0_f32));

    use_effect(move || {
        let mut viewport = viewport;
        spawn(async move {
            let mut eval = dioxus::document::eval(ELEMENT_SIZE_OBSERVER);
            while let Ok((width, height)) = eval.recv::<(f64, f64)>().await {
                let width = (width as f32).clamp(1.0, 4_096.0);
                let height = (height as f32).clamp(1.0, 4_096.0);
                let previous = *viewport.read();
                if (previous.0 - width).abs() >= 1.0 || (previous.1 - height).abs() >= 1.0 {
                    viewport.set((width, height));
                }
            }
        });
    });

    viewport
}

async fn delay_sector_dispatch(batch_index: usize) {
    let delay_ms = FETCH_COOLDOWN_MS.saturating_mul(batch_index as u32);
    if delay_ms == 0 {
        return;
    }
    #[cfg(feature = "web")]
    gloo_timers::future::TimeoutFuture::new(delay_ms).await;
    #[cfg(not(feature = "web"))]
    tokio::time::sleep(std::time::Duration::from_millis(delay_ms.into())).await;
}

fn use_sync_sector_loading(
    game: Signal<StellarScene>,
    version: Signal<u64>,
    viewport: Signal<(f32, f32)>,
    active: Signal<bool>,
) {
    use_resource(move || async move {
        let _ = version();
        let is_active = active();
        let vp = *viewport.read();
        if !is_active || vp.0 <= 0.0 || vp.1 <= 0.0 {
            return;
        }

        // Claim the full batch synchronously before spawning requests. This is
        // what makes MAX_CONCURRENT_FETCHES a real global cap across renders.
        let pending = game.read().claim_sectors_to_fetch(vp);
        if pending.is_empty() {
            return;
        }
        let temperature = game.read().temperature();
        let bp_rp = game.read().bp_rp();
        let g_mag = game.read().g_mag();
        for (batch_index, (chunk, center)) in pending.into_iter().enumerate() {
            spawn(async move {
                delay_sector_dispatch(batch_index).await;
                let scene: StellarScene = game.read().clone();
                if !*active.peek() {
                    scene.fail_sector(chunk);
                    return;
                }
                let _ = scene
                    .fetch_claimed_sector(chunk, center, temperature, bp_rp, g_mag)
                    .await;
            });
        }
    });
}

fn use_sync_sector_eviction(
    game: Signal<StellarScene>,
    version: Signal<u64>,
    viewport: Signal<(f32, f32)>,
    active: Signal<bool>,
) {
    use_resource(move || async move {
        let _ = version();
        let is_active = active();
        let vp = *viewport.read();
        if !is_active || vp.0 <= 0.0 || vp.1 <= 0.0 {
            return;
        }
        game.read().evict_excess_sectors(vp);
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
    mouse_scene: Signal<(f32, f32)>,
}

fn use_star_map_interactions() -> InteractionState {
    let last_mouse = use_signal(|| (0.0_f32, 0.0_f32));
    let mouse_scene = use_signal(|| (0.0_f32, 0.0_f32));

    InteractionState {
        last_mouse,
        mouse_scene,
    }
}

#[component]
pub fn StarMap(
    game: Signal<StellarScene>,
    scene_stars: Vec<ResponseStar>,
    center_x: f32,
    center_y: f32,
    selected_id: Option<u32>,
    active: Signal<bool>,
    on_select: EventHandler<ResponseStar>,
) -> Element {
    let viewport = use_viewport_measurement();
    let version = use_stellar_scene_version();

    use_sync_sector_loading(game, version, viewport, active);
    use_sync_sector_eviction(game, version, viewport, active);

    let mut interact = use_star_map_interactions();

    let is_active = active();
    let snap = game.read().snapshot();
    let offset = snap.camera.offset;
    let zoom = snap.camera.zoom;
    let dragging = snap.camera.dragging;

    let sector_stars: Vec<ResponseStar> = snap.sector_stars.clone();
    let loading: HashSet<SectorKey> = snap.sector_loading.clone();
    let (starfield_small, starfield_medium, starfield_distant) = use_starfield_backgrounds();

    let handle_zoom = move |factor: f32| {
        let vp = *viewport.read();
        let g = game.read().clone();
        g.zoom_camera(vp, factor);
    };

    rsx! {
        div {
            class: if is_active { "starmap-root absolute inset-0 cursor-grab active:cursor-grabbing" } else { "starmap-root absolute inset-0 pointer-events-none" },
            onpointerdown: move |e| {
                let g = game.read().clone();
                g.set_dragging(true);
                interact.last_mouse.set((e.client_coordinates().x as f32, e.client_coordinates().y as f32));
            },
            onpointermove: move |e| {
                let nx = e.client_coordinates().x as f32;
                let ny = e.client_coordinates().y as f32;
                let vp = *viewport.read();

                if dragging {
                    let (lx, ly) = (interact.last_mouse)();
                    let g = game.read().clone();
                    g.pan_camera((nx - lx, ny - ly));
                    interact.last_mouse.set((nx, ny));
                }

                let mw = mouse_to_scene(nx, ny, vp, offset, zoom, center_x, center_y);
                let old_mw = (interact.mouse_scene)();

                let dist_sq = (mw.0 - old_mw.0).powi(2) + (mw.1 - old_mw.1).powi(2);
                if dist_sq > 4.0 {
                    interact.mouse_scene.set(mw);
                }

                if !dragging {
                    interact.last_mouse.set((nx, ny));
                }
            },
            onpointerup: move |_| {
                let g = game.read().clone();
                g.set_dragging(false);
            },
            onpointerleave: move |_| {
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
                        render_star("sector", &star, center_x, center_y, is_sel, zoom >= 0.35, on_select)
                    }
                }

                for star in scene_stars {
                    {
                        let is_sel = selected_id.map(|id| id == star.id).unwrap_or(false);
                        render_star("scene", &star, center_x, center_y, is_sel, true, on_select)
                    }
                }

                for key in loading.iter() {
                    {
                        render_loading_chunk(*key, center_x, center_y)
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

fn mouse_to_scene(
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
