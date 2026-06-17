use crate::api;
use crate::assets::FONT_SANS;
use crate::components::editor::sidebar::StarSidebar;
use crate::components::editor::star_map::StarMap;
use crate::components::editor::world_panel::{WorldCreator, WorldPicker};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use dioxus::prelude::*;
use lunar_structures::{
    GnnResponse, LoreMetadata, PinnResponse, PipelineRequest, PipelineResponse, RandomStarResponse,
    ResponseStar, StarLore, StellarMetadata, World, WorldListResponse, WorldSummary,
};
use std::io::Cursor;

#[cfg(feature = "web")]
const CAMERA_STORAGE_PREFIX: &str = "lunar.world.camera.";

#[cfg(feature = "web")]
fn save_camera_state(world_id: &str, offset: (f32, f32), zoom: f32) {
    if let Some(window) = web_sys::window() {
        if let Ok(Some(storage)) = window.local_storage() {
            let key = format!("{CAMERA_STORAGE_PREFIX}{world_id}");
            let value = format!(r#"{{"offset":[{},{}],"zoom":{}}}"#, offset.0, offset.1, zoom);
            let _ = storage.set_item(&key, &value);
        }
    }
}

#[cfg(feature = "web")]
fn load_camera_state(world_id: &str) -> Option<((f32, f32), f32)> {
    let window = web_sys::window()?;
    let storage = window.local_storage().ok()??;
    let key = format!("{CAMERA_STORAGE_PREFIX}{world_id}");
    let raw = storage.get_item(&key).ok()??;
    let parsed: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let arr = parsed.as_array()?;
    if arr.len() != 3 { return None; }
    let ox = arr[0].as_f64()? as f32;
    let oy = arr[1].as_f64()? as f32;
    let z = arr[2].as_f64()? as f32;
    Some(((ox, oy), z))
}

#[cfg(not(feature = "web"))]
fn save_camera_state(_world_id: &str, _offset: (f32, f32), _zoom: f32) {}

#[cfg(not(feature = "web"))]
fn load_camera_state(_world_id: &str) -> Option<((f32, f32), f32)> {
    None
}

fn build_star_lore(metadata: &StellarMetadata) -> StarLore {
    StarLore {
        designated_name: metadata.designated_name.clone(),
        category: metadata.category.clone(),
        visual_profile: metadata.spectral_class.clone(),
        system_lore: metadata.description.clone(),
        metadata: LoreMetadata {
            simulation_engine: "Lunar Pipeline v2.0".to_string(),
            data_source: "PINN + GNN + SIREN".to_string(),
            complexity_level: "High".to_string(),
        },
    }
}

fn encode_siren_to_png_b64(siren: &lunar_structures::SirenTextureResponse) -> Option<String> {
    let img = image::RgbImage::from_raw(siren.width, siren.height, siren.pixels.clone())?;
    let mut png_data = Cursor::new(Vec::new());
    if img.write_to(&mut png_data, image::ImageFormat::Png).is_ok() {
        Some(STANDARD.encode(png_data.into_inner()))
    } else {
        None
    }
}

fn process_pipeline_data(
    pipeline: &PipelineResponse,
) -> (Option<PinnResponse>, Option<StarLore>, Option<String>) {
    let pinn = pipeline.pinn.clone();
    let lore = build_star_lore(&pipeline.metadata);
    let texture = encode_siren_to_png_b64(&pipeline.siren);
    (Some(pinn), Some(lore), texture)
}

fn resolve_map_center_and_stars(
    active_world: Option<&World>,
    pregen: Option<&GnnResponse>,
    sector_center: Option<(f32, f32, f32)>,
) -> (f32, f32, f32, Vec<ResponseStar>) {
    if let Some(w) = active_world {
        return (w.center_x, w.center_y, w.center_z, w.stars.clone());
    }

    if let Some(resp) = pregen {
        if let Some(s) = resp.stars.first() {
            let (cx, cy, cz) = sector_center.unwrap_or((s.x, s.y, s.z));
            return (cx, cy, cz, resp.stars.clone());
        }
    }

    (0.0, 0.0, 0.0, Vec::new())
}

fn camera_storage_pair_for_world(world_id: &str) -> ((f32, f32), f32) {
    load_camera_state(world_id).unwrap_or(((0.0, 0.0), 1.0))
}

fn use_camera_persistence(
    active_world: Signal<Option<World>>,
    mut cam_offset: Signal<(f32, f32)>,
    mut cam_zoom: Signal<f32>,
) {
    let mut prev_world_id = use_signal(|| Option::<String>::None);

    use_effect(move || {
        if let Some(w) = active_world() {
            if prev_world_id.peek().as_ref() != Some(&w.id) {
                prev_world_id.set(Some(w.id.clone()));
                let (offset, zoom) = camera_storage_pair_for_world(&w.id);
                cam_offset.set(offset);
                cam_zoom.set(zoom.clamp(0.05, 15.0));
            }
        } else if prev_world_id.peek().is_some() {
            prev_world_id.set(None);
        }
    });

    use_effect(move || {
        if let Some(w) = active_world() {
            save_camera_state(&w.id, cam_offset(), cam_zoom());
        }
    });
}

fn use_pregen_resource(
    sector_center: Signal<Option<(f32, f32, f32)>>,
    temperature: Signal<f32>,
    bp_rp: Signal<f32>,
    g_mag: Signal<f32>,
) -> Resource<Option<GnnResponse>> {
    use_resource(move || {
        let center = sector_center();
        let t = temperature();
        let b = bp_rp();
        let g = g_mag();
        async move {
            if let Some((cx, cy, cz)) = center {
                api::fetch_sector_stars(cx, cy, cz, t, b, g).await.ok()
            } else {
                api::fetch_random_star(t)
                    .await
                    .ok()
                    .map(|r: RandomStarResponse| GnnResponse { stars: vec![r.star] })
            }
        }
    })
}

fn use_pipeline_resource(
    selected_star: Signal<Option<ResponseStar>>,
    active_world: Signal<Option<World>>,
) -> Resource<Option<PipelineResponse>> {
    use_resource(move || {
        let star_opt = selected_star();
        let world = active_world();
        async move {
            if let (Some(star), Some(_w)) = (star_opt, world) {
                let req = PipelineRequest {
                    x_pc: star.x,
                    y_pc: star.y,
                    z_pc: star.z,
                    bp_rp: 1.0,
                    g_mag: 10.0,
                    texture_size: 256,
                };
                api::fetch_pipeline(req).await.ok()
            } else {
                None
            }
        }
    })
}

fn reset_world_selection(
    mut active_world: Signal<Option<World>>,
    mut show_picker: Signal<bool>,
    mut sidebar_open: Signal<bool>,
    mut selected_star: Signal<Option<ResponseStar>>,
) {
    active_world.set(None);
    show_picker.set(true);
    sidebar_open.set(false);
    selected_star.set(None);
}

fn adopt_world(
    world: World,
    mut active_world: Signal<Option<World>>,
    mut show_picker: Signal<bool>,
    mut sidebar_open: Signal<bool>,
    mut selected_star: Signal<Option<ResponseStar>>,
) {
    active_world.set(Some(world));
    show_picker.set(false);
    sidebar_open.set(false);
    selected_star.set(None);
}

#[component]
fn WorldBadge(
    world: World,
    mut active_world: Signal<Option<World>>,
    mut show_picker: Signal<bool>,
    mut sidebar_open: Signal<bool>,
    mut selected_star: Signal<Option<ResponseStar>>,
) -> Element {
    rsx! {
        div {
            class: "absolute left-4 top-4 px-3 py-2 bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl text-white/70 text-[10px] font-bold uppercase tracking-[0.2em] cursor-pointer hover:bg-white/10 hover:text-white transition-colors shadow-lg pointer-events-auto flex items-center gap-2",
            onclick: move |_| {
                reset_world_selection(active_world, show_picker, sidebar_open, selected_star);
            },
            div { class: "w-1.5 h-1.5 rounded-full bg-amber-300" }
            span { "{world.name}" }
        }
    }
}

#[component]
fn WorldStatusBar(world: World) -> Element {
    rsx! {
        div {
            class: "absolute top-4 right-4 flex items-center gap-3 bg-black/40 backdrop-blur-xl border border-white/10 px-4 py-2 rounded-xl text-white/60 text-[10px] uppercase tracking-widest pointer-events-none",
            div { class: "w-3 h-3 border-2 border-emerald-400/30 border-t-emerald-400/80 rounded-full" }
            span { class: "text-emerald-300", "World Crystallized" }
            div { class: "w-px h-3 bg-white/20 mx-1" }
            span { "Center: [{world.center_x:.0}, {world.center_y:.0}, {world.center_z:.0}]" }
            div { class: "w-px h-3 bg-white/20 mx-1" }
            span { "{world.stars.len()} Stars" }
        }
    }
}

#[component]
fn OpenScannerButton(sidebar_open: Signal<bool>) -> Element {
    rsx! {
        div {
            class: "absolute left-4 top-4 px-5 py-2.5 bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl text-white/70 text-[10px] font-bold uppercase tracking-[0.2em] cursor-pointer hover:bg-white/10 hover:text-white transition-colors shadow-lg pointer-events-auto",
            style: "margin-top: 3.5rem;",
            onclick: move |_| sidebar_open.set(true),
            "Open Scanner"
        }
    }
}

#[component]
fn SidebarOrScanner(
    sidebar_open: bool,
    has_active_world: bool,
    selected: bool,
    selected_teff: f32,
    pinn_data: Option<PinnResponse>,
    lore_data: Option<StarLore>,
    siren_texture_b64: Option<String>,
    sidebar_open_signal: Signal<bool>,
) -> Element {
    if sidebar_open {
        rsx! {
            StarSidebar {
                selected,
                selected_teff,
                pinn_data,
                lore_data,
                siren_texture_b64,
            }
        }
    } else if has_active_world {
        rsx! { OpenScannerButton { sidebar_open: sidebar_open_signal } }
    } else {
        rsx! {}
    }
}

#[component]
fn EditorOverlays(
    show_picker: bool,
    show_creator: bool,
    mut worlds_resource: Resource<Option<WorldListResponse>>,
    mut active_world: Signal<Option<World>>,
    mut show_picker_signal: Signal<bool>,
    mut sidebar_open: Signal<bool>,
    mut selected_star: Signal<Option<ResponseStar>>,
    mut show_creator_signal: Signal<bool>,
    refresh_picker: EventHandler<()>,
) -> Element {
    if show_picker {
        let worlds_list: Vec<WorldSummary> = worlds_resource
            .read()
            .as_ref()
            .and_then(|opt| opt.as_ref())
            .map(|r| r.worlds.clone())
            .unwrap_or_default();
        let loading = worlds_resource.read().is_none();
        return rsx! {
            WorldPicker {
                worlds: worlds_list,
                loading,
                on_select: move |id: String| {
                    spawn(async move {
                        if let Ok(w) = api::get_world(&id).await {
                            adopt_world(w, active_world, show_picker_signal, sidebar_open, selected_star);
                        }
                    });
                },
                on_create: move |_| {
                    show_creator_signal.set(true);
                },
                on_delete: move |id: String| {
                    spawn(async move {
                        let _ = api::delete_world(&id).await;
                        refresh_picker.call(());
                    });
                },
            }
        };
    }

    if show_creator {
        return rsx! {
            WorldCreator {
                on_cancel: move |_| show_creator_signal.set(false),
                on_created: move |w: World| {
                    adopt_world(w, active_world, show_picker_signal, sidebar_open, selected_star);
                    refresh_picker.call(());
                },
            }
        };
    }

    rsx! {}
}

#[component]
pub fn EditorPage() -> Element {
    let mut sidebar_open = use_signal(|| false);
    let mut selected_star = use_signal(|| Option::<ResponseStar>::None);

    let show_picker = use_signal(|| true);
    let show_creator = use_signal(|| false);
    let active_world = use_signal(|| Option::<World>::None);

    let cam_offset = use_signal(|| (0.0_f32, 0.0_f32));
    let cam_zoom = use_signal(|| 1.0_f32);
    let cam_dragging = use_signal(|| false);

    let temperature = use_signal(|| 0.7_f32);
    let bp_rp = use_signal(|| 1.0_f32);
    let g_mag = use_signal(|| 10.0_f32);

    let mut sector_center = use_signal(|| Option::<(f32, f32, f32)>::None);
    let mut last_temp = use_signal(|| 0.7_f32);

    use_camera_persistence(active_world, cam_offset, cam_zoom);

    let pregen = use_pregen_resource(sector_center, temperature, bp_rp, g_mag);

    use_effect(move || {
        let temp = temperature();
        if *last_temp.peek() != temp {
            last_temp.set(temp);
            sector_center.set(None);
        }
    });

    use_effect(move || {
        if active_world().is_none() && sector_center.peek().is_none() {
            if let Some(Some(resp)) = pregen.read().as_ref() {
                if let Some(s) = resp.stars.first() {
                    sector_center.set(Some((s.x, s.y, s.z)));
                }
            }
        }
    });

    let mut worlds_resource = use_resource(move || async move {
        api::list_worlds().await.ok()
    });

    let mut refresh_picker = move || {
        worlds_resource.restart();
    };

    let pipeline_data = use_pipeline_resource(selected_star, active_world);

    let (pinn_data, lore_data, siren_texture_b64) = match pipeline_data.read().as_ref().and_then(|opt| opt.as_ref()) {
        Some(pipeline) => process_pipeline_data(pipeline),
        _ => (None, None, None),
    };

    let (center_x, center_y, center_z, stars) = resolve_map_center_and_stars(
        active_world().as_ref(),
        pregen.read().as_ref().and_then(|opt| opt.as_ref()),
        sector_center(),
    );

    let world_id = active_world().map(|w| w.id.clone());
    let world_temp = active_world().map(|w| w.temperature).unwrap_or_else(|| temperature());
    let world_bp_rp = active_world().map(|w| w.bp_rp).unwrap_or_else(|| bp_rp());
    let world_g_mag = active_world().map(|w| w.g_mag).unwrap_or_else(|| g_mag());
    let selected_teff = selected_star().map(|s| s.temperature_k).unwrap_or(5778.0);

    rsx! {
        div {
            class: "h-screen w-screen bg-[#050505] overflow-hidden relative select-none [background-image:radial-gradient(ellipse_at_center,_rgba(255,255,255,0.015)_0%,_transparent_70%)]",
            style: "font-family: {FONT_SANS};",

            StarMap {
                stars,
                center_x,
                center_y,
                center_z,
                temperature: world_temp,
                bp_rp: world_bp_rp,
                g_mag: world_g_mag,
                world_id: world_id.clone(),
                selected_star,
                cam_offset,
                cam_zoom,
                cam_dragging,
                on_select: move |star: ResponseStar| {
                    selected_star.set(Some(star));
                    sidebar_open.set(true);
                }
            }

            if let Some(w) = active_world() {
                WorldBadge {
                    world: w,
                    active_world,
                    show_picker,
                    sidebar_open,
                    selected_star,
                }
            }

            SidebarOrScanner {
                sidebar_open: sidebar_open(),
                has_active_world: active_world().is_some(),
                selected: selected_star().is_some(),
                selected_teff,
                pinn_data,
                lore_data,
                siren_texture_b64,
                sidebar_open_signal: sidebar_open,
            }

            if let Some(w) = active_world() {
                WorldStatusBar { world: w }
            }

            EditorOverlays {
                show_picker: show_picker(),
                show_creator: show_creator(),
                worlds_resource,
                active_world,
                show_picker_signal: show_picker,
                sidebar_open,
                selected_star,
                show_creator_signal: show_creator,
                refresh_picker: move |_| refresh_picker(),
            }
        }
    }
}
