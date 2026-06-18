use crate::assets::FONT_SANS;
use crate::components::editor::sidebar::StarSidebar;
use crate::components::editor::star_map::StarMap;
use crate::components::editor::world_panel::{WorldCreator, WorldPicker};
use crate::game_state::{
    hydrate_world_camera_from_storage, use_provide_world_camera_persistence, use_game,
    use_game_snapshot, use_persist_world_camera, use_pipeline_snapshot,
    use_world_id_change,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use dioxus::prelude::*;
use lunar_game_backend::{Game, GameSnapshot};
use lunar_structures::{
    PinnResponse, PipelineResponse, ResponseStar, StarLore, StellarMetadata, World,
    WorldSummary,
};
use std::io::Cursor;
use tracing::warn;

fn build_star_lore(metadata: &StellarMetadata) -> StarLore {
    StarLore {
        designated_name: metadata.designated_name.clone(),
        category: format!("{}-type {}", metadata.spectral_class, metadata.category),
        visual_profile: metadata.spectral_class.clone(),
        system_lore: metadata.description.clone(),
    }
}

fn encode_siren_to_png_b64(siren: &lunar_structures::SirenTextureResponse) -> Option<String> {
    let img = image::RgbImage::from_raw(siren.width, siren.height, siren.pixels.clone())?;
    let mut png_data = Cursor::new(Vec::new());
    img.write_to(&mut png_data, image::ImageFormat::Png).ok()?;
    Some(STANDARD.encode(png_data.into_inner()))
}

fn process_pipeline_data(
    pipeline: &PipelineResponse,
) -> (Option<PinnResponse>, Option<StarLore>, Option<String>) {
    (
        Some(pipeline.pinn.clone()),
        Some(build_star_lore(&pipeline.metadata)),
        encode_siren_to_png_b64(&pipeline.siren),
    )
}

fn resolve_map_inputs(snap: &GameSnapshot) -> (f32, f32, f32, Vec<ResponseStar>) {
    if let Some(w) = &snap.active_world {
        return (w.center_x, w.center_y, w.center_z, w.stars.clone());
    }
    if let Some(resp) = &snap.pregen {
        if let Some(s) = resp.stars.first() {
            let (cx, cy, cz) = snap.sector_center.unwrap_or((s.x, s.y, s.z));
            return (cx, cy, cz, resp.stars.clone());
        }
    }
    (0.0, 0.0, 0.0, Vec::new())
}

fn use_sync_world_camera_loader(game: Signal<Game>) {
    use_world_id_change(move |_prev, current| {
        if let Some(id) = current {
            let g = game.read().clone();
            hydrate_world_camera_from_storage(&g, id);
            if let Err(e) = g.apply_world_camera(id) {
                warn!(error = %e, "apply_world_camera failed");
            }
        }
    });
}

fn use_sync_worlds_list(game: Signal<Game>, refresh_tick: Signal<u32>) {
    use_resource(move || async move {
        let _ = refresh_tick();
        let g = game.read().clone();
        if let Err(e) = g.refresh_worlds().await {
            warn!(error = %e, "refresh_worlds failed");
        }
    });
}

fn use_sync_sector_center_alignment(game: Signal<Game>, version: Signal<u64>) {
    use_resource(move || async move {
        let _ = version();
        let g = game.read().clone();
        let has_no_world_or_center = g.active_world().is_none() && g.sector_center().is_none();
        if has_no_world_or_center {
            if let Some(s) = g.pregen().and_then(|p| p.stars.first().cloned()) {
                if let Err(e) = g.set_sector_center(Some((s.x, s.y, s.z))) {
                    warn!(error = %e, "set_sector_center failed");
                }
            }
        }
    });
}

fn use_sync_temperature_tracker(game: Signal<Game>, version: Signal<u64>) {
    use_resource(move || async move {
        let _ = version();
        let g = game.read().clone();
        let temp = g.temperature();
        let last = g.last_temp();
        if (temp - last).abs() > f32::EPSILON {
            g.set_last_temp(temp);
            if let Err(e) = g.set_sector_center(None) {
                warn!(error = %e, "set_sector_center failed");
            }
            g.set_pregen(None);
        }
    });
}

fn use_sync_pregen_stars(game: Signal<Game>, version: Signal<u64>) {
    use_resource(move || async move {
        let _ = version();
        let g = game.read().clone();
        if g.active_world().is_none() && g.pregen().is_none() {
            if let Err(e) = g.fetch_pregen().await {
                warn!(error = %e, "fetch_pregen failed");
            }
        }
    });
}

fn use_sync_star_pipeline(game: Signal<Game>, version: Signal<u64>) {
    use_resource(move || async move {
        let _ = version();
        let g = game.read().clone();
        if let Some(star) = g.selected_star() {
            if g.pipeline().is_none() {
                if let Err(e) = g.fetch_pipeline(star).await {
                    warn!(error = %e, "fetch_pipeline failed");
                }
            }
        }
    });
}

fn use_editor_synchronization(
    game: Signal<Game>,
    version: Signal<u64>,
    refresh_tick: Signal<u32>,
) {
    use_provide_world_camera_persistence();
    use_persist_world_camera();

    use_sync_world_camera_loader(game);
    use_sync_worlds_list(game, refresh_tick);
    use_sync_sector_center_alignment(game, version);
    use_sync_temperature_tracker(game, version);
    use_sync_pregen_stars(game, version);
    use_sync_star_pipeline(game, version);
}

async fn delete_world_action(g: Game, id: String, mut refresh_tick: Signal<u32>) {
    if let Err(e) = g.delete_world(&id).await {
        warn!(error = %e, "delete_world failed");
    }
    refresh_tick.set(refresh_tick() + 1);
}

async fn load_world_action(g: Game, id: String) {
    if let Err(e) = g.load_world(&id).await {
        warn!(error = %e, "load_world failed");
    }
}

fn handle_world_creation(
    g: Game,
    w: World,
    mut show_creator: Signal<bool>,
    mut show_picker: Signal<bool>,
    mut sidebar_open: Signal<bool>,
) {
    g.adopt_world(Some(w));
    show_creator.set(false);
    show_picker.set(false);
    sidebar_open.set(false);
}

#[component]
fn WorldBadge(world: World, game: Signal<Game>, show_picker: Signal<bool>) -> Element {
    rsx! {
        div {
            class: "absolute left-4 top-4 px-3 py-2 bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl text-white/70 text-[10px] font-bold uppercase tracking-[0.2em] cursor-pointer hover:bg-white/10 hover:text-white transition-colors shadow-lg pointer-events-auto flex items-center gap-2",
            onclick: move |_| {
                game.read().clone().clear_active_world();
                show_picker.set(true);
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
            span { "Center: [{world.center_x:.0}, {world.center_y:.0}, {world.center_z:.0}]" }
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
    game: Signal<Game>,
    open: Signal<bool>,
    selected: bool,
    selected_teff: f32,
    pinn_data: Option<PinnResponse>,
    lore_data: Option<StarLore>,
    siren_texture_b64: Option<String>,
) -> Element {
    if open() {
        rsx! {
            StarSidebar {
                selected,
                selected_teff,
                pinn_data,
                lore_data,
                siren_texture_b64,
            }
        }
    } else if game.read().active_world().is_some() {
        rsx! { OpenScannerButton { sidebar_open: open } }
    } else {
        rsx! {}
    }
}

#[component]
fn EditorOverlays(
    show_picker: bool,
    show_creator: bool,
    worlds_list: Vec<WorldSummary>,
    worlds_loading: bool,
    on_select_world: EventHandler<String>,
    on_create_world: EventHandler<()>,
    on_delete_world: EventHandler<String>,
    on_cancel_create: EventHandler<()>,
    on_created_world: EventHandler<World>,
) -> Element {
    if show_creator {
        rsx! {
            WorldCreator {
                on_cancel: move |_| on_cancel_create.call(()),
                on_created: move |w: World| on_created_world.call(w),
            }
        }
    } else if show_picker {
        rsx! {
            WorldPicker {
                worlds: worlds_list,
                loading: worlds_loading,
                on_select: move |id| on_select_world.call(id),
                on_create: move |_| on_create_world.call(()),
                on_delete: move |id| on_delete_world.call(id),
            }
        }
    } else {
        rsx! {}
    }
}

#[component]
fn ActiveWorldOverlays(
    active_world: Option<World>,
    game: Signal<Game>,
    show_picker: Signal<bool>,
) -> Element {
    let Some(w) = active_world else {
        return rsx! {};
    };
    rsx! {
        WorldBadge { world: w.clone(), game, show_picker }
        WorldStatusBar { world: w.clone() }
    }
}


#[component]
pub fn Editor() -> Element {
    let game = use_game();
    let version = use_context::<Signal<u64>>();

    let mut sidebar_open = use_signal(|| false);
    let show_picker = use_signal(|| true);
    let mut show_creator = use_signal(|| false);
    let refresh_tick = use_signal(|| 0u32);

    use_editor_synchronization(game, version, refresh_tick);

    let snap = use_game_snapshot();
    let worlds_list = snap.worlds.clone();
    let worlds_loading = false;

    let pipeline_data = use_pipeline_snapshot();
    let (pinn_data, lore_data, siren_texture_b64) = pipeline_data
        .as_ref()
        .map(process_pipeline_data)
        .unwrap_or_default();

    let (center_x, center_y, _center_z, world_stars) = resolve_map_inputs(&snap);
    let selected_id = snap.selected_star.as_ref().map(|s| s.id);
    let selected_teff = snap
        .selected_star
        .as_ref()
        .map(|s| s.temperature_k)
        .unwrap_or(5778.0);

    let on_select_star = move |star: ResponseStar| {
        game.read().clone().select_star(Some(star));
        sidebar_open.set(true);
    };

    let on_create_world = move |_: ()| {
        show_creator.set(true);
    };

    let on_cancel_create = move |_: ()| {
        show_creator.set(false);
    };

    let on_created_world = move |w| {
        handle_world_creation(game.read().clone(), w, show_creator, show_picker, sidebar_open);
    };

    let on_delete_world = move |id| {
        spawn(delete_world_action(game.read().clone(), id, refresh_tick));
    };

    let on_select_world = move |id| {
        spawn(load_world_action(game.read().clone(), id));
    };

    rsx! {
        div {
            class: "h-screen w-screen bg-[#050505] overflow-hidden relative select-none [background-image:radial-gradient(ellipse_at_center,_rgba(255,255,255,0.015)_0%,_transparent_70%)]",
            style: "font-family: {FONT_SANS};",

            StarMap {
                game,
                world_stars,
                center_x,
                center_y,
                selected_id,
                on_select: on_select_star,
            }

            ActiveWorldOverlays { active_world: snap.active_world.clone(), game, show_picker }

            SidebarOrScanner {
                game,
                open: sidebar_open,
                selected: snap.selected_star.is_some(),
                selected_teff,
                pinn_data,
                lore_data,
                siren_texture_b64,
            }

            EditorOverlays {
                show_picker: show_picker(),
                show_creator: show_creator(),
                worlds_list,
                worlds_loading,
                on_select_world,
                on_create_world,
                on_delete_world,
                on_cancel_create,
                on_created_world,
            }
        }
    }
}
