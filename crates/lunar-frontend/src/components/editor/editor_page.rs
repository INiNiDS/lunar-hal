use crate::assets::FONT_SANS;
use crate::components::editor::sidebar::StarSidebar;
use crate::components::editor::star_map::StarMap;
use crate::components::editor::world_panel::{WorldCreator, WorldPicker};
use crate::game_state::{
    hydrate_world_camera_from_storage, provide_world_camera_persistence, use_game,
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

fn resolve_map_inputs(snap: &GameSnapshot) -> (f32, f32, f32, Vec<ResponseStar>) {
    if let Some(w) = &snap.active_world {
        return (w.center_x, w.center_y, w.center_z, w.stars.clone());
    }
    if let Some(resp) = &snap.pregen
        && let Some(s) = resp.stars.first()
    {
        let (cx, cy, cz) = snap.sector_center.unwrap_or((s.x, s.y, s.z));
        return (cx, cy, cz, resp.stars.clone());
    }
    (0.0, 0.0, 0.0, Vec::new())
}

#[component]
fn WorldBadge(world: World, game: Signal<Game>, show_picker: Signal<bool>) -> Element {
    rsx! {
        div {
            class: "absolute left-4 top-4 px-3 py-2 bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl text-white/70 text-[10px] font-bold uppercase tracking-[0.2em] cursor-pointer hover:bg-white/10 hover:text-white transition-colors shadow-lg pointer-events-auto flex items-center gap-2",
            onclick: move |_| {
                let g = game.read().clone();
                g.clear_active_world();
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
    let has_active_world = game.read().active_world().is_some();
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
    } else if has_active_world {
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
        return rsx! {
            WorldCreator {
                on_cancel: move |_| on_cancel_create.call(()),
                on_created: move |w: World| on_created_world.call(w),
            }
        };
    }
    if show_picker {
        return rsx! {
            WorldPicker {
                worlds: worlds_list,
                loading: worlds_loading,
                on_select: move |id: String| on_select_world.call(id),
                on_create: move |_| on_create_world.call(()),
                on_delete: move |id: String| on_delete_world.call(id),
            }
        };
    }
    rsx! {}
}


#[nah::high_complexity]
#[component]
pub fn Editor() -> Element {
    let game = use_game();
    let version = use_context::<Signal<u64>>();

    let mut sidebar_open = use_signal(|| false);
    let mut show_picker = use_signal(|| true);
    let mut show_creator = use_signal(|| false);
    let mut refresh_tick = use_signal(|| 0u32);

    provide_world_camera_persistence();
    use_persist_world_camera();
    use_world_id_change(move |_prev, current| {
        if let Some(id) = current {
            let g: Game = game.read().clone();
            hydrate_world_camera_from_storage(&g, id);
            if let Err(e) = g.apply_world_camera(id) {
                warn!(error = %e, "apply_world_camera failed");
            }
        }
    });

    use_resource(move || async move {
        let _ = refresh_tick();
        let g: Game = game.read().clone();
        if let Err(e) = g.refresh_worlds().await {
            warn!(error = %e, "refresh_worlds failed");
        }
    });

    use_resource(move || async move {
        let _ = version();
        let g: Game = game.read().clone();
        if g.active_world().is_none()
            && g.sector_center().is_none()
            && g.pregen().is_some()
            && let Some(s) = g.pregen().and_then(|p| p.stars.first().cloned())
        {
            if let Err(e) = g.set_sector_center(Some((s.x, s.y, s.z))) {
                warn!(error = %e, "set_sector_center failed");
            }
        }
    });

    use_resource(move || async move {
        let _ = version();
        let g: Game = game.read().clone();
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

    use_resource(move || async move {
        let _ = version();
        let g: Game = game.read().clone();
        if g.active_world().is_none() && g.pregen().is_none() {
            if let Err(e) = g.fetch_pregen().await {
                warn!(error = %e, "fetch_pregen failed");
            }
        }
    });

    use_resource(move || async move {
        let _ = version();
        let g: Game = game.read().clone();
        if let Some(star) = g.selected_star()
            && g.pipeline().is_none()
        {
            if let Err(e) = g.fetch_pipeline(star).await {
                warn!(error = %e, "fetch_pipeline failed");
            }
        }
    });

    let snap = use_game_snapshot();
    let worlds_list = snap.worlds.clone();
    let worlds_loading = false;

    let pipeline_data = use_pipeline_snapshot();
    let (pinn_data, lore_data, siren_texture_b64) = match &pipeline_data {
        Some(p) => process_pipeline_data(p),
        None => (None, None, None),
    };

    let (center_x, center_y, _center_z, world_stars) = resolve_map_inputs(&snap);
    let selected_id = snap.selected_star.as_ref().map(|s| s.id);
    let selected_teff = snap
        .selected_star
        .as_ref()
        .map(|s| s.temperature_k)
        .unwrap_or(5778.0);

    let on_select_star = move |star: ResponseStar| {
        let g = game.read().clone();
        g.select_star(Some(star));
        sidebar_open.set(true);
    };

    let on_create_world = move |_: ()| {
        show_creator.set(true);
    };

    let on_cancel_create = move |_: ()| {
        show_creator.set(false);
    };

    let on_created_world = move |w: World| {
        let g: Game = game.read().clone();
        g.adopt_world(Some(w));
        show_creator.set(false);
        show_picker.set(false);
        sidebar_open.set(false);
    };

    let on_delete_world = move |id: String| {
        let g: Game = game.read().clone();
        spawn(async move {
            if let Err(e) = g.delete_world(&id).await {
                warn!(error = %e, "delete_world failed");
            }
            refresh_tick.set(refresh_tick() + 1);
        });
    };

    let on_select_world = move |id: String| {
        let g: Game = game.read().clone();
        spawn(async move {
            if let Err(e) = g.load_world(&id).await {
                warn!(error = %e, "load_world failed");
            }
        });
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

            if let Some(w) = &snap.active_world {
                WorldBadge { world: w.clone(), game, show_picker }
            }

            SidebarOrScanner {
                game,
                open: sidebar_open,
                selected: snap.selected_star.is_some(),
                selected_teff,
                pinn_data,
                lore_data,
                siren_texture_b64,
            }

            if let Some(w) = &snap.active_world {
                WorldStatusBar { world: w.clone() }
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
