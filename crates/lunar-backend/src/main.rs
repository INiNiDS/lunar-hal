use axum::{
    Json, Router,
    extract::Query,
    routing::{get, patch, post},
};
use lunar_utils::*;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::broadcast;
use tower_http::cors::{Any, CorsLayer};

use crate::ai::{
    RandomStellarInputs, generate_hybrid_metadata, generate_random_inputs, get_lore_cache,
    get_pinn, warmup_models,
};
use lunar_structures::{
    PinnResponse, PipelineRequest, PipelineResponse, RandomStarRequest, RandomStarResponse,
    ResponseStar, SirenTextureRequest, SirenTextureResponse, StarDescriptionPayload, StarLore,
};
use lunar_utils::env::{get_gallery_dir, get_host, get_port, get_scenes_dir};

#[cfg(feature = "siren")]
use crate::ai::{get_siren, siren_generate_texture};

pub mod ai;
pub mod gallery;
pub mod scenes;

use crate::ai::PinnInputs;
use crate::gallery::GalleryStore;
use crate::scenes::{SceneStore, calculate_absolute_magnitude, infer_pinn_async};
use lunar_structures::SceneEvent;

#[derive(Clone)]
pub struct AppState {
    pub scenes: Arc<SceneStore>,
    pub gallery: Arc<GalleryStore>,
    pub scene_events: broadcast::Sender<SceneEvent>,
}


pub(crate) async fn generate_siren_pixels(
    width: u32,
    height: u32,
    bp_rp: f32,
    m_g: f32,
    log_teff: f32,
) -> Option<Vec<u8>> {
    let siren = get_siren().await?;
    tokio::task::spawn_blocking(move || {
        siren_generate_texture(&siren, width, height, bp_rp, m_g, log_teff)
    })
    .await
    .ok()
}

async fn siren_texture(Json(payload): Json<SirenTextureRequest>) -> Json<SirenTextureResponse> {
    let pixels = generate_siren_pixels(
        payload.width,
        payload.height,
        payload.bp_rp,
        payload.m_g,
        payload.log_teff,
    )
    .await
    .unwrap_or_else(|| vec![0; (payload.width as usize) * (payload.height as usize) * 3]);

    Json(SirenTextureResponse {
        width: payload.width,
        height: payload.height,
        pixels,
    })
}

#[derive(Deserialize)]
struct SirenPngParams {
    width: Option<u32>,
    height: Option<u32>,
    bp_rp: Option<f32>,
    m_g: Option<f32>,
    temperature_k: Option<f32>,
}

async fn siren_png(Query(params): Query<SirenPngParams>) -> Vec<u8> {
    let w = params.width.unwrap_or(256);
    let h = params.height.unwrap_or(256);
    let bp_rp = params.bp_rp.unwrap_or(1.5);
    let m_g = params.m_g.unwrap_or(5.0);
    let teff = params.temperature_k.unwrap_or(5778.0);

    let log_teff = if teff > 0.0 { teff.log10() } else { 3.75 };

    let Some(rgb) = generate_siren_pixels(w, h, bp_rp, m_g, log_teff).await else {
        let mut png = vec![0u8; 8];
        png[0] = 0x89;
        png[1] = 0x50;
        png[2] = 0x4E;
        png[3] = 0x47;
        return png;
    };

    encode_rgb_png(&rgb, w, h)
}

async fn description(Json(payload): Json<StarDescriptionPayload>) -> Json<StarLore> {
    let teff = payload.pinn_payload.temperature_k;
    let rad = payload.pinn_payload.radius_solar;
    let mass = payload.pinn_payload.mass_solar;
    let lum = payload.pinn_payload.luminosity_solar;

    let lore = get_lore_cache().await;
    let meta = generate_hybrid_metadata(
        teff.max(0.0),
        rad.max(0.0),
        mass.max(0.0),
        lum.max(0.0),
        0.5,
        lore.as_deref(),
    );

    Json(StarLore {
        designated_name: meta.designated_name,
        category: format!("{}-type {}", meta.spectral_class, meta.category),
        visual_profile: meta.description.clone(),
        system_lore: meta.description,
    })
}

async fn pipeline_handler(Json(payload): Json<PipelineRequest>) -> Json<PipelineResponse> {
    let [teff, rad, mass, lum] = infer_pinn_async(PinnInputs {
        position: [payload.x_pc, payload.y_pc, payload.z_pc],
        bp_rp: payload.bp_rp,
        g_mag: payload.g_mag,
    })
    .await;

    let m_g = calculate_absolute_magnitude(payload.x_pc, payload.y_pc, payload.z_pc, payload.g_mag);
    let log_teff = if teff > 0.0 { teff.log10() } else { 3.75 };

    let pixels = generate_siren_pixels(
        payload.texture_size,
        payload.texture_size,
        payload.bp_rp,
        m_g,
        log_teff,
    )
    .await
    .unwrap_or_else(|| {
        vec![0; (payload.texture_size as usize) * (payload.texture_size as usize) * 3]
    });

    let siren_texture = SirenTextureResponse {
        width: payload.texture_size,
        height: payload.texture_size,
        pixels,
    };

    let lore = get_lore_cache().await;
    let meta = generate_hybrid_metadata(
        teff.max(0.0),
        rad.max(0.0),
        mass.max(0.0),
        lum.max(0.0),
        0.5,
        lore.as_deref(),
    );

    Json(PipelineResponse {
        pinn: PinnResponse {
            temperature_k: teff,
            radius_solar: rad,
            mass_solar: mass,
            luminosity_solar: lum,
        },
        siren: siren_texture,
        metadata: meta,
    })
}

async fn random_star(Json(payload): Json<RandomStarRequest>) -> Json<RandomStarResponse> {
    let entropy = payload.entropy_temperature;
    let pinn = get_pinn().await;

    let inputs = tokio::task::spawn_blocking(move || generate_random_inputs(entropy, &pinn.norm))
        .await
        .unwrap_or(RandomStellarInputs {
            x_pc: 0.0,
            y_pc: 0.0,
            z_pc: 0.0,
            bp_rp: 1.0,
            g_mag: 10.0,
        });

    let [teff, rad, mass, lum] = infer_pinn_async(PinnInputs {
        position: [inputs.x_pc, inputs.y_pc, inputs.z_pc],
        bp_rp: inputs.bp_rp,
        g_mag: inputs.g_mag,
    })
    .await;

    let lore = get_lore_cache().await;

    let metadata = generate_hybrid_metadata(
        teff.max(0.0),
        rad.max(0.0),
        mass.max(0.0),
        lum.max(0.0),
        entropy,
        lore.as_deref(),
    );

    // Stage 6.6: single-node GNN is excluded from production — one star
    // has no neighbors, so there is no valid group to run the model on.
    // Zero velocity is the explicit fallback (same as missing-GNN),
    // never a self-loop forward.
    let vel = [0.0, 0.0, 0.0];

    let star = ResponseStar {
        id: 0,
        x: inputs.x_pc,
        y: inputs.y_pc,
        z: inputs.z_pc,
        temperature_k: teff,
        radius: rad,
        mass,
        luminosity: lum,
        description: metadata.description,
        name: metadata.designated_name,
        type_hint: metadata.spectral_class,
        velocity_vector: vel,
    };

    Json(RandomStarResponse {
        bp_rp: inputs.bp_rp,
        g_mag: inputs.g_mag,
        x_pc: inputs.x_pc,
        y_pc: inputs.y_pc,
        z_pc: inputs.z_pc,
        star,
    })
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let host = get_host();
    let port = get_port();

    println!("Warming up neural network models...");
    warmup_models().await;
    println!("Models ready. Starting server on {host}:{port}");

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let scenes_dir = get_scenes_dir();
    let gallery_dir = get_gallery_dir();
    let _ = std::fs::create_dir_all(&scenes_dir);
    let _ = std::fs::create_dir_all(&gallery_dir);
    let (scene_events, _) = broadcast::channel(256);
    let state = AppState {
        scenes: Arc::new(SceneStore::new(scenes_dir)),
        gallery: Arc::new(GalleryStore::new(gallery_dir)),
        scene_events,
    };

    let app = Router::new()
        .route("/pinn", post(scenes::pinn))
        .route("/gnn", post(scenes::gnn))
        .route("/sector/stars", post(scenes::sector_stars))
        .route("/description", post(description))
        .route("/random_star", post(random_star))
        .route("/siren/texture", post(siren_texture))
        .route("/siren/png", get(siren_png))
        .route("/pipeline", post(pipeline_handler))
        .route("/scenes", get(scenes::list_scenes))
        .route("/scenes/create", post(scenes::create_scene))
        .route(
            "/scenes/{id}",
            get(scenes::get_scene).delete(scenes::delete_scene),
        )
        .route("/scenes/{id}/events", get(scenes::scene_events))
        .route("/scenes/{id}/stars/generate", post(scenes::generate_scene_stars))
        .route("/scenes/{id}/stars", post(scenes::create_scene_star))
        .route(
            "/scenes/{id}/stars/{star_id}",
            patch(scenes::update_scene_star).delete(scenes::delete_scene_star),
        )
        .route("/scenes/{id}/clear", post(scenes::clear_scene))
        .route(
            "/gallery/stars",
            get(gallery::list_gallery_stars).post(gallery::create_gallery_star),
        )
        .route(
            "/gallery/stars/{id}",
            get(gallery::get_gallery_star)
                .patch(gallery::update_gallery_star)
                .delete(gallery::delete_gallery_star),
        )
        .route("/gallery/stars/{id}/texture.png", get(gallery::gallery_texture))
        .route("/gallery/stars/{id}/thumbnail", get(gallery::gallery_thumbnail))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}")).await?;
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}
