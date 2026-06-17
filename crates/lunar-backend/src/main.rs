use axum::{
    extract::Query,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use lunar_utils::*;

use lunar_structures::{
    PipelineRequest, PipelineResponse, RandomStarRequest, RandomStarResponse,
    SirenTextureRequest, SirenTextureResponse, StarDescriptionPayload, StarLore,
    LoreMetadata, ResponseStar, PinnResponse,
};
use lunar_utils::env::{get_url, get_worlds_dir};
use crate::ai::{generate_hybrid_metadata, generate_random_inputs, get_gnn, get_lore_cache, get_pinn, gnn_infer, warmup_models, RandomStellarInputs, StarFeatures};

#[cfg(feature = "siren")]
use crate::ai::{get_siren, siren_generate_texture};

pub mod ai;
pub mod worlds;

use crate::worlds::{
    calculate_absolute_magnitude, infer_pinn_async, WorldStore,
};

async fn generate_siren_pixels(
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
        png[0] = 0x89; png[1] = 0x50; png[2] = 0x4E; png[3] = 0x47;
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
        teff.max(0.0), rad.max(0.0), mass.max(0.0), lum.max(0.0),
        0.5, lore.as_deref(),
    );

    Json(StarLore {
        designated_name: meta.designated_name,
        category: format!("{}-type {}", meta.spectral_class, meta.category),
        visual_profile: meta.description.clone(),
        system_lore: meta.description,
        metadata: LoreMetadata {
            simulation_engine: "LunarSim v1.0".to_string(),
            data_source: "Procedurally Generated".to_string(),
            complexity_level: "High".to_string(),
        },
    })
}

async fn pipeline_handler(Json(payload): Json<PipelineRequest>) -> Json<PipelineResponse> {
    let [teff, rad, mass, lum] = infer_pinn_async(
        payload.x_pc,
        payload.y_pc,
        payload.z_pc,
        payload.bp_rp,
        payload.g_mag,
    )
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
    .unwrap_or_else(|| vec![0; (payload.texture_size as usize) * (payload.texture_size as usize) * 3]);

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
        pinn: PinnResponse { temperature_k: teff, radius_solar: rad, mass_solar: mass, luminosity_solar: lum },
        siren: siren_texture,
        metadata: meta,
    })
}

async fn random_star(Json(payload): Json<RandomStarRequest>) -> Json<RandomStarResponse> {
    let entropy = payload.entropy_temperature;
    let pinn = get_pinn().await;

    let inputs = tokio::task::spawn_blocking(move || {
        generate_random_inputs(entropy, &pinn.norm)
    }).await.unwrap_or_else(|_| RandomStellarInputs {
        x_pc: 0.0, y_pc: 0.0, z_pc: 0.0, bp_rp: 1.0, g_mag: 10.0,
    });

    let [teff, rad, mass, lum] = infer_pinn_async(
        inputs.x_pc,
        inputs.y_pc,
        inputs.z_pc,
        inputs.bp_rp,
        inputs.g_mag,
    )
    .await;

    let mg = calculate_absolute_magnitude(inputs.x_pc, inputs.y_pc, inputs.z_pc, inputs.g_mag);

    let gnn_opt = get_gnn().await;
    let lore = get_lore_cache().await;

    let metadata = generate_hybrid_metadata(
        teff.max(0.0), rad.max(0.0), mass.max(0.0), lum.max(0.0),
        entropy, lore.as_deref(),
    );

    let vel = if let Some(gnn) = gnn_opt {
        let stars = vec![StarFeatures {
            coords: [inputs.x_pc, inputs.y_pc, inputs.z_pc],
            log_teff: teff.max(0.01).log10(),
            log_rad: rad.max(0.01).log10(),
            log_mass: mass.max(0.01).log10(),
            log_lum: lum.max(0.01).log10(),
            mg,
        }];
        let stars_clone = stars.clone();
        let velocities = tokio::task::spawn_blocking(move || {
            gnn_infer(&gnn, &stars_clone, 8, entropy)
        }).await.unwrap_or_default();

        velocities.first().copied().unwrap_or([0.0, 0.0, 0.0])
    } else {
        [0.0, 0.0, 0.0]
    };

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
    let port: u16 = {
        let args: Vec<String> = std::env::args().collect();
        args.windows(2)
            .find(|w| w[0] == "--port" || w[0] == "-p")
            .and_then(|w| w[1].parse().ok())
            .unwrap_or(25255)
    };

    println!("Warming up neural network models...");
    warmup_models().await;
    println!("Models ready. Starting server on 127.0.0.1:{port}");

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let worlds_dir = get_worlds_dir();
    let _ = std::fs::create_dir_all(&worlds_dir);
    let world_store = Arc::new(WorldStore::new(worlds_dir));

    let app = Router::new()
        .route("/pinn", post(worlds::pinn))
        .route("/gnn", post(worlds::gnn))
        .route("/sector/stars", post(worlds::sector_stars))
        .route("/description", post(description))
        .route("/random_star", post(random_star))
        .route("/siren/texture", post(siren_texture))
        .route("/siren/png", get(siren_png))
        .route("/pipeline", post(pipeline_handler))
        .route("/worlds", get(worlds::list_worlds))
        .route("/worlds/create", post(worlds::create_world))
        .route("/worlds/{id}", get(worlds::get_world).delete(worlds::delete_world))
        .layer(cors)
        .with_state(world_store);

    let listener = tokio::net::TcpListener::bind(get_url()).await?;
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}
