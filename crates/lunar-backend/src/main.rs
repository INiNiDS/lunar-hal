pub mod ai;

use axum::{routing::post, Json, Router};
use anyhow::Result;
use crate::ai::{get_gnn, get_pinn, gnn_infer, pinn_infer, generate_stochastic_metadata, StarFeatures, warmup_models};
use lunar_structures::{GnnRequest, GnnResponse, ImageRequest, ImageResponse, PinnRequest, PinnResponse, ResponseStar, StarDescriptionPayload, StarLore, LoreMetadata};

async fn pinn(Json(payload): Json<PinnRequest>) -> Json<PinnResponse> {
    let pinn = get_pinn().await;
    let result = tokio::task::spawn_blocking(move || {
        pinn_infer(&pinn.model, &pinn.device, &pinn.norm, payload.x_pc, payload.y_pc, payload.z_pc, payload.bp_rp, payload.g_mag)
    }).await.unwrap_or([0.0, 0.0, 0.0, 0.0]);

    Json(PinnResponse {
        temperature_k: result[0],
        radius_solar: result[1],
        mass_solar: result[2],
        luminosity_solar: result[3],
    })
}

async fn gnn(Json(payload): Json<GnnRequest>) -> Json<GnnResponse> {
    let gnn_opt = get_gnn().await;

    let Some(gnn) = gnn_opt else {
        return Json(GnnResponse { stars: vec![] });
    };

    let pinn = get_pinn().await;
    let temperature = payload.temperature;

    let stars = vec![StarFeatures {
        coords: [payload.center_x, payload.center_y, payload.center_z],
        log_teff: 3.75,
        log_rad: 0.0,
        log_mass: 0.0,
        log_lum: 0.0,
        mg: 0.0,
    }];

    let stars_clone = stars.clone();
    let velocities = tokio::task::spawn_blocking(move || {
        gnn_infer(&gnn, &stars_clone, 8, temperature)
    }).await.unwrap_or_default();

    let response_stars: Vec<ResponseStar> = stars
        .iter()
        .zip(velocities.iter())
        .enumerate()
        .map(|(i, (star, vel))| {
            let [teff, rad, mass, lum] = pinn_infer(
                &pinn.model, &pinn.device, &pinn.norm,
                star.coords[0], star.coords[1], star.coords[2],
                1.0, 10.0,
            );

            let metadata = generate_stochastic_metadata(
                teff.max(0.0), rad.max(0.0), mass.max(0.0), lum.max(0.0),
                temperature,
            );

            ResponseStar {
                id: i as u32,
                x: star.coords[0],
                y: star.coords[1],
                z: star.coords[2],
                radius: rad,
                mass,
                luminosity: lum,
                description: metadata.description,
                name: metadata.designated_name,
                type_hint: metadata.spectral_class,
                velocity_vector: *vel,
            }
        })
        .collect();

    Json(GnnResponse { stars: response_stars })
}

async fn image(Json(_payload): Json<ImageRequest>) -> Json<ImageResponse> {
    Json(ImageResponse {
        shader_seed: 0,
        latent_surface_parameters: vec![],
    })
}

async fn description(Json(payload): Json<StarDescriptionPayload>) -> Json<StarLore> {
        Json(StarLore {
            designated_name: "Lunar Star".to_string(),
            category: "G-type Main Sequence".to_string(),
            visual_profile: "A bright yellow star with a serene glow.".to_string(),
            system_lore: "This star is the central anchor of the Lunar System, around which all planets orbit. It has been a beacon of hope and wonder for explorers venturing into the cosmos.".to_string(),
            metadata: LoreMetadata {
                simulation_engine: "LunarSim v1.0".to_string(),
                data_source: "Procedurally Generated".to_string(),
                complexity_level: "High".to_string(),
            },
        })
}


#[tokio::main]
async fn main() -> Result<()> {
    println!("Warming up neural network models...");
    warmup_models().await;
    println!("Models ready. Starting server on 127.0.0.1:25255");

    let app = Router::new()
        .route("/pinn", post(pinn))
        .route("/gnn", post(gnn))
        .route("/image", post(image))
        .route("/description", post(description));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:25255").await?;
    axum::serve(listener, app).await?;
    Ok(())
}