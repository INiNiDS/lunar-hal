use axum::{routing::{get, post}, Json, Router, extract::Query};
use serde::Deserialize;
use tower_http::cors::{Any, CorsLayer};

use lunar_structures::{
    GnnRequest, GnnResponse, PinnRequest, PinnResponse,
    PipelineRequest, PipelineResponse, RandomStarRequest, RandomStarResponse, ResponseStar,
    SirenTextureRequest, SirenTextureResponse, StarDescriptionPayload, StarLore, LoreMetadata,
};

use crate::ai::{
    generate_hybrid_metadata, generate_random_inputs, get_gnn, get_lore_cache, get_pinn,
    gnn_infer, pinn_infer, warmup_models, RandomStellarInputs, SimpleRng, StarFeatures,
};

#[cfg(feature = "siren")]
use crate::ai::{get_siren, siren_generate_texture};

pub mod ai;

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

const STARS_PER_SECTOR: usize = 30;

fn generate_sector_stars(
    cx: f32, cy: f32, cz: f32,
    search_radius: f32,
    base_teff: f32, base_rad: f32, base_mass: f32, base_lum: f32,
    base_mg: f32,
    seed: u64,
) -> Vec<StarFeatures> {
    let mut rng = SimpleRng::new(seed);
    let mut stars = Vec::with_capacity(STARS_PER_SECTOR);

    stars.push(StarFeatures {
        coords: [cx, cy, cz],
        log_teff: base_teff.max(0.01).log10(),
        log_rad: base_rad.max(0.01).log10(),
        log_mass: base_mass.max(0.01).log10(),
        log_lum: base_lum.max(0.01).log10(),
        mg: base_mg,
    });

    for _i in 1..STARS_PER_SECTOR {
        let angle1 = rng.next_f32() * std::f32::consts::PI * 2.0;
        let angle2 = rng.next_f32() * std::f32::consts::PI * 2.0;
        let dist = rng.next_f32().sqrt() * search_radius * 0.8;

        let x = cx + dist * angle1.cos() * angle2.cos();
        let y = cy + dist * angle1.sin() * angle2.cos();
        let z = cz + dist * angle2.sin();

        let noise_teff = 1.0 + (rng.gaussian() * 0.08).clamp(-0.2, 0.2);
        let noise_rad  = 1.0 + (rng.gaussian() * 0.10).clamp(-0.25, 0.25);
        let noise_mass = 1.0 + (rng.gaussian() * 0.10).clamp(-0.25, 0.25);
        let noise_lum  = 1.0 + (rng.gaussian() * 0.12).clamp(-0.3, 0.3);

        let _d = (x * x + y * y + z * z).sqrt().max(0.1);
        let mg_val = base_mg + rng.gaussian() * 0.3;

        stars.push(StarFeatures {
            coords: [x, y, z],
            log_teff: (base_teff * noise_teff).max(0.01).log10(),
            log_rad: (base_rad * noise_rad).max(0.01).log10(),
            log_mass: (base_mass * noise_mass).max(0.01).log10(),
            log_lum: (base_lum * noise_lum).max(0.01).log10(),
            mg: mg_val,
        });
    }

    stars
}

async fn gnn(Json(payload): Json<GnnRequest>) -> Json<GnnResponse> {
    let gnn_opt = get_gnn().await;

    let Some(gnn) = gnn_opt else {
        return Json(GnnResponse { stars: vec![] });
    };

    let pinn = get_pinn().await;
    let temperature = payload.temperature;

    let [teff, rad, mass, lum] = pinn_infer(
        &pinn.model, &pinn.device, &pinn.norm,
        payload.center_x, payload.center_y, payload.center_z,
        payload.bp_rp, payload.g_mag,
    );

    let d_raw = (payload.center_x.powi(2) + payload.center_y.powi(2) + payload.center_z.powi(2)).sqrt();
    let mg = if d_raw < 0.1 { 4.67 } else { payload.g_mag - 5.0 * d_raw.log10() + 5.0 };

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
        ^ (temperature * 1000.0) as u64;

    let stars = tokio::task::spawn_blocking(move || {
        generate_sector_stars(
            payload.center_x, payload.center_y, payload.center_z,
            payload.search_radius,
            teff, rad, mass, lum, mg,
            seed,
        )
    }).await.unwrap_or_default();

    let stars_clone = stars.clone();
    let velocities = tokio::task::spawn_blocking(move || {
        gnn_infer(&gnn, &stars_clone, 8.min(stars_clone.len()), temperature)
    }).await.unwrap_or_default();

    let lore = get_lore_cache().await;
    let response_stars: Vec<ResponseStar> = stars
        .iter()
        .zip(velocities.iter())
        .enumerate()
        .map(|(i, (star, vel))| {
            let teff_i = 10f32.powf(star.log_teff);
            let rad_i = 10f32.powf(star.log_rad);
            let mass_i = 10f32.powf(star.log_mass);
            let lum_i = 10f32.powf(star.log_lum);

            let metadata = generate_hybrid_metadata(
                teff_i, rad_i, mass_i, lum_i,
                temperature, lore.as_deref(),
            );

            ResponseStar {
                id: i as u32,
                x: star.coords[0],
                y: star.coords[1],
                z: star.coords[2],
                temperature_k: teff_i,
                radius: rad_i,
                mass: mass_i,
                luminosity: lum_i,
                description: metadata.description,
                name: metadata.designated_name,
                type_hint: metadata.spectral_class,
                velocity_vector: *vel,
            }
        })
        .collect();

    Json(GnnResponse { stars: response_stars })
}

async fn siren_texture(Json(payload): Json<SirenTextureRequest>) -> Json<SirenTextureResponse> {
    let siren_opt = get_siren().await;
    let Some(siren) = siren_opt else {
        return Json(SirenTextureResponse {
            width: payload.width,
            height: payload.height,
            pixels: vec![0; (payload.width as usize) * (payload.height as usize) * 3],
        });
    };

    let pixels = tokio::task::spawn_blocking(move || {
        siren_generate_texture(&siren, payload.width, payload.height, payload.bp_rp, payload.m_g, payload.log_teff)
    }).await.unwrap_or_default();

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

    let siren_opt = get_siren().await;
    let Some(siren) = siren_opt else {
        let mut png = vec![0u8; 8];
        png[0] = 0x89; png[1] = 0x50; png[2] = 0x4E; png[3] = 0x47;
        return png;
    };

    let log_teff = if teff > 0.0 { teff.log10() } else { 3.75 };

    let rgb = tokio::task::spawn_blocking(move || {
        siren_generate_texture(&siren, w, h, bp_rp, m_g, log_teff)
    }).await.unwrap_or_default();

    encode_rgb_png(&rgb, w, h)
}

fn encode_rgb_png(rgb: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;

    let mut raw = Vec::with_capacity(h * (1 + w * 3));
    for y in 0..h {
        raw.push(0);
        for x in 0..w {
            let i = (y * w + x) * 3;
            raw.push(rgb[i]);
            raw.push(rgb[i + 1]);
            raw.push(rgb[i + 2]);
        }
    }

    let deflate = deflate_minimal(&raw);

    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

    let ihdr_data = ihdr_chunk(width, height);
    png.extend_from_slice(&ihdr_data);

    let idat_data = idat_chunk(&deflate);
    png.extend_from_slice(&idat_data);

    let iend = [0, 0, 0, 0, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82];
    png.extend_from_slice(&iend);

    png
}

fn ihdr_chunk(width: u32, height: u32) -> [u8; 25] {
    let mut out = [0u8; 25];
    out[0..4].copy_from_slice(&13u32.to_be_bytes());
    out[4..8].copy_from_slice(b"IHDR");
    out[8..12].copy_from_slice(&width.to_be_bytes());
    out[12..16].copy_from_slice(&height.to_be_bytes());
    out[16] = 8;
    out[17] = 2;
    out[18..21].copy_from_slice(&[0, 0, 0]);
    let crc = crc32(&out[4..21]);
    out[21..25].copy_from_slice(&crc.to_be_bytes());
    out
}

fn idat_chunk(deflated: &[u8]) -> Vec<u8> {
    let len = deflated.len() as u32;
    let mut out = Vec::with_capacity(12 + deflated.len());
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(b"IDAT");
    out.extend_from_slice(deflated);
    let crc = crc32(&out[4..4 + 4 + deflated.len()]);
    out.extend_from_slice(&crc.to_be_bytes());
    out
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn deflate_minimal(data: &[u8]) -> Vec<u8> {
    let max_block = 65535;
    let num_blocks = (data.len() + max_block - 1) / max_block;
    let mut compressed = Vec::with_capacity(data.len() + num_blocks * 5 + 6);
    compressed.push(0x78);
    compressed.push(0x01);

    let mut offset = 0;
    for i in 0..num_blocks {
        let end = (offset + max_block).min(data.len());
        let block_len = end - offset;
        let bfinal: u8 = if i == num_blocks - 1 { 1 } else { 0 };
        compressed.push(bfinal);
        compressed.extend_from_slice(&(block_len as u16).to_le_bytes());
        compressed.extend_from_slice(&(!(block_len as u16)).to_le_bytes());
        compressed.extend_from_slice(&data[offset..end]);
        offset = end;
    }

    compressed.extend_from_slice(&adler32(data).to_be_bytes());
    compressed
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
    let pinn = get_pinn().await;
    let [teff, rad, mass, lum] = tokio::task::spawn_blocking(move || {
        pinn_infer(&pinn.model, &pinn.device, &pinn.norm, payload.x_pc, payload.y_pc, payload.z_pc, payload.bp_rp, payload.g_mag)
    }).await.unwrap_or([0.0, 0.0, 0.0, 0.0]);

    let d_raw = (payload.x_pc.powi(2) + payload.y_pc.powi(2) + payload.z_pc.powi(2)).sqrt();
    let m_g = if d_raw < 0.1 { 4.67 } else { payload.g_mag - 5.0 * d_raw.log10() + 5.0 };

    let log_teff = if teff > 0.0 { teff.log10() } else { 3.75 };

    let siren_opt = get_siren().await;
    let siren_texture = if let Some(siren) = siren_opt {
        let pixels = tokio::task::spawn_blocking(move || {
            siren_generate_texture(&siren, payload.texture_size, payload.texture_size, payload.bp_rp, m_g, log_teff)
        }).await.unwrap_or_default();
        SirenTextureResponse { width: payload.texture_size, height: payload.texture_size, pixels }
    } else {
        SirenTextureResponse { width: payload.texture_size, height: payload.texture_size, pixels: vec![0; (payload.texture_size as usize) * (payload.texture_size as usize) * 3] }
    };

    let lore = get_lore_cache().await;
    let meta = generate_hybrid_metadata(teff.max(0.0), rad.max(0.0), mass.max(0.0), lum.max(0.0), 0.5, lore.as_deref());

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

    let pinn = get_pinn().await;
    let [teff, rad, mass, lum] = tokio::task::spawn_blocking(move || {
        pinn_infer(&pinn.model, &pinn.device, &pinn.norm, inputs.x_pc, inputs.y_pc, inputs.z_pc, inputs.bp_rp, inputs.g_mag)
    }).await.unwrap_or([0.0, 0.0, 0.0, 0.0]);

    let d_raw = (inputs.x_pc.powi(2) + inputs.y_pc.powi(2) + inputs.z_pc.powi(2)).sqrt();
    let mg = if d_raw < 0.1 { 4.67 } else { inputs.g_mag - 5.0 * d_raw.log10() + 5.0 };

    let gnn_opt = get_gnn().await;

    let lore = get_lore_cache().await;

    let star = if let Some(gnn) = gnn_opt {
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

        let vel = velocities.first().copied().unwrap_or([0.0, 0.0, 0.0]);
        let metadata = generate_hybrid_metadata(
            teff.max(0.0), rad.max(0.0), mass.max(0.0), lum.max(0.0),
            entropy, lore.as_deref(),
        );

        ResponseStar {
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
        }
    } else {
        let metadata = generate_hybrid_metadata(
            teff.max(0.0), rad.max(0.0), mass.max(0.0), lum.max(0.0),
            entropy, lore.as_deref(),
        );
        ResponseStar {
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
            velocity_vector: [0.0, 0.0, 0.0],
        }
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

    let app = Router::new()
        .route("/pinn", post(pinn))
        .route("/gnn", post(gnn))
        .route("/description", post(description))
        .route("/random_star", post(random_star))
        .route("/siren/texture", post(siren_texture))
        .route("/siren/png", get(siren_png))
        .route("/pipeline", post(pipeline_handler))
        .layer(cors);

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
