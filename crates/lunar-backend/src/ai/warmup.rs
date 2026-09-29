use super::gnn::get_gnn;
use super::gnn_infer::gnn_infer;
use super::localization::get_localization;
use super::lore::get_lore_cache;
use super::pinn::{get_pinn, pinn_infer};
#[cfg(feature = "siren")]
use super::siren::{SirenInputs, get_siren, siren_infer_point};
use super::types::{PinnInputs, StarFeatures};

pub async fn warmup_models() {
    match get_pinn().await {
        Ok(pinn) => {
            println!("  PINN model loaded, warming up GPU shaders...");
            if let Err(err) = tokio::task::spawn_blocking(move || {
                let _ = pinn_infer(
                    &pinn.model,
                    &pinn.device,
                    &pinn.norm,
                    PinnInputs {
                        position: [0.0, 0.0, 0.0],
                        bp_rp: 1.0,
                        g_mag: 10.0,
                    },
                );
            })
            .await
            {
                eprintln!("  PINN warmup failed: {err}");
            }
        }
        Err(err) => eprintln!("  PINN model unavailable: {err:#}"),
    }

    if let Some(gnn) = get_gnn().await {
        let gnn_arc = gnn.clone();
        println!(
            "  GNN model loaded (variational={}), warming up...",
            gnn_arc.variational
        );
        tokio::task::spawn_blocking(move || {
            // Two-node group: single-node inference is excluded (Stage
            // 6.6), so warmup runs the smallest valid group instead.
            let star = StarFeatures {
                coords: [0.0, 0.0, 0.0],
                log_teff: 3.75,
                log_rad: 0.0,
                log_mass: 0.0,
                log_lum: 0.0,
                mg: 0.0,
            };
            let neighbor = StarFeatures {
                coords: [1.0, 0.5, -0.5],
                ..star
            };
            let _ = gnn_infer(&gnn_arc, &[star, neighbor], 1, 0.0);
        })
        .await
        .ok();
    } else {
        println!("  GNN model not available (no .bpk file found)");
    }

    let _ = get_lore_cache().await;

    #[cfg(feature = "siren")]
    if let Some(siren) = get_siren().await {
        let siren_arc = siren.clone();
        println!("  SIREN model loaded, warming up...");
        tokio::task::spawn_blocking(move || {
            let norm = &siren_arc.norm;
            let _ = siren_infer_point(
                &siren_arc.model,
                &siren_arc.device,
                norm,
                SirenInputs {
                    uv: [0.0, 0.0],
                    bp_rp: 1.0,
                    m_g: 5.0,
                    log_teff: 3.75,
                },
            );
        })
        .await
        .ok();
    }

    let _ = get_localization().await;

    println!("  All models warmed up and ready.");
}
