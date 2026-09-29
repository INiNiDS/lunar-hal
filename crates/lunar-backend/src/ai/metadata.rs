use lunar_structures::StellarMetadata;

use super::lore::LoreCache;
use super::lore_desc::generate_description;
use super::metadata_names::generate_name;
use super::rng::SimpleRng;
use super::types::{RandomStellarInputs, StellarNorm};

pub fn generate_random_inputs(entropy: f32, norm: &StellarNorm) -> RandomStellarInputs {
    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let base = (entropy * 1000.0) as u64 ^ epoch;
    let seed = if entropy > 0.5 {
        base.wrapping_mul(6364136223846793005)
    } else {
        base
    };

    let mut rng = SimpleRng::new(seed);

    let x_pc = norm.x_mean + norm.x_std * rng.gaussian();
    let y_pc = norm.y_mean + norm.y_std * rng.gaussian();
    let z_pc = norm.z_mean + norm.z_std * rng.gaussian();

    let bp_rp = (norm.bp_rp_mean + norm.bp_rp_std * rng.gaussian()).clamp(-0.5, 5.0);

    let mg = norm.mg_mean + norm.mg_std * rng.gaussian();
    let d = (x_pc * x_pc + y_pc * y_pc + z_pc * z_pc).sqrt().max(0.1);
    let g_mag = (mg + 5.0 * d.log10() - 5.0).clamp(0.0, 20.0);

    RandomStellarInputs {
        x_pc,
        y_pc,
        z_pc,
        bp_rp,
        g_mag,
    }
}

pub fn classify_star(teff: f32, rad: f32) -> (String, String) {
    let spectral_class = if teff >= 30_000.0 {
        "O".to_string()
    } else if teff >= 10_000.0 {
        "B".to_string()
    } else if teff >= 7_500.0 {
        "A".to_string()
    } else if teff >= 6_000.0 {
        "F".to_string()
    } else if teff >= 5_200.0 {
        "G".to_string()
    } else if teff >= 3_700.0 {
        "K".to_string()
    } else {
        "M".to_string()
    };

    let category = if rad >= 100.0 {
        "Hypergiant".to_string()
    } else if rad >= 10.0 {
        "Supergiant".to_string()
    } else if rad >= 3.0 {
        "Giant".to_string()
    } else if rad >= 1.5 {
        "Subgiant".to_string()
    } else if rad >= 0.8 {
        "Main Sequence".to_string()
    } else {
        "Dwarf".to_string()
    };

    (spectral_class, category)
}

fn compute_stellar_seed(teff: f32, rad: f32, mass: f32, entropy: f32) -> u64 {
    let base = ((teff * 100.0) as u64) ^ ((rad * 1000.0) as u64) ^ ((mass * 100.0) as u64);
    if entropy > 0.5 {
        let epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        base ^ epoch.wrapping_mul((entropy * 1000.0) as u64)
    } else {
        base
    }
}

pub fn generate_stochastic_metadata(
    teff: f32,
    rad: f32,
    mass: f32,
    _lum: f32,
    entropy_temperature: f32,
) -> StellarMetadata {
    let (spectral_class, category) = classify_star(teff, rad);
    let final_seed = compute_stellar_seed(teff, rad, mass, entropy_temperature);
    let mut rng = SimpleRng::new(final_seed);

    let designated_name = generate_name(&mut rng, entropy_temperature);
    let description = generate_description(
        &mut rng,
        &spectral_class,
        &category,
        entropy_temperature,
        teff,
        rad,
    );

    StellarMetadata {
        spectral_class,
        category,
        designated_name,
        description,
    }
}

pub fn is_rare_star(teff: f32, rad: f32, mass: f32, entropy: f32) -> bool {
    if entropy > 1.5 {
        return true;
    }
    if teff >= 30_000.0 || teff <= 2_500.0 {
        return true;
    }
    if rad >= 50.0 || rad <= 0.1 {
        return true;
    }
    if mass >= 15.0 || mass <= 0.08 {
        return true;
    }
    let mut rng = SimpleRng::new(((teff * 100.0) as u64) ^ ((rad * 1000.0) as u64));
    rng.next_f32() < 0.05
}

pub fn generate_hybrid_metadata(
    teff: f32,
    rad: f32,
    mass: f32,
    lum: f32,
    entropy_temperature: f32,
    lore_cache: Option<&LoreCache>,
) -> StellarMetadata {
    let seed = compute_stellar_seed(teff, rad, mass, entropy_temperature);
    let (spectral_class, category) = classify_star(teff, rad);

    if is_rare_star(teff, rad, mass, entropy_temperature)
        && let Some(cache) = lore_cache
        && let Some(entry) = cache.pick_by_class(seed, &spectral_class)
    {
        return StellarMetadata {
            spectral_class: entry.spectral_class.clone(),
            category,
            designated_name: entry.designated_name.clone(),
            description: format!("{}. {}", entry.description, entry.system_lore),
        };
    }

    generate_stochastic_metadata(teff, rad, mass, lum, entropy_temperature)
}
