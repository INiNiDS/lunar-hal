use super::rng::SimpleRng;

pub(crate) fn generate_description(
    rng: &mut SimpleRng,
    spectral_class: &str,
    category: &str,
    entropy: f32,
    teff: f32,
    rad: f32,
) -> String {
    let classification = format!(
        "Classified as {}-type {}",
        spectral_class,
        category.to_lowercase()
    );

    let stable_traits = [
        "Stable hydrogen fusion cycle with predictable luminosity output.",
        "Convective envelope maintains consistent surface granulation patterns.",
        "Radiative core operates within standard CNO cycle parameters.",
        "Chromospheric activity within nominal bounds for its spectral type.",
        "Metallicity consistent with Population I galactic disk composition.",
        "Rotational velocity and magnetic dynamo in equilibrium.",
        "Photospheric absorption lines indicate normal heavy element abundance.",
        "Hydrostatic balance maintained across all stellar layers.",
        "Lithium depletion consistent with main-sequence age estimates.",
        "Helium ash accumulation in core proceeding at expected rates.",
    ];

    let unusual_traits = [
        "Scattered crystalline structures detected forming in the solar corona.",
        "Stellar scans reveal unnatural heavy element concentrations in the core.",
        "Its stellar flares appear structurally symmetric, suggesting external manipulation.",
        "Periodic radio emissions follow a non-natural prime-number sequence.",
        "Coronal mass ejections exhibit spiral trajectories inconsistent with magnetic models.",
        "Surface granulation patterns display fractal symmetry beyond statistical expectation.",
        "Anomalous spectral lines suggest transuranic elements in the photosphere.",
        "The magnetosphere contains structured plasma formations resembling information encoding.",
        "Doppler shifts indicate subsurface resonance patterns of unknown origin.",
        "X-ray luminosity fluctuates with a precision that suggests artificial regulation.",
        "The stellar wind carries trace isotopes not producible by natural nucleosynthesis.",
        "Helioseismic data reveals a geometric core structure inconsistent with spherical models.",
        "Coronal loops reconnect in synchronized bursts at exact time intervals.",
        "The star's proper motion includes micro-corrections too precise to be gravitational.",
        "Absorption line variations spell out repeating mathematical sequences in base-12.",
        "Photon sphere measurements indicate localized spacetime curvature anomalies.",
    ];

    let anomalous_traits = [
        "Localized gravitational inversion has been detected near the photosphere.",
        "The star appears to be phase-shifting between parallel realities.",
        "Encased in a decaying ancient Dyson Swarm structure of unknown origin.",
        "Emitting anomalous tachyonic pulses that violate local causality.",
        "A micro-wormhole appears to orbit within the corona, connecting to unknown coordinates.",
        "The fusion core has been replaced by an artifact emitting Hawking radiation at unnatural frequencies.",
        "Photons leaving the photosphere carry quantum entanglement signatures from another epoch.",
        "The star's timeline contains embedded temporal loops — events repeat with escalating variance.",
        "Gravitational lensing around this star reveals a shadow biosphere in a higher spatial dimension.",
        "The magnetosphere encodes a complete mathematical proof of a civilization's existence theorem.",
        "Stellar evolution appears to be running in reverse — the core is growing younger.",
        "A crystalline computational substrate surrounds the star, computing an unknown function for eons.",
        "The star emits neutrinos with oscillation patterns that encode compressed data streams.",
        "Space-time in the vicinity exhibits topological defects consistent with engineered metric tensor manipulation.",
        "The photosphere contains stable plasma structures that spell out warnings in a dead language.",
    ];

    let hot_star_notes = [
        "Ultraviolet flux dominates the radiation spectrum.",
        "Stellar wind velocities exceed 2000 km/s.",
        "Intense Lyman-alpha emission ionizes the surrounding interstellar medium.",
        "P-Cygni profiles in the spectrum indicate massive mass loss.",
        "The O-star radiation field creates an HII region spanning several parsecs.",
    ];

    let cool_star_notes = [
        "Molecular absorption bands of TiO and VO dominate the red spectrum.",
        "Chromospheric flare activity can double the star's luminosity within minutes.",
        "Convective cells span a significant fraction of the stellar surface.",
        "Magnetic field loops create persistent starspot regions.",
        "The photosphere is cool enough for dust formation in the outer atmosphere.",
    ];

    let giant_notes = [
        "Helium shell burning produces irregular thermal pulses.",
        "The expanded envelope shows signs of recent mass ejection events.",
        "A-instabilities in the helium shell drive luminosity variations.",
        "The extended atmosphere contains molecular layers at unusual depths.",
        "Stellar pulsations suggest dredge-up of processed material to the surface.",
    ];

    let dwarf_notes = [
        "Fully convective interior allows efficient mixing throughout the star.",
        "Magnetic activity driven by a rotational dynamo despite low luminosity.",
        "The star is expected to maintain fusion for trillions of years.",
        "Flare activity can generate X-ray bursts detectable over interstellar distances.",
        "Tidal locking in close binary systems enhances magnetic field generation.",
    ];

    let spectral_note = if teff >= 7500.0 {
        let idx = (rng.next_u64() as usize) % hot_star_notes.len();
        hot_star_notes[idx]
    } else if teff < 4000.0 {
        let idx = (rng.next_u64() as usize) % cool_star_notes.len();
        cool_star_notes[idx]
    } else if rad >= 3.0 {
        let idx = (rng.next_u64() as usize) % giant_notes.len();
        giant_notes[idx]
    } else {
        let idx = (rng.next_u64() as usize) % dwarf_notes.len();
        dwarf_notes[idx]
    };

    let trait_note = if entropy > 1.5 {
        let idx = (rng.next_u64() as usize) % anomalous_traits.len();
        anomalous_traits[idx]
    } else if entropy > 0.7 {
        let idx = (rng.next_u64() as usize) % unusual_traits.len();
        unusual_traits[idx]
    } else {
        let idx = (rng.next_u64() as usize) % stable_traits.len();
        stable_traits[idx]
    };

    let connector = [
        " Additionally, ",
        " Furthermore, ",
        " Analysis shows ",
        " Deep scans indicate ",
        " Long-range sensors detect ",
        " Survey data reveals ",
        " Spectral analysis confirms ",
        " Gravitometric readings show ",
        " Helioseismic probing reveals ",
        " Interferometric data indicates ",
    ];

    let extra = if entropy > 0.3 {
        let idx = (rng.next_u64() as usize) % connector.len();
        format!("{}{}", connector[idx], spectral_note)
    } else {
        String::new()
    };

    format!("{}. {}{}{}", classification, trait_note, extra, "")
}
