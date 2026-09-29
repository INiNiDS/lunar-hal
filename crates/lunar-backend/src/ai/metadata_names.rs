use super::rng::SimpleRng;

pub(crate) fn generate_name(rng: &mut SimpleRng, entropy: f32) -> String {
    let catalog_prefixes = [
        "UVS", "AX", "KX", "ZQ", "HD", "TYC", "GSC", "BD", "LP", "LHS", "Wolf", "Ross",
        "Gliese", "Kepler", "TrES", "XO", "HAT-P", "WASP", "K2", "TOI", "LTT", "GJ", "HIP",
        "SAO", "NGC", "IC", "Melotte", "Collinder", "Trumpler",
    ];

    let greek = ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ"];

    let name_prefixes = [
        "Aethel", "Belis", "Cygnia", "Draconis", "Eshana", "Ferox", "Glyph", "Helios", "Iridia",
        "Jovant", "Kael", "Lysand", "Mythrix", "Nocturn", "Orvex", "Pyralis", "Quinari",
        "Rhadaman", "Solace", "Thalor", "Umbra", "Vesper", "Wyrmborn", "Xanthic", "Ysolde",
        "Zephyria", "Astrar", "Celestis", "Dawnfire", "Eternis",
    ];

    let name_roots = [
        "gard", "thor", "val", "nox", "ra", "mir", "dun", "fen", "kal", "oth", "ven", "zur", "ash",
        "bel", "cor", "drak", "eld", "fal", "gor", "hak", "ion", "jer", "kre", "lux", "mor", "ner",
        "oph", "pho", "qar", "ryn",
    ];

    let name_suffixes = [
        "is", "us", "ax", "on", "ar", "el", "ix", "um", "or", "an", "ia", "os", "en", "al", "ic",
    ];

    let chaotic_prefixes = [
        "Void-Slayer",
        "Singularity",
        "Rogue-Titan",
        "Chrono-Tear",
        "Aether-Anomaly",
        "Null-Fracture",
        "Entropy-Well",
        "Quantum-Heretic",
        "Oblivion-Seed",
        "Paradox-Engine",
        "Abyss-Walker",
        "Flux-Revenant",
        "Nova-Phage",
        "Dark-Matter-Saint",
        "Gravity-Heretic",
    ];

    if entropy > 1.2 {
        let ci = (rng.next_u64() as usize) % catalog_prefixes.len();
        let num = (rng.next_u64() % 9999) + 1;
        let pi = (rng.next_u64() as usize) % chaotic_prefixes.len();
        format!("{}-{} {}", catalog_prefixes[ci], num, chaotic_prefixes[pi])
    } else if entropy > 0.5 {
        let ci = (rng.next_u64() as usize) % catalog_prefixes.len();
        let gi = (rng.next_u64() as usize) % greek.len();
        let num = (rng.next_u64() % 999) + 1;
        format!(
            "{} {}-{} {}",
            catalog_prefixes[ci],
            greek[gi],
            num,
            name_prefixes[(rng.next_u64() as usize) % name_prefixes.len()]
        )
    } else {
        let pi = (rng.next_u64() as usize) % name_prefixes.len();
        let ri = (rng.next_u64() as usize) % name_roots.len();
        let si = (rng.next_u64() as usize) % name_suffixes.len();
        let num = (rng.next_u64() % 999) + 1;
        format!(
            "{}{}{}-{}",
            name_prefixes[pi], name_roots[ri], name_suffixes[si], num
        )
    }
}
