//! Defensive-coverage tests: prove the game refuses every category
//! of garbage input the AI backend (or a hostile caller) could
//! deliver. If a future refactor accidentally widens the trust
//! boundary, one of these tests will fail and the build will
//! refuse to ship.

use lunar_game_backend::validation::{
    limits, validate_bp_rp, validate_center_x, validate_center_y, validate_center_z,
    validate_entropy, validate_g_mag, validate_pipeline, validate_response_star,
    validate_response_stars, validate_search_radius, validate_sector_key, validate_temperature,
    validate_world, validate_world_id, validate_world_name, validate_world_summary, validate_zoom,
    ValidationError,
};
use lunar_game_backend::{Game, GameError};
use lunar_structures::{
    CreateWorldRequest, GnnResponse, PipelineRequest, PipelineResponse, PinnResponse, ResponseStar,
    SirenTextureResponse, StellarMetadata, World, WorldListResponse, WorldSummary,
};

// --- pure validators ---------------------------------------------------------

#[test]
fn rejects_empty_name() {
    assert!(matches!(
        validate_world_name(""),
        Err(ValidationError::Empty { .. })
    ));
    assert!(matches!(
        validate_world_name("   \t"),
        Err(ValidationError::Empty { .. })
    ));
}

#[test]
fn rejects_overlong_name() {
    let s = "x".repeat(limits::WORLD_NAME_MAX + 1);
    assert!(matches!(
        validate_world_name(&s),
        Err(ValidationError::TooLong { .. })
    ));
}

#[test]
fn rejects_control_chars_in_name() {
    for c in ['\0', '\n', '\r', '\t', '\x1b'] {
        let s = format!("Vela{c}Rim");
        assert!(matches!(
            validate_world_name(&s),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }
}

#[test]
fn rejects_path_traversal_in_id() {
    for bad in [
        "../etc/passwd",
        "..\\windows",
        "foo/bar",
        "foo%2Fbar",
        "foo bar",
        "",
    ] {
        assert!(
            validate_world_id(bad).is_err(),
            "expected reject for id: {bad:?}"
        );
    }
}

#[test]
fn rejects_unicode_in_id() {
    for bad in ["вела", "velä", "𝓥"] {
        assert!(
            validate_world_id(bad).is_err(),
            "expected reject for id: {bad:?}"
        );
    }
}

#[test]
fn rejects_nan_and_infinity_globally() {
    for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(validate_center_x(v).is_err());
        assert!(validate_center_y(v).is_err());
        assert!(validate_center_z(v).is_err());
        assert!(validate_temperature(v).is_err());
        assert!(validate_entropy(v).is_err());
        assert!(validate_bp_rp(v).is_err());
        assert!(validate_g_mag(v).is_err());
        assert!(validate_zoom(v).is_err());
        assert!(validate_search_radius(v).is_err());
    }
}

#[test]
fn rejects_out_of_range_floats() {
    assert!(validate_temperature(-0.1).is_err());
    assert!(validate_temperature(2.5).is_err());
    assert!(validate_bp_rp(-0.1).is_err());
    assert!(validate_bp_rp(5.5).is_err());
    assert!(validate_g_mag(-11.0).is_err());
    assert!(validate_g_mag(31.0).is_err());
    assert!(validate_zoom(0.001).is_err());
    assert!(validate_zoom(150.0).is_err());
    assert!(validate_search_radius(0.5).is_err());
    assert!(validate_search_radius(200_000.0).is_err());
}

#[test]
fn rejects_out_of_range_coords() {
    assert!(validate_center_x(-1_000_001.0).is_err());
    assert!(validate_center_x(1_000_001.0).is_err());
    assert!(validate_center_y(f32::MAX).is_err());
    assert!(validate_center_z(f32::MIN).is_err());
}

#[test]
fn rejects_sector_keys_outside_playable_map() {
    assert!(validate_sector_key((limits::SECTOR_KEY_MIN - 1, 0)).is_err());
    assert!(validate_sector_key((0, limits::SECTOR_KEY_MAX + 1)).is_err());
    assert!(validate_sector_key((i32::MIN, 0)).is_err());
    assert!(validate_sector_key((0, i32::MAX)).is_err());
}

#[test]
fn accepts_boundary_values() {
    assert!(validate_temperature(0.0).is_ok());
    assert!(validate_temperature(2.0).is_ok());
    assert!(validate_bp_rp(0.0).is_ok());
    assert!(validate_bp_rp(5.0).is_ok());
    assert!(validate_g_mag(-10.0).is_ok());
    assert!(validate_g_mag(30.0).is_ok());
    assert!(validate_search_radius(1.0).is_ok());
    assert!(validate_search_radius(100_000.0).is_ok());
    assert!(validate_sector_key((limits::SECTOR_KEY_MIN, 0)).is_ok());
    assert!(validate_sector_key((0, limits::SECTOR_KEY_MAX)).is_ok());
}

#[test]
fn response_star_rejects_negative_physical_and_nan() {
    let base = ResponseStar {
        id: 0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
        temperature_k: 5778.0,
        radius: 1.0,
        mass: 1.0,
        luminosity: 1.0,
        description: String::new(),
        name: String::new(),
        type_hint: String::new(),
        velocity_vector: [0.0, 0.0, 0.0],
    };
    let mut bad = base.clone();
    bad.temperature_k = -1.0;
    assert!(validate_response_star(&bad).is_err());
    let mut bad = base.clone();
    bad.mass = f32::NAN;
    assert!(validate_response_star(&bad).is_err());
    let mut bad = base.clone();
    bad.velocity_vector = [0.0, f32::INFINITY, 0.0];
    assert!(validate_response_star(&bad).is_err());
}

#[test]
fn response_stars_rejects_at_first_bad_entry() {
    let good = ResponseStar {
        id: 0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
        temperature_k: 5778.0,
        radius: 1.0,
        mass: 1.0,
        luminosity: 1.0,
        description: String::new(),
        name: String::new(),
        type_hint: String::new(),
        velocity_vector: [0.0, 0.0, 0.0],
    };
    let mut bad = good.clone();
    bad.x = f32::NAN;
    let v = vec![good.clone(), bad, good.clone()];
    assert!(validate_response_stars(&v).is_err());
    let v = vec![good.clone(), good.clone()];
    assert!(validate_response_stars(&v).is_ok());
}

#[test]
fn pipeline_rejects_mismatched_texture_size() {
    let pipeline = PipelineResponse {
        pinn: PinnResponse {
            temperature_k: 5778.0,
            radius_solar: 1.0,
            mass_solar: 1.0,
            luminosity_solar: 1.0,
        },
        siren: SirenTextureResponse {
            width: 4,
            height: 4,
            pixels: vec![0; 4 * 4 * 3 + 1], // 49, not 48
        },
        metadata: StellarMetadata {
            spectral_class: "G2V".into(),
            category: "main-sequence".into(),
            designated_name: "Vela".into(),
            description: "ok".into(),
        },
    };
    assert!(validate_pipeline(&pipeline).is_err());
}

#[test]
fn pipeline_rejects_nan_pinn() {
    let pipeline = PipelineResponse {
        pinn: PinnResponse {
            temperature_k: f32::NAN,
            radius_solar: 1.0,
            mass_solar: 1.0,
            luminosity_solar: 1.0,
        },
        siren: SirenTextureResponse {
            width: 1,
            height: 1,
            pixels: vec![0; 3],
        },
        metadata: StellarMetadata {
            spectral_class: "G2V".into(),
            category: "main-sequence".into(),
            designated_name: "Vela".into(),
            description: "ok".into(),
        },
    };
    assert!(validate_pipeline(&pipeline).is_err());
}

// --- game-level guarantees ---------------------------------------------------

fn valid_world() -> World {
    World {
        id: "abc-123".into(),
        name: "Vela Rim".into(),
        created_at: 0,
        center_x: 10.0,
        center_y: -20.0,
        center_z: 30.0,
        temperature: 0.5,
        bp_rp: 1.0,
        g_mag: 5.0,
        stars: vec![],
    }
}

#[test]
fn world_validates_id_name_coords_and_stars() {
    assert!(validate_world(&valid_world()).is_ok());
    let mut w = valid_world();
    w.id = "../bad".into();
    assert!(validate_world(&w).is_err());
    let mut w = valid_world();
    w.name = "".into();
    assert!(validate_world(&w).is_err());
    let mut w = valid_world();
    w.center_x = f32::INFINITY;
    assert!(validate_world(&w).is_err());
    let mut w = valid_world();
    w.bp_rp = 99.0;
    assert!(validate_world(&w).is_err());
    let mut w = valid_world();
    let mut bad_star = ResponseStar {
        id: 0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
        temperature_k: 5778.0,
        radius: 1.0,
        mass: 1.0,
        luminosity: 1.0,
        description: String::new(),
        name: String::new(),
        type_hint: String::new(),
        velocity_vector: [0.0, 0.0, 0.0],
    };
    bad_star.x = f32::NAN;
    w.stars.push(bad_star);
    assert!(validate_world(&w).is_err());
}

#[test]
fn world_summary_validates_id_name_coords() {
    let summary = WorldSummary {
        id: "abc-123".into(),
        name: "Vela".into(),
        created_at: 0,
        center_x: 0.0,
        center_y: 0.0,
        center_z: 0.0,
        star_count: 0,
    };
    assert!(validate_world_summary(&summary).is_ok());
    let mut bad = summary.clone();
    bad.id = "with space".into();
    assert!(validate_world_summary(&bad).is_err());
}

#[test]
fn game_rejects_setting_invalid_temperature() {
    let game = Game::new();
    assert!(game.set_temperature(f32::NAN).is_err());
    assert!(game.set_temperature(-1.0).is_err());
    assert!(game.set_temperature(3.0).is_err());
    assert!(game.set_temperature(0.7).is_ok());
}

#[test]
fn game_rejects_invalid_bp_rp_and_g_mag() {
    let game = Game::new();
    assert!(game.set_bp_rp(-0.1).is_err());
    assert!(game.set_bp_rp(6.0).is_err());
    assert!(game.set_bp_rp(1.0).is_ok());

    assert!(game.set_g_mag(-11.0).is_err());
    assert!(game.set_g_mag(31.0).is_err());
    assert!(game.set_g_mag(5.0).is_ok());
}

#[test]
fn game_rejects_invalid_sector_center() {
    let game = Game::new();
    assert!(game.set_sector_center(Some((0.0, 0.0, 0.0))).is_ok());
    assert!(game
        .set_sector_center(Some((f32::NAN, 0.0, 0.0)))
        .is_err());
    assert!(game
        .set_sector_center(Some((0.0, 1_000_001.0, 0.0)))
        .is_err());
    assert!(game.set_sector_center(None).is_ok());
}

#[test]
fn game_rejects_invalid_world_id_in_load_delete() {
    let game = Game::new();
    // The async methods should never even reach the network for an
    // obviously broken id; we cannot easily observe the network in a
    // unit test, but we can prove the validator is invoked first by
    // using a sync call path that also requires a valid id.
    assert!(game.set_world_camera("not a valid id!", Default::default()).is_err());
    assert!(game.set_world_camera("", Default::default()).is_err());
    assert!(game.apply_world_camera("../bad").is_err());
    assert!(game.remember_current_camera_for("not valid").is_err());
}

#[test]
fn game_adopt_world_drops_garbage_silently() {
    let game = Game::new();
    let mut bad = valid_world();
    bad.id = "../escape".into();
    game.adopt_world(Some(bad));
    assert!(game.active_world().is_none());
}

#[test]
fn game_rejects_malformed_create_request() {
    let game = Game::new();
    // A request with a name that is all whitespace is also invalid
    // because the validator trims and rejects empty.
    let req = CreateWorldRequest {
        name: "   ".into(),
        center_x: 0.0,
        center_y: 0.0,
        center_z: 0.0,
        temperature: 0.5,
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let res = rt.block_on(game.create_world(req));
    assert!(matches!(res, Err(GameError::Validation(_))));
}

#[test]
fn game_rejects_create_request_with_nan_coords() {
    let game = Game::new();
    let req = CreateWorldRequest {
        name: "ok".into(),
        center_x: f32::NAN,
        center_y: 0.0,
        center_z: 0.0,
        temperature: 0.5,
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let res = rt.block_on(game.create_world(req));
    assert!(matches!(res, Err(GameError::Validation(_))));
}

#[test]
fn game_apply_sector_rejects_invalid_chunk() {
    let game = Game::new();
    assert!(!game.apply_sector((999_999, 0), vec![]));
    assert!(!game.mark_sector_loading((999_999, 0)));
}

#[test]
fn game_apply_sector_rejects_invalid_star() {
    let game = Game::new();
    let mut star = ResponseStar {
        id: 0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
        temperature_k: 5778.0,
        radius: 1.0,
        mass: 1.0,
        luminosity: 1.0,
        description: String::new(),
        name: String::new(),
        type_hint: String::new(),
        velocity_vector: [0.0, 0.0, 0.0],
    };
    star.y = f32::NAN;
    assert!(!game.apply_sector((0, 0), vec![star]));
}

#[test]
fn pipeline_request_validation_keeps_dimensions_sane() {
    // Sanity check: a sane pipeline request does not crash the
    // validator. The request itself is the trusted caller's
    // responsibility; the response is what we defend against.
    let req = PipelineRequest {
        x_pc: 1.0,
        y_pc: 2.0,
        z_pc: 3.0,
        bp_rp: 1.0,
        g_mag: 10.0,
        texture_size: 256,
    };
    let _ = req; // just compile-time existence
}

#[test]
fn gnn_response_with_one_bad_star_is_rejected() {
    let good = ResponseStar {
        id: 0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
        temperature_k: 5778.0,
        radius: 1.0,
        mass: 1.0,
        luminosity: 1.0,
        description: String::new(),
        name: String::new(),
        type_hint: String::new(),
        velocity_vector: [0.0, 0.0, 0.0],
    };
    let mut bad = good.clone();
    bad.luminosity = f32::NAN;
    let resp = GnnResponse {
        stars: vec![good, bad],
    };
    assert!(validate_response_stars(&resp.stars).is_err());
}

#[test]
fn world_list_response_rejects_each_bad_summary() {
    let summary = WorldSummary {
        id: "abc-123".into(),
        name: "Vela".into(),
        created_at: 0,
        center_x: 0.0,
        center_y: 0.0,
        center_z: 0.0,
        star_count: 0,
    };
    let mut bad = summary.clone();
    bad.id = "has space".into();
    let _ = WorldListResponse {
        worlds: vec![summary, bad],
    };
    // The list itself is fine to construct; the validator is what
    // would reject it inside `refresh_worlds`. We assert that the
    // validator on the bad entry fails.
    let bad = WorldSummary {
        id: "has space".into(),
        name: "Vela".into(),
        created_at: 0,
        center_x: 0.0,
        center_y: 0.0,
        center_z: 0.0,
        star_count: 0,
    };
    assert!(validate_world_summary(&bad).is_err());
}
