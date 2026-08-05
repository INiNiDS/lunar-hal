use crate::api;
use dioxus::prelude::*;
use lunar_stellar_core::{
    Game, PX_PER_PC, Projectile, STAR_MAX_HP,
    enemy::{Enemy, EnemyAction, EnemyType},
};
use lunar_structures::ResponseStar;
use nah::{duplicate, have_duplicate_code};
use serde_json::json;

const PLAYER_ATTACK_DAMAGE: f32 = 1.0;
const TICK_MS: u32 = 80;
const FIELD_HALF: i32 = 18000;
const SPAWN_ID_BASE: u32 = 90000;

#[derive(Clone, Debug, PartialEq)]
pub struct CustomStarFields {
    name: String,
    type_hint: String,
    teff: f64,
    radius: f64,
    mass: f64,
    lum: f64,
    x: f64,
    y: f64,
    z: f64,
    hp: f64,
}

impl Default for CustomStarFields {
    fn default() -> Self {
        Self {
            name: String::new(),
            type_hint: String::new(),
            teff: 5778.0,
            radius: 1.0,
            mass: 1.0,
            lum: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            hp: STAR_MAX_HP as f64,
        }
    }
}

#[duplicate]
fn teff_to_rgb(teff: f32) -> (f32, f32, f32) {
    if teff > 30000.0 {
        (0.62, 0.69, 1.0)
    } else if teff > 10000.0 {
        let f = (teff - 10000.0) / 20000.0;
        (0.70 - 0.08 * f, 0.77 - 0.08 * f, 0.95 + 0.05 * f)
    } else if teff > 7500.0 {
        let f = (teff - 7500.0) / 2500.0;
        (0.82 - 0.12 * f, 0.85 - 0.08 * f, 0.95)
    } else if teff > 6000.0 {
        let f = (teff - 6000.0) / 1500.0;
        (0.95 - 0.13 * f, 0.93 - 0.08 * f, 0.90 + 0.05 * f)
    } else if teff > 5200.0 {
        let f = (teff - 5200.0) / 800.0;
        (1.0, 1.0 - 0.07 * f, 0.82 + 0.08 * f)
    } else if teff > 3700.0 {
        let f = (teff - 3700.0) / 1500.0;
        (1.0, 0.85 + 0.08 * f, 0.65 + 0.25 * f)
    } else {
        (1.0, 0.55, 0.35)
    }
}

#[duplicate]
fn teff_to_rgb8(teff: f32) -> (u8, u8, u8) {
    let (r, g, b) = teff_to_rgb(teff);
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

fn enemy_type_palette(t: EnemyType) -> (&'static str, &'static str) {
    match t {
        EnemyType::Tank => ("#ff4444", "#cc0000"),
        EnemyType::Thief => ("#ffaa00", "#cc8800"),
        EnemyType::Invisible => ("#aa66ff", "#7733cc"),
        EnemyType::Scavenger => ("#ff6644", "#cc3300"),
        EnemyType::Backstabber => ("#ff2288", "#cc0066"),
        EnemyType::Coward => ("#88ccff", "#5599cc"),
    }
}

fn enemy_type_label(t: EnemyType) -> &'static str {
    match t {
        EnemyType::Tank => "Tank",
        EnemyType::Thief => "Thief",
        EnemyType::Invisible => "Invisible",
        EnemyType::Scavenger => "Scavenger",
        EnemyType::Backstabber => "Backstabber",
        EnemyType::Coward => "Coward",
    }
}

fn action_label(a: &EnemyAction) -> &'static str {
    match a {
        EnemyAction::Nothing => "Idle",
        EnemyAction::AttackingStar(_) => "Attacking star",
        EnemyAction::AttackingEnemy(_) => "Attacking enemy",
        EnemyAction::Flying(_, _) => "Flying",
        EnemyAction::Escaping { .. } => "Escaping",
    }
}

fn hp_color(ratio: f32) -> &'static str {
    if ratio > 0.6 {
        "#22c55e"
    } else if ratio > 0.25 {
        "#eab308"
    } else {
        "#ef4444"
    }
}

fn mouse_to_world(mx: f32, my: f32, vp: (f32, f32), offset: (f32, f32), zoom: f32) -> (f32, f32) {
    let cx = vp.0 * 0.5;
    let cy = vp.1 * 0.5;
    let wx = (mx - cx - offset.0) / zoom / PX_PER_PC;
    let wy = (my - cy - offset.1) / zoom / PX_PER_PC;
    (wx, wy)
}

#[have_duplicate_code]
#[cfg(feature = "web")]
fn measure_viewport_size() -> Option<(f32, f32)> {
    let window = web_sys::window();
    let document = window.as_ref().and_then(|w| w.document());
    let el = document.and_then(|d| d.query_selector(".sandbox-root").ok().flatten());
    el.map(|el| {
        let rect = el.get_bounding_client_rect();
        let w = (rect.width() as f32).clamp(100.0, 4000.0);
        let h = (rect.height() as f32).clamp(100.0, 4000.0);
        (w, h)
    })
}

#[cfg(not(feature = "web"))]
fn measure_viewport_size() -> Option<(f32, f32)> {
    Some((1280.0, 800.0))
}

fn use_viewport_measurement() -> Signal<(f32, f32)> {
    let mut viewport = use_signal(|| (0.0_f32, 0.0_f32));
    use_future(move || async move {
        let mut last = (0.0_f32, 0.0_f32);
        for _ in 0..8 {
            gloo_timers::future::TimeoutFuture::new(80).await;
            let Some((w, h)) = measure_viewport_size() else {
                continue;
            };
            if w <= 0.0 || h <= 0.0 {
                continue;
            }
            if (w, h) != last {
                last = (w, h);
                viewport.set((w, h));
            }
            if w >= 800.0 && h >= 400.0 {
                break;
            }
        }
    });
    viewport
}

#[duplicate]
fn starfield(seed_offset: u32, count: usize, half_extent: i32) -> String {
    use std::fmt::Write;
    let span = (half_extent * 2) as f32;
    let mut out = String::new();
    for i in 0..count {
        let idx = seed_offset.wrapping_add(i as u32);
        let x = (prng(idx.wrapping_mul(3)) * span) as i32 - half_extent;
        let y = (prng(idx.wrapping_mul(3).wrapping_add(1)) * span) as i32 - half_extent;
        let opacity = 0.1 + prng(idx.wrapping_mul(3).wrapping_add(2)) * 0.7;
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{}px {}px rgba(255,255,255,{:.2})", x, y, opacity);
    }
    out
}

#[duplicate]
fn prng(mut seed: u32) -> f32 {
    seed = seed.wrapping_mul(0x45d9f3b);
    seed = (seed ^ (seed >> 16)).wrapping_mul(0x45d9f3b);
    seed ^= seed >> 16;
    (seed as f32) / (u32::MAX as f32)
}

#[duplicate]
fn use_starfield_backgrounds() -> (Memo<String>, Memo<String>, Memo<String>) {
    let small = use_memo(move || starfield(0, 3000, FIELD_HALF));
    let medium = use_memo(move || starfield(10000, 1200, FIELD_HALF));
    let distant = use_memo(move || starfield(20000, 2000, FIELD_HALF));
    (small, medium, distant)
}

#[component]
fn SandboxStar(
    star: ResponseStar,
    zoom: f32,
    selected: bool,
    on_select: EventHandler<ResponseStar>,
) -> Element {
    let px = star.x * PX_PER_PC;
    let py = star.y * PX_PER_PC;
    let (r, g, b) = teff_to_rgb8(star.temperature_k);
    let size = (star.radius * 3.0).clamp(3.0, 44.0);
    let inner = format!("rgba({},{},{},1.0)", r, g, b);
    let mid = format!("rgba({},{},{},0.45)", r, g, b);
    let mid_size = size * 4.0;
    let hp_ratio = (star.hp / STAR_MAX_HP).clamp(0.0, 1.0);
    let dead = star.hp <= 0.0;
    let opacity = if dead { 0.3 } else { 0.95 };
    let delay = (star.id as f32 * 1.7).fract() * 5.0;
    let hp_c = hp_color(hp_ratio);
    let border_style = if selected {
        format!(
            "border: 1.5px solid {inner}; box-shadow: 0 0 {size}px {inner}, 0 0 {mid_size}px {mid};"
        )
    } else {
        format!("box-shadow: 0 0 {size}px {inner}, 0 0 {mid_size}px {mid};")
    };
    let star_cloned = star.clone();

    rsx! {
        div {
            class: "sandbox-star",
            style: "
                left: {px}px;
                top: {py}px;
                width: {size}px;
                height: {size}px;
                background: radial-gradient(circle, {inner} 0%, rgba({r},{g},{b},0.6) 50%, transparent 100%);
                opacity: {opacity};
                animation-delay: {delay}s;
                {border_style}
            ",
            onclick: move |e: Event<MouseData>| {
                e.stop_propagation();
                on_select.call(star_cloned.clone());
            },
        }
        div {
            class: "absolute pointer-events-none",
            style: "
                left: {px}px;
                top: {py + size * 0.5 + 4.0}px;
                width: {size}px;
                height: 3px;
                transform: translateX(-50%);
                background: rgba(255,255,255,0.06);
                border-radius: 2px;
                overflow: hidden;
            ",
            div {
                style: "
                    height: 100%;
                    width: {hp_ratio * 100.0}%;
                    background: {hp_c};
                    border-radius: 2px;
                    transition: width 0.3s ease;
                ",
            }
        }
    }
}

#[component]
fn SandboxEnemy(enemy: Enemy, selected: bool, on_click: EventHandler<Enemy>) -> Element {
    let ex = enemy.coordinates.0 * PX_PER_PC;
    let ey = enemy.coordinates.1 * PX_PER_PC;
    let max_hp = enemy.enemy_type.hp();
    let hp_ratio = (enemy.hp / max_hp).clamp(0.0, 1.0);
    let (color, dark_color) = enemy_type_palette(enemy.enemy_type);
    let size: f32 = (enemy.radius * 2.5).max(30.0).min(72.0);
    let hp_width = (size * hp_ratio).max(0.0);
    let opacity = enemy.visibility.clamp(0.0, 1.0);
    let label = enemy_type_label(enemy.enemy_type);
    let action = action_label(&enemy.action);
    let angle = (enemy.id as f32 * 1.7).fract() * 6.28;
    let offset = (angle.cos() * 4.0, angle.sin() * 4.0);
    let e_cloned = enemy.clone();

    let (line_op, line_c, line_len, dash) = match enemy.action {
        EnemyAction::AttackingStar(_) => ("0.7", color, 3.0, "3, 3"),
        EnemyAction::AttackingEnemy(_) => ("0.8", "#ff0000", 3.5, "none"),
        _ => ("0.0", color, 0.0, "none"),
    };

    let is_escaping = matches!(enemy.action, EnemyAction::Escaping { .. });
    let escape_op = if is_escaping { "0.35" } else { "0.0" };
    let sel_op = if selected { "0.9" } else { "0.0" };

    rsx! {
        div {
            class: "sandbox-enemy-wrap",
            style: "
                position: absolute;
                left: {ex}px;
                top: {ey}px;
                width: {size}px;
                height: {size}px;
                transform: translate(-50%, -50%);
                opacity: {opacity};
            ",
            onclick: move |evt: Event<MouseData>| {
                evt.stop_propagation();
                on_click.call(e_cloned.clone());
            },

            svg {
                view_box: "-30 -30 60 60",
                width: "{size}",
                height: "{size}",

                circle { cx: "0", cy: "0", r: "22", fill: "none", stroke: "{color}", stroke_width: "1", stroke_dasharray: "6, 5", opacity: "0.45" }

                g {
                    style: "filter: drop-shadow(0 0 5px {color}); animation: sb-enemy-rot 6s linear infinite; transform-origin: center;",
                    path { d: "M 0,-15 L -11,0 L 0,15 L -3,0 Z", fill: "none", stroke: "{color}", stroke_width: "1.5" }
                    path { d: "M 0,-15 L 11,0 L 0,15 L 3,0 Z", fill: "none", stroke: "{color}", stroke_width: "1.5" }
                }

                circle { cx: "0", cy: "0", r: "3", fill: "#ffffff", style: "filter: drop-shadow(0 0 6px {color}); animation: sb-enemy-pulse 1.8s ease-in-out infinite; transform-origin: center;" }

                line {
                    x1: "{offset.0}", y1: "{offset.1}",
                    x2: "{offset.0 * line_len}", y2: "{offset.1 * line_len}",
                    stroke: "{line_c}", stroke_width: "1.5", opacity: "{line_op}", stroke_dasharray: "{dash}"
                }

                circle { cx: "0", cy: "0", r: "26", fill: "none", stroke: "#ffffff", stroke_width: "0.8", stroke_dasharray: "2, 4", opacity: "{escape_op}" }
                circle { cx: "0", cy: "0", r: "27", fill: "none", stroke: "#ffffff", stroke_width: "1.5", opacity: "{sel_op}", style: "filter: drop-shadow(0 0 6px {color});" }
            }

            div {
                class: "absolute left-1/2 -translate-x-1/2 rounded-full overflow-hidden",
                style: "top: {size + 2.0}px; width: {size}px; height: 3px; background: rgba(0,0,0,0.6); border: 0.5px solid {color}30; pointer-events: none;",
                div {
                    style: "height: 100%; width: {hp_width}px; background: linear-gradient(90deg, {dark_color}, {color}); box-shadow: 0 0 4px {color}60; border-radius: 999px; transition: width 0.25s ease;"
                }
            }

            div {
                class: "absolute left-1/2 -translate-x-1/2 text-center",
                style: "top: {size + 7.0}px; width: {size}px; font-size: 7px; color: rgba(255,255,255,0.5); font-family: var(--mono); font-variant-numeric: tabular-nums; pointer-events: none; white-space: nowrap;",
                "{label} {enemy.hp:.0}/{max_hp:.0}"
            }

            div {
                class: "absolute left-1/2 -translate-x-1/2 text-center",
                style: "top: {size + 18.0}px; width: {size}px; font-size: 7px; color: {color}; font-family: var(--mono); pointer-events: none; white-space: nowrap; opacity: 0.7;",
                "{action}"
            }
        }
    }
}

#[component]
fn SandboxProjectile(projectile: Projectile, on_click: EventHandler<usize>) -> Element {
    let px = projectile.coordinates.0 * PX_PER_PC;
    let py = projectile.coordinates.1 * PX_PER_PC;
    let size = (projectile.radius * 2.0).max(12.0);
    let pid = projectile.id;

    rsx! {
        div {
            class: "pointer-events-auto cursor-pointer flex items-center justify-center",
            style: "position: absolute; left: {px}px; top: {py}px; width: {size}px; height: {size}px; transform: translate(-50%, -50%); z-index: 50;",
            onclick: move |e: Event<MouseData>| {
                e.stop_propagation();
                on_click.call(pid);
            },
            div {
                style: "width: 100%; height: 100%; background-color: #f97316; border-radius: 50%; box-shadow: 0 0 10px #f97316, 0 0 20px #ef4444; animation: sb-bullet-pulse 0.3s ease-in-out infinite alternate;",
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SandboxStats {
    stars_spawned: u32,
    enemies_spawned: u32,
    player_kills: u32,
    star_damage_dealt: f32,
    ticks: u64,
    ai_stars: u32,
}

#[component]
fn Chip(label: String, color: String, on_click: EventHandler<()>) -> Element {
    rsx! {
        button {
            class: "sandbox-chip",
            style: "border-color: {color}40; background: {color}12; color: {color};",
            onclick: move |e: Event<MouseData>| {
                e.stop_propagation();
                on_click.call(());
            },
            "{label}"
        }
    }
}

#[component]
fn StarField(
    offset: (f32, f32),
    zoom: f32,
    starfield_small: Memo<String>,
    starfield_medium: Memo<String>,
    starfield_distant: Memo<String>,
) -> Element {
    rsx! {
        div {
            class: "absolute inset-0 pointer-events-none flex items-center justify-center",
            style: "transform: translate({offset.0}px, {offset.1}px) scale({zoom}); transition: transform 0.08s linear;",
            div {
                class: "sandbox-grid",
                style: "width: 36000px; height: 36000px; left: -18000px; top: -18000px; position: absolute;",
            }
            div {
                class: "absolute bg-transparent",
                style: "width: 1px; height: 1px; box-shadow: {starfield_small()};",
            }
            div {
                class: "absolute bg-transparent rounded-full",
                style: "width: 2px; height: 2px; box-shadow: {starfield_medium()};",
            }
            div {
                class: "absolute bg-transparent",
                style: "width: 1px; height: 1px; box-shadow: {starfield_distant()};",
            }
            div { class: "sandbox-center-cross" }
        }
    }
}

#[component]
fn WorldLayer(
    offset: (f32, f32),
    zoom: f32,
    sector_stars: Vec<ResponseStar>,
    enemies: Vec<Enemy>,
    projectiles: Vec<Projectile>,
    selected_star_id: Option<u32>,
    selected_enemy_id: Option<usize>,
    on_select_star: EventHandler<ResponseStar>,
    on_click_enemy: EventHandler<Enemy>,
    on_click_projectile: EventHandler<usize>,
) -> Element {
    rsx! {
        div {
            class: "absolute inset-0 pointer-events-none",
            style: "transform: translate({offset.0}px, {offset.1}px) scale({zoom});",
            for star in sector_stars {
                SandboxStar {
                    key: "star-wrap-{star.id}",
                    selected: selected_star_id == Some(star.id),
                    star: star.clone(),
                    zoom,
                    on_select: on_select_star,
                }
            }
            for enemy in enemies {
                SandboxEnemy {
                    key: "enemy-wrap-{enemy.id}",
                    selected: selected_enemy_id == Some(enemy.id),
                    enemy: enemy.clone(),
                    on_click: on_click_enemy,
                }
            }
            for proj in projectiles {
                SandboxProjectile {
                    key: "projectile-{proj.id}",
                    projectile: proj,
                    on_click: on_click_projectile,
                }
            }
        }
    }
}

#[component]
fn StatRow(label: String, value: String, value_color: Option<String>) -> Element {
    let val_style = match value_color {
        Some(c) => format!("color: {c};"),
        None => String::new(),
    };
    rsx! {
        div { class: "sandbox-stat-row",
            span { class: "sandbox-stat-key", "{label}" }
            span { class: "sandbox-stat-val", style: "{val_style}", "{value}" }
        }
    }
}

#[component]
fn HpBarRow(label: String, ratio: f32, hp_str: String) -> Element {
    let hp_c = hp_color(ratio);
    let pct = (ratio * 100.0).clamp(0.0, 100.0);
    rsx! {
        div { style: "display: flex; align-items: center; gap: 6px; margin-top: 4px;",
            span { style: "font-size: 9px; color: rgba(255,255,255,0.4); min-width: 18px;", "{label}" }
            div { class: "sandbox-hp-bar",
                div { class: "sandbox-hp-fill", style: "width: {pct}%; background: {hp_c};" }
            }
            span { style: "font-size: 9px; color: {hp_c}; font-weight: 600;", "{hp_str}" }
        }
    }
}

#[component]
fn SelectedEnemyPanel(
    enemy: Enemy,
    game: Signal<Game>,
    mut selected_enemy_id: Signal<Option<usize>>,
    mut stats: Signal<SandboxStats>,
    mut snap: Signal<lunar_stellar_core::GameSnapshot>,
) -> Element {
    let max_hp = enemy.enemy_type.hp();
    let hp_ratio = (enemy.hp / max_hp).clamp(0.0, 1.0);
    let (color, _) = enemy_type_palette(enemy.enemy_type);
    let label = enemy_type_label(enemy.enemy_type);
    let action = action_label(&enemy.action);
    let hp_str = format!("{:.0} / {:.0}", enemy.hp, max_hp);
    let pos_str = format!("({:.1}, {:.1})", enemy.coordinates.0, enemy.coordinates.1);
    let eid = enemy.id;
    let cooldown_str = format!("{:.2}s", enemy.attack_timer);
    let dmg_label = format!("⚔ Attack (-{:.0} HP)", PLAYER_ATTACK_DAMAGE);

    rsx! {
        div { class: "sandbox-section-title", "Selected Enemy" }
        div {
            style: "background: rgba(255,255,255,0.03); border-radius: 8px; padding: 10px; margin-bottom: 8px; border-left: 2px solid {color};",
            div { style: "color: {color}; font-size: 12px; font-weight: 600; margin-bottom: 4px;",
                "{label}  ·  #{eid}"
            }
            div { style: "color: rgba(255,255,255,0.5); font-size: 10px; margin-bottom: 4px;",
                "Action: {action}"
            }
            div { style: "color: rgba(255,255,255,0.5); font-size: 10px; margin-bottom: 4px;",
                "Cooldown Timer: {cooldown_str}"
            }
            div { style: "color: rgba(255,255,255,0.5); font-size: 10px; margin-bottom: 6px;",
                "Pos: {pos_str}"
            }
            HpBarRow { label: "HP".to_string(), ratio: hp_ratio, hp_str }
            div { style: "display: flex; gap: 6px; margin-top: 8px;",
                button {
                    class: "sandbox-chip",
                    style: "flex: 1; border-color: rgba(239,68,68,0.3); background: rgba(239,68,68,0.1); color: #ef4444;",
                    onclick: move |e: Event<MouseData>| {
                        e.stop_propagation();
                        let killed = game.read().damage_enemy(eid, PLAYER_ATTACK_DAMAGE);
                        stats.with_mut(|st| {
                            st.star_damage_dealt += PLAYER_ATTACK_DAMAGE;
                            if killed { st.player_kills += 1; }
                        });
                        snap.set(game.read().snapshot());
                        if killed { selected_enemy_id.set(None); }
                    },
                    "{dmg_label}"
                }
            }
        }
    }
}

#[component]
fn SelectedStarPanel(
    star: ResponseStar,
    mut selected_star: Signal<Option<ResponseStar>>,
) -> Element {
    let hp_ratio = (star.hp / STAR_MAX_HP).clamp(0.0, 1.0);
    let (r, g, b) = teff_to_rgb8(star.temperature_k);
    let teff_str = format!("{:.0}", star.temperature_k);
    let rad_str = format!("{:.2}", star.radius);
    let mass_str = format!("{:.2}", star.mass);
    let lum_str = format!("{:.2}", star.luminosity);
    let hp_str = format!("{:.0} / {:.0}", star.hp, STAR_MAX_HP);
    let star_name = star.name.clone();
    let star_type = star.type_hint.clone();
    let pos_str = format!("({:.1}, {:.1})", star.x, star.y);
    let id_str = format!("#{}", star.id);

    rsx! {
        div { class: "sandbox-section-title", "Selected Star" }
        div {
            style: "background: rgba(255,255,255,0.03); border-radius: 8px; padding: 10px; margin-bottom: 8px;",
            div {
                style: "display: flex; align-items: center; gap: 8px; margin-bottom: 6px;",
                div {
                    style: "
                        width: 10px;
                        height: 10px;
                        background: #r97316;
                        border-radius: 50%;
                        box-shadow: 0 0 8px rgb({r},{g},{b});
                        flex-shrink: 0;
                    ",
                }
                div { style: "color: rgba(255,255,255,0.85); font-size: 12px; font-weight: 600;",
                    "{star_name}"
                }
            }
            div { style: "color: rgba(255,255,255,0.4); font-size: 10px; margin-bottom: 6px;",
                "{id_str} · {star_type} · {pos_str}"
            }
            div { style: "color: rgba(255,255,255,0.5); font-size: 10px; margin-bottom: 2px;",
                "T = {teff_str} K  ·  R = {rad_str} R⊙"
            }
            div { style: "color: rgba(255,255,255,0.5); font-size: 10px; margin-bottom: 6px;",
                "M = {mass_str} M⊙  ·  L = {lum_str} L⊙"
            }
            HpBarRow { label: "HP".to_string(), ratio: hp_ratio, hp_str }
            div { style: "display: flex; gap: 6px; margin-top: 8px;",
                button {
                    class: "sandbox-chip",
                    style: "border-color: rgba(255,255,255,0.1); background: rgba(255,255,255,0.05); color: rgba(255,255,255,0.6);",
                    onclick: move |e: Event<MouseData>| {
                        e.stop_propagation();
                        selected_star.set(None);
                    },
                    "✕ Deselect"
                }
            }
        }
    }
}

#[component]
fn EntropySlider(mut entropy: Signal<f32>) -> Element {
    let entropy_str = format!("{:.2}", entropy());
    rsx! {
        div { style: "margin-bottom: 8px;",
            label {
                class: "sandbox-field-label",
                "Entropy (0.0 – 2.0)"
            }
            input {
                class: "sandbox-input",
                r#type: "range",
                min: "0",
                max: "2",
                step: "0.05",
                value: "{entropy_str}",
                style: "padding: 0; border: none; background: transparent;",
                oninput: move |e| {
                    if let Ok(v) = e.value().as_str().parse::<f32>() {
                        entropy.set(v);
                    }
                },
            }
            div { style: "display: flex; justify-content: space-between; font-size: 9px; color: rgba(255,255,255,0.3);",
                span { "0.0" }
                span { style: "color: var(--accent); font-weight: 600;", "{entropy_str}" }
                span { "2.0" }
            }
        }
    }
}

#[component]
fn CursorSpawnCheckbox(mut use_cursor_pos: Signal<bool>) -> Element {
    rsx! {
        div { style: "display: flex; align-items: center; gap: 6px; margin-bottom: 8px;",
            input {
                r#type: "checkbox",
                checked: "{use_cursor_pos()}",
                style: "accent-color: var(--accent);",
                onclick: move |e: Event<MouseData>| {
                    e.stop_propagation();
                    use_cursor_pos.set(!use_cursor_pos());
                },
            }
            span { style: "font-size: 10px; color: rgba(255,255,255,0.6);",
                "Spawn at cursor (else viewport center)"
            }
        }
    }
}

async fn perform_ai_star_generation(
    entropy: f32,
    pos: (f32, f32),
    game: Signal<Game>,
    mut star_counter: Signal<u32>,
    mut stats: Signal<SandboxStats>,
    mut snap: Signal<lunar_stellar_core::GameSnapshot>,
) -> Result<(), String> {
    let payload = json!({ "entropy_temperature": entropy });
    let raw_res = api::random_star(&payload).await?;
    let star_val = raw_res.get("star").ok_or("No star in response")?;
    let mut star = serde_json::from_value::<ResponseStar>(star_val.clone())
        .map_err(|_| "Failed to decode star")?;

    let c = star_counter() + 1;
    star_counter.set(c);

    star.id = SPAWN_ID_BASE + c;
    star.x = pos.0;
    star.y = pos.1;
    star.z = 0.0;
    star.hp = STAR_MAX_HP;

    game.read().spawn_star(star);
    stats.with_mut(|st| {
        st.stars_spawned += 1;
        st.ai_stars += 1;
    });
    snap.set(game.read().snapshot());
    Ok(())
}

async fn perform_ai_star_batch_generation(
    entropy: f32,
    center: (f32, f32),
    game: Signal<Game>,
    mut star_counter: Signal<u32>,
    mut stats: Signal<SandboxStats>,
    mut snap: Signal<lunar_stellar_core::GameSnapshot>,
) -> Result<(), String> {
    let mut spawned = 0u32;
    for i in 0..5u32 {
        let payload = json!({ "entropy_temperature": entropy + i as f32 * 0.13 });
        if let Ok(v) = api::random_star(&payload).await {
            if let Some(star_val) = v.get("star") {
                if let Ok(mut star) = serde_json::from_value::<ResponseStar>(star_val.clone()) {
                    let c = star_counter() + 1;
                    star_counter.set(c);
                    star.id = SPAWN_ID_BASE + c;
                    star.x = center.0 + (i as f32 - 2.0) * 25.0;
                    star.y = center.1 + (((i as f32).sin()) * 25.0);
                    star.z = 0.0;
                    star.hp = STAR_MAX_HP;
                    game.read().spawn_star(star);
                    spawned += 1;
                }
            }
        }
    }
    if spawned > 0 {
        stats.with_mut(|st| {
            st.stars_spawned += spawned;
            st.ai_stars += spawned;
        });
        snap.set(game.read().snapshot());
    }
    Ok(())
}

#[component]
fn AiStarGenerator(
    game: Signal<Game>,
    mouse_world: Signal<(f32, f32)>,
    viewport: Signal<(f32, f32)>,
    star_counter: Signal<u32>,
    stats: Signal<SandboxStats>,
    snap: Signal<lunar_stellar_core::GameSnapshot>,
    mut ai_busy: Signal<bool>,
    mut ai_entropy: Signal<f32>,
    mut ai_error: Signal<Option<String>>,
    mut use_cursor_pos: Signal<bool>,
) -> Element {
    let viewport_center = move || {
        let vp = viewport();
        ((vp.0 * 0.5) / PX_PER_PC, (vp.1 * 0.5) / PX_PER_PC)
    };

    let mut on_generate = move |_: ()| {
        if ai_busy() {
            return;
        }
        ai_busy.set(true);
        ai_error.set(None);
        let entropy = ai_entropy();
        let pos = if use_cursor_pos() {
            mouse_world()
        } else {
            viewport_center()
        };

        spawn(async move {
            if let Err(e) =
                perform_ai_star_generation(entropy, pos, game, star_counter, stats, snap).await
            {
                ai_error.set(Some(e));
            }
            ai_busy.set(false);
        });
    };

    let mut on_batch = move |_: ()| {
        if ai_busy() {
            return;
        }
        ai_busy.set(true);
        ai_error.set(None);
        let entropy = ai_entropy();
        let center = viewport_center();

        spawn(async move {
            if let Err(e) =
                perform_ai_star_batch_generation(entropy, center, game, star_counter, stats, snap)
                    .await
            {
                ai_error.set(Some(e));
            }
            ai_busy.set(false);
        });
    };

    rsx! {
        div { class: "sandbox-section-title",
            "✨ AI Star Generation"
        }
        EntropySlider { entropy: ai_entropy }
        CursorSpawnCheckbox { use_cursor_pos }
        div { style: "display: flex; gap: 6px; margin-bottom: 8px;",
            button {
                class: "sandbox-chip",
                style: "flex: 1; border-color: rgba(110,168,255,0.4); background: rgba(110,168,255,0.15); color: #6ea8ff;",
                disabled: "{ai_busy()}",
                onclick: move |e: Event<MouseData>| {
                    e.stop_propagation();
                    on_generate(());
                },
                if ai_busy() { "…" } else { "✨ Generate 1" }
            }
            button {
                class: "sandbox-chip",
                style: "flex: 1; border-color: rgba(180,134,255,0.4); background: rgba(180,134,255,0.15); color: #b486ff;",
                disabled: "{ai_busy()}",
                onclick: move |e: Event<MouseData>| {
                    e.stop_propagation();
                    on_batch(());
                },
                if ai_busy() { "…" } else { "✨✨ Generate 5" }
            }
        }
        if let Some(ref err) = ai_error() {
            div {
                style: "font-size: 10px; color: var(--err); background: rgba(255,107,107,0.08); border: 1px solid rgba(255,107,107,0.2); border-radius: 5px; padding: 5px 8px; margin-bottom: 8px; word-break: break-word;",
                "{err}"
            }
        }
    }
}

#[component]
fn FormInput(
    label: String,
    #[props(default = String::new())] placeholder: String,
    #[props(default = "text".to_string())] input_type: String,
    #[props(default = String::new())] step: String,
    #[props(default = String::new())] min: String,
    value: String,
    oninput: EventHandler<String>,
) -> Element {
    rsx! {
        div {
            label { class: "sandbox-field-label", "{label}" }
            input {
                class: "sandbox-input",
                r#type: "{input_type}",
                placeholder: "{placeholder}",
                step: "{step}",
                min: "{min}",
                value: "{value}",
                oninput: move |e| oninput.call(e.value()),
            }
        }
    }
}

#[component]
fn StarIdentityInputs(mut form: Signal<CustomStarFields>) -> Element {
    rsx! {
        div { style: "display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 6px;",
            FormInput {
                label: "Name".to_string(),
                placeholder: "auto".to_string(),
                value: form().name,
                oninput: move |v: String| form.write().name = v,
            }
            FormInput {
                label: "Type hint".to_string(),
                placeholder: "custom".to_string(),
                value: form().type_hint,
                oninput: move |v: String| form.write().type_hint = v,
            }
        }
    }
}

#[component]
fn StarPhysicalInputs(mut form: Signal<CustomStarFields>) -> Element {
    rsx! {
        div { style: "display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 6px;",
            FormInput {
                label: "Teff (K)".to_string(),
                input_type: "number".to_string(),
                step: "100".to_string(),
                value: form().teff.to_string(),
                oninput: move |v: String| {
                    if let Ok(val) = v.parse::<f64>() { form.write().teff = val; }
                },
            }
            FormInput {
                label: "Radius (R⊙)".to_string(),
                input_type: "number".to_string(),
                step: "0.1".to_string(),
                value: form().radius.to_string(),
                oninput: move |v: String| {
                    if let Ok(val) = v.parse::<f64>() { form.write().radius = val; }
                },
            }
        }
        div { style: "display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 6px;",
            FormInput {
                label: "Mass (M⊙)".to_string(),
                input_type: "number".to_string(),
                step: "0.1".to_string(),
                value: form().mass.to_string(),
                oninput: move |v: String| {
                    if let Ok(val) = v.parse::<f64>() { form.write().mass = val; }
                },
            }
            FormInput {
                label: "Luminosity (L⊙)".to_string(),
                input_type: "number".to_string(),
                step: "0.1".to_string(),
                value: form().lum.to_string(),
                oninput: move |v: String| {
                    if let Ok(val) = v.parse::<f64>() { form.write().lum = val; }
                },
            }
        }
    }
}

#[component]
fn StarCoordinateInputs(
    mut form: Signal<CustomStarFields>,
    mouse_world: Signal<(f32, f32)>,
    viewport: Signal<(f32, f32)>,
) -> Element {
    let vp_center = move || {
        let vp = viewport();
        ((vp.0 * 0.5) / PX_PER_PC, (vp.1 * 0.5) / PX_PER_PC)
    };

    let mut on_at_cursor = move |_: ()| {
        let m = mouse_world();
        form.write().x = m.0 as f64;
        form.write().y = m.1 as f64;
    };

    let mut on_at_center = move |_: ()| {
        let c = vp_center();
        form.write().x = c.0 as f64;
        form.write().y = c.1 as f64;
    };

    rsx! {
        div { style: "display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; margin-bottom: 6px;",
            FormInput {
                label: "X (pc)".to_string(),
                input_type: "number".to_string(),
                step: "1".to_string(),
                value: form().x.to_string(),
                oninput: move |v: String| {
                    if let Ok(val) = v.parse::<f64>() { form.write().x = val; }
                },
            }
            FormInput {
                label: "Y (pc)".to_string(),
                input_type: "number".to_string(),
                step: "1".to_string(),
                value: form().y.to_string(),
                oninput: move |v: String| {
                    if let Ok(val) = v.parse::<f64>() { form.write().y = val; }
                },
            }
            FormInput {
                label: "Z (pc)".to_string(),
                input_type: "number".to_string(),
                step: "1".to_string(),
                value: form().z.to_string(),
                oninput: move |v: String| {
                    if let Ok(val) = v.parse::<f64>() { form.write().z = val; }
                },
            }
        }
        div { style: "display: flex; gap: 4px; margin-bottom: 8px;",
            button {
                class: "sandbox-chip",
                style: "flex: 1; border-color: rgba(255,255,255,0.1); background: rgba(255,255,255,0.05); color: rgba(255,255,255,0.7);",
                onclick: move |e: Event<MouseData>| {
                    e.stop_propagation();
                    on_at_cursor(());
                },
                "Use cursor"
            }
            button {
                class: "sandbox-chip",
                style: "flex: 1; border-color: rgba(255,255,255,0.1); background: rgba(255,255,255,0.05); color: rgba(255,255,255,0.7);",
                onclick: move |e: Event<MouseData>| {
                    e.stop_propagation();
                    on_at_center(());
                },
                "Use center"
            }
        }
    }
}

#[component]
fn CustomStarForm(
    game: Signal<Game>,
    mouse_world: Signal<(f32, f32)>,
    viewport: Signal<(f32, f32)>,
    mut star_counter: Signal<u32>,
    mut stats: Signal<SandboxStats>,
    mut snap: Signal<lunar_stellar_core::GameSnapshot>,
    form: Signal<CustomStarFields>,
) -> Element {
    let hp_max_label = format!("{:.0}", STAR_MAX_HP);

    let mut on_create = move |_: ()| {
        let f = form();
        let c = star_counter() + 1;
        star_counter.set(c);
        let star = ResponseStar {
            id: SPAWN_ID_BASE + c,
            x: f.x as f32,
            y: f.y as f32,
            z: f.z as f32,
            temperature_k: f.teff as f32,
            radius: f.radius as f32,
            mass: f.mass as f32,
            luminosity: f.lum as f32,
            hp: f.hp as f32,
            description: String::new(),
            name: if f.name.trim().is_empty() {
                format!("CUSTOM-{}", SPAWN_ID_BASE + c)
            } else {
                f.name.clone()
            },
            type_hint: if f.type_hint.trim().is_empty() {
                "custom".into()
            } else {
                f.type_hint.clone()
            },
            velocity_vector: [0.0, 0.0, 0.0],
        };
        let ok = game.read().spawn_star(star);
        if ok {
            stats.with_mut(|st| st.stars_spawned += 1);
            snap.set(game.read().snapshot());
        }
    };

    rsx! {
        div { class: "sandbox-section-title",
            "★ Custom Star Creator"
        }
        StarIdentityInputs { form }
        StarPhysicalInputs { form }
        StarCoordinateInputs { form, mouse_world, viewport }
        div { style: "margin-bottom: 8px;",
            FormInput {
                label: format!("HP (1 – {hp_max_label})"),
                input_type: "number".to_string(),
                step: "1".to_string(),
                min: "1".to_string(),
                value: form().hp.to_string(),
                oninput: move |v: String| {
                    if let Ok(val) = v.parse::<f64>() { form.write().hp = val; }
                },
            }
        }
        button {
            class: "sandbox-chip",
            style: "width: 100%; border-color: rgba(34,197,94,0.4); background: rgba(34,197,94,0.15); color: #22c55e; font-weight: 600;",
            onclick: move |e: Event<MouseData>| {
                e.stop_propagation();
                on_create(());
            },
            "★ Create Star"
        }
    }
}

#[component]
fn EnemySpawner(
    game: Signal<Game>,
    mouse_world: Signal<(f32, f32)>,
    mut enemy_counter: Signal<u32>,
    mut stats: Signal<SandboxStats>,
    mut snap: Signal<lunar_stellar_core::GameSnapshot>,
) -> Element {
    let make = move |et: EnemyType| {
        let mut enemy_counter = enemy_counter;
        let mw = mouse_world;
        let game_c = game;
        let mut stats_c = stats;
        let mut snap = snap;
        EventHandler::new(move |_: ()| {
            let c = enemy_counter() + 1;
            enemy_counter.set(c);
            let m = mw();
            game_c.read().spawn_enemy(et, m);
            stats_c.with_mut(|st| st.enemies_spawned += 1);
            snap.set(game_c.read().snapshot());
        })
    };

    rsx! {
        div { class: "sandbox-section-title",
            "☠ Spawn Enemy at Cursor"
        }
        div { style: "display: flex; gap: 4px; flex-wrap: wrap;",
            Chip { label: "Tank".to_string(), color: "#ff4444".to_string(), on_click: make(EnemyType::Tank) }
            Chip { label: "Thief".to_string(), color: "#ffaa00".to_string(), on_click: make(EnemyType::Thief) }
            Chip { label: "Invis".to_string(), color: "#aa66ff".to_string(), on_click: make(EnemyType::Invisible) }
            Chip { label: "Scav".to_string(), color: "#ff6644".to_string(), on_click: make(EnemyType::Scavenger) }
            Chip { label: "Backs".to_string(), color: "#ff2288".to_string(), on_click: make(EnemyType::Backstabber) }
            Chip { label: "Coward".to_string(), color: "#88ccff".to_string(), on_click: make(EnemyType::Coward) }
        }
    }
}

#[component]
fn CameraControls(game: Signal<Game>, viewport: Signal<(f32, f32)>, zoom: f32) -> Element {
    let on_zoom = move |factor: f32| {
        let vp = viewport();
        game.read().zoom_camera(vp, factor);
    };
    let on_recenter = move |_| {
        game.read().recenter_camera();
    };
    let zoom_pct = format!("{:.0}%", zoom * 100.0);

    rsx! {
        div { class: "sandbox-section-title",
            "Camera"
        }
        div { style: "display: flex; gap: 4px; align-items: center; margin-bottom: 6px;",
            button {
                class: "sandbox-chip",
                style: "border-color: rgba(255,255,255,0.1); background: rgba(255,255,255,0.05); color: #fff; min-width: 28px;",
                onclick: move |e: Event<MouseData>| {
                    e.stop_propagation();
                    on_zoom(1.0 / 1.3);
                },
                "−"
            }
            span {
                style: "flex: 1; text-align: center; font-size: 11px; color: var(--accent); font-weight: 600;",
                "{zoom_pct}"
            }
            button {
                class: "sandbox-chip",
                style: "border-color: rgba(255,255,255,0.1); background: rgba(255,255,255,0.05); color: #fff; min-width: 28px;",
                onclick: move |e: Event<MouseData>| {
                    e.stop_propagation();
                    on_zoom(1.3);
                },
                "+"
            }
        }
        button {
            class: "sandbox-chip",
            style: "width: 100%; border-color: rgba(255,255,255,0.1); background: rgba(255,255,255,0.05); color: rgba(255,255,255,0.7);",
            onclick: move |e: Event<MouseData>| {
                e.stop_propagation();
                on_recenter(());
            },
            "⌖ Recenter"
        }
    }
}

#[component]
fn PlaybackPanel(mut paused: Signal<bool>, mut speed: Signal<u32>) -> Element {
    rsx! {
        div {
            class: "sandbox-panel",
            style: "position: absolute; left: 12px; bottom: 12px; display: flex; gap: 8px; align-items: center; padding: 8px 12px;",

            button {
                class: "sandbox-chip",
                style: "border-color: rgba(110,168,255,0.3); background: rgba(110,168,255,0.12); color: #6ea8ff; min-width: 70px; padding: 5px 12px;",
                onclick: move |e: Event<MouseData>| {
                    e.stop_propagation();
                    paused.set(!paused());
                },
                if paused() { "▶ Play" } else { "⏸ Pause" }
            }

            span { style: "color: rgba(255,255,255,0.35); font-size: 10px;", "speed" }
            for sp in [1u32, 2, 4] {
                {
                    let active = speed() == sp;
                    let bg = if active { "rgba(110,168,255,0.25)" } else { "rgba(255,255,255,0.04)" };
                    let color = if active { "#6ea8ff" } else { "rgba(255,255,255,0.6)" };
                    let border = if active { "rgba(110,168,255,0.5)" } else { "rgba(255,255,255,0.08)" };
                    rsx! {
                        button {
                            key: "sp-{sp}",
                            class: "sandbox-chip",
                            style: "padding: 4px 10px; border-color: {border}; background: {bg}; color: {color};",
                            onclick: move |e: Event<MouseData>| {
                                e.stop_propagation();
                                speed.set(sp);
                            },
                            "{sp}x"
                        }
                    }
                }
            }

            span { style: "color: rgba(255,255,255,0.25); margin: 0 6px;", "·" }
            span { style: "color: rgba(255,255,255,0.5);", "drag · scroll to zoom · click enemy to attack" }
        }
    }
}

#[component]
fn StatsPanel(
    mw_str: String,
    alive_stars: usize,
    dead_stars: usize,
    enemies_count: usize,
    projectiles_count: usize,
    ai_stars_count: u32,
    player_kills: u32,
    dmg_str: String,
    ticks_count: u64,
) -> Element {
    rsx! {
        div {
            style: "background: rgba(255,255,255,0.03); border-radius: 8px; padding: 8px 10px; margin-bottom: 10px;",
            StatRow { label: "Mouse".to_string(), value: mw_str, value_color: None }
            StatRow { label: "Stars".to_string(), value: format!("{alive_stars} alive · {dead_stars} dead"), value_color: None }
            StatRow { label: "Enemies".to_string(), value: format!("{enemies_count}"), value_color: None }
            StatRow { label: "Projectiles".to_string(), value: format!("{projectiles_count}"), value_color: Some("#f97316".to_string()) }
            StatRow { label: "AI stars".to_string(), value: format!("{ai_stars_count}"), value_color: Some("#b486ff".to_string()) }
            StatRow { label: "Player kills".to_string(), value: format!("{player_kills}"), value_color: Some("#ef4444".to_string()) }
            StatRow { label: "Damage dealt".to_string(), value: dmg_str, value_color: Some("#f0b350".to_string()) }
            StatRow { label: "Tick".to_string(), value: format!("{ticks_count}"), value_color: None }
        }
    }
}

#[component]
fn AdminSidebar(
    game: Signal<Game>,
    mouse_world: Signal<(f32, f32)>,
    viewport: Signal<(f32, f32)>,
    star_counter: Signal<u32>,
    enemy_counter: Signal<u32>,
    stats: Signal<SandboxStats>,
    snap: Signal<lunar_stellar_core::GameSnapshot>,
    f_custom_star: Signal<CustomStarFields>,
    zoom: f32,
    selected_star: Signal<Option<ResponseStar>>,
    selected_enemy_id: Signal<Option<usize>>,
    ai_busy: Signal<bool>,
    ai_entropy: Signal<f32>,
    ai_error: Signal<Option<String>>,
    use_cursor_pos: Signal<bool>,
) -> Element {
    let s = snap();
    let sector_stars = s.sector_stars.clone();
    let enemies = s.enemies.clone();
    let projectiles = s.projectiles.clone();

    let selected_enemy =
        selected_enemy_id().and_then(|id| enemies.iter().find(|e| e.id == id).cloned());

    let selected_star_live =
        selected_star().and_then(|sel| sector_stars.iter().find(|st| st.id == sel.id).cloned());

    let mw_str = format!("({:.1}, {:.1})", mouse_world().0, mouse_world().1);
    let alive_stars = sector_stars.iter().filter(|st| st.hp > 0.0).count();
    let dead_stars = sector_stars.len() - alive_stars;
    let stats_snap = stats();
    let dmg_str = format!("{:.0}", stats_snap.star_damage_dealt);
    let ai_stars_count = stats_snap.ai_stars;
    let player_kills = stats_snap.player_kills;
    let ticks_count = stats_snap.ticks;
    let enemies_count = enemies.len();
    let projectiles_count = projectiles.len();

    let mut sel_star = selected_star;
    let mut sel_enemy = selected_enemy_id;

    rsx! {
        div {
            class: "sandbox-panel",
            style: "
                position: absolute; right: 12px; top: 12px; width: 290px;
                padding: 14px; max-height: calc(100vh - 24px); overflow-y: auto;
            ",

            div { style: "font-size: 13px; font-weight: 700; color: #6ea8ff; margin-bottom: 2px;",
                "ADMIN SANDBOX"
            }
            div { style: "font-size: 10px; color: rgba(255,255,255,0.35); margin-bottom: 10px;",
                "Live simulation · AI star generation · custom stars"
            }

            StatsPanel {
                mw_str,
                alive_stars,
                dead_stars,
                enemies_count,
                projectiles_count,
                ai_stars_count,
                player_kills,
                dmg_str,
                ticks_count,
            }

            div { class: "sandbox-divider" }

            AiStarGenerator {
                game,
                mouse_world,
                viewport,
                star_counter,
                stats,
                snap,
                ai_busy,
                ai_entropy,
                ai_error,
                use_cursor_pos,
            }

            div { class: "sandbox-divider" }

            CustomStarForm {
                game,
                mouse_world,
                viewport,
                star_counter,
                stats,
                snap,
                form: f_custom_star,
            }

            div { class: "sandbox-divider" }

            EnemySpawner {
                game,
                mouse_world,
                enemy_counter,
                stats,
                snap,
            }

            div { class: "sandbox-divider" }

            CameraControls { game, viewport, zoom }

            div { class: "sandbox-divider" }

            button {
                class: "sandbox-chip",
                style: "padding: 5px 10px; border-color: rgba(239,68,68,0.3); background: rgba(239,68,68,0.1); color: #ef4444; width: 100%;",
                onclick: move |e: Event<MouseData>| {
                    e.stop_propagation();
                    game.read().clear_sandbox();
                    sel_star.set(None);
                    sel_enemy.set(None);
                    snap.set(game.read().snapshot());
                },
                "✕ Clear All"
            }

            div { class: "sandbox-divider" }

            if let Some(enemy) = selected_enemy {
                SelectedEnemyPanel {
                    key: "sel-enemy-{enemy.id}",
                    enemy,
                    game,
                    selected_enemy_id,
                    stats,
                    snap,
                }
            } else if let Some(star) = selected_star_live {
                SelectedStarPanel {
                    key: "sel-star-{star.id}",
                    star,
                    selected_star,
                }
            } else {
                div {
                    style: "font-size: 10px; color: rgba(255,255,255,0.3); text-align: center; padding: 8px;",
                    "Click a star or enemy to inspect"
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct SandboxState {
    game: Signal<Game>,
    last_mouse: Signal<(f32, f32)>,
    mouse_world: Signal<(f32, f32)>,
    selected_star: Signal<Option<ResponseStar>>,
    selected_enemy_id: Signal<Option<usize>>,
    star_counter: Signal<u32>,
    enemy_counter: Signal<u32>,
    tick: Signal<u64>,
    snap: Signal<lunar_stellar_core::GameSnapshot>,
    paused: Signal<bool>,
    speed: Signal<u32>,
    stats: Signal<SandboxStats>,
    viewport: Signal<(f32, f32)>,
    starfield_small: Memo<String>,
    starfield_medium: Memo<String>,
    starfield_distant: Memo<String>,
    ai_busy: Signal<bool>,
    ai_entropy: Signal<f32>,
    ai_error: Signal<Option<String>>,
    use_cursor_pos: Signal<bool>,
    f_custom_star: Signal<CustomStarFields>,
}

impl SandboxState {
    fn handle_mousedown(&mut self, e: Event<MouseData>) {
        self.game.read().set_dragging(true);
        let mx = e.client_coordinates().x as f32;
        let my = e.client_coordinates().y as f32;
        self.last_mouse.set((mx, my));
    }

    fn handle_mouseup(&self) {
        self.game.read().set_dragging(false);
    }

    fn handle_mousemove(&mut self, e: Event<MouseData>) {
        let mx = e.client_coordinates().x as f32;
        let my = e.client_coordinates().y as f32;
        let vp = (self.viewport)();
        let camera = self.game.read().camera();
        let mw_coords = mouse_to_world(mx, my, vp, camera.offset, camera.zoom);
        self.mouse_world.set(mw_coords);
        if camera.dragging {
            let (lx, ly) = (self.last_mouse)();
            self.game.read().pan_camera((mx - lx, my - ly));
        }
        self.last_mouse.set((mx, my));
    }

    fn handle_wheel(&self, e: Event<WheelData>) {
        let dy = e.delta().strip_units().y;
        let factor = if dy > 0.0 { 1.0 / 1.15 } else { 1.15 };
        let mx = e.client_coordinates().x as f32;
        let my = e.client_coordinates().y as f32;
        let vp = (self.viewport)();
        self.game.read().zoom_camera_at(vp, (mx, my), factor);
    }

    fn handle_click(&mut self) {
        self.selected_star.set(None);
        self.selected_enemy_id.set(None);
    }
}

fn use_sandbox_state() -> SandboxState {
    let game = use_signal(Game::new);
    let last_mouse = use_signal(|| (0.0f32, 0.0f32));
    let mouse_world = use_signal(|| (0.0f32, 0.0f32));
    let selected_star = use_signal(|| None::<ResponseStar>);
    let selected_enemy_id = use_signal(|| None::<usize>);
    let star_counter = use_signal(|| 0u32);
    let enemy_counter = use_signal(|| 0u32);
    let tick = use_signal(|| 0u64);
    let snap = use_signal(|| game.read().snapshot());
    let paused = use_signal(|| false);
    let speed = use_signal(|| 1u32);
    let stats = use_signal(SandboxStats::default);

    let viewport = use_viewport_measurement();
    let (starfield_small, starfield_medium, starfield_distant) = use_starfield_backgrounds();

    let ai_busy = use_signal(|| false);
    let ai_entropy = use_signal(|| 1.0f32);
    let ai_error = use_signal(|| None::<String>);
    let use_cursor_pos = use_signal(|| true);

    let f_custom_star = use_signal(CustomStarFields::default);

    SandboxState {
        game,
        last_mouse,
        mouse_world,
        selected_star,
        selected_enemy_id,
        star_counter,
        enemy_counter,
        tick,
        snap,
        paused,
        speed,
        stats,
        viewport,
        starfield_small,
        starfield_medium,
        starfield_distant,
        ai_busy,
        ai_entropy,
        ai_error,
        use_cursor_pos,
        f_custom_star,
    }
}

fn use_sandbox_game_loop(mut state: SandboxState) {
    use_future(move || async move {
        loop {
            gloo_timers::future::TimeoutFuture::new(TICK_MS).await;
            if (state.paused)() {
                continue;
            }
            let g = state.game.read().clone();
            let dt = 0.08 * (state.speed)() as f32;
            let _payload = g.update(dt);
            g.remove_dead_enemies();
            let s = g.snapshot();
            let mw = *state.mouse_world.read();
            let all_stars: Vec<ResponseStar> = s.sector_stars.clone();
            if !all_stars.is_empty() {
                g.tick_attention(dt, Some(mw), &all_stars);
            }
            let s = g.snapshot();
            state.snap.set(s);
            let new_tick = (state.tick)() + 1;
            state.tick.set(new_tick);
            state.stats.with_mut(|st| st.ticks = new_tick);
        }
    });
}

#[component]
pub fn Sandbox() -> Element {
    let mut state = use_sandbox_state();
    use_sandbox_game_loop(state);

    let on_select_star = EventHandler::new(move |star: ResponseStar| {
        state.selected_star.set(Some(star));
        state.selected_enemy_id.set(None);
    });

    let on_click_enemy = EventHandler::new(move |enemy: Enemy| {
        let id = enemy.id;
        let killed = state.game.read().damage_enemy(id, PLAYER_ATTACK_DAMAGE);
        state.selected_enemy_id.set(Some(id));
        state.selected_star.set(None);
        state.stats.with_mut(|st| {
            st.star_damage_dealt += PLAYER_ATTACK_DAMAGE;
            if killed {
                st.player_kills += 1;
            }
        });
        state.snap.set(state.game.read().snapshot());
        if killed {
            state.selected_enemy_id.set(None);
        }
    });

    let s = (state.snap)();
    let camera = state.game.read().camera();
    let offset = camera.offset;
    let zoom = camera.zoom;
    let sector_stars: Vec<ResponseStar> = s.sector_stars.clone();
    let enemies: Vec<Enemy> = s.enemies.clone();
    let projectiles = s.projectiles.clone();

    rsx! {
        style {
            "@keyframes sb-enemy-rot {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}"
            "@keyframes sb-enemy-pulse {{ 0%, 100% {{ transform: scale(0.85); opacity: 0.7; }} 50% {{ transform: scale(1.15); opacity: 1; }} }}"
            "@keyframes sb-bullet-pulse {{ 0% {{ transform: scale(0.85); }} 100% {{ transform: scale(1.3); }} }}"
        }

        div {
            class: "sandbox-root",

            onmousedown: move |e: Event<MouseData>| state.handle_mousedown(e),
            onmouseup: move |_| state.handle_mouseup(),
            onmouseleave: move |_| state.handle_mouseup(),
            onmousemove: move |e: Event<MouseData>| state.handle_mousemove(e),
            onwheel: move |e: Event<WheelData>| state.handle_wheel(e),
            onclick: move |_| state.handle_click(),

            StarField {
                offset,
                zoom,
                starfield_small: state.starfield_small,
                starfield_medium: state.starfield_medium,
                starfield_distant: state.starfield_distant,
            }

            WorldLayer {
                offset,
                zoom,
                sector_stars,
                enemies,
                selected_star_id: (state.selected_star)().as_ref().map(|s| s.id),
                selected_enemy_id: (state.selected_enemy_id)(),
                on_select_star,
                on_click_enemy,
                projectiles,
                on_click_projectile: EventHandler::new(move |pid: usize| {
                    state.game.read().click_projectile(pid);
                }),
            }

            PlaybackPanel { paused: state.paused, speed: state.speed }

            AdminSidebar {
                game: state.game,
                mouse_world: state.mouse_world,
                viewport: state.viewport,
                star_counter: state.star_counter,
                enemy_counter: state.enemy_counter,
                stats: state.stats,
                snap: state.snap,
                f_custom_star: state.f_custom_star,
                zoom,
                selected_star: state.selected_star,
                selected_enemy_id: state.selected_enemy_id,
                ai_busy: state.ai_busy,
                ai_entropy: state.ai_entropy,
                ai_error: state.ai_error,
                use_cursor_pos: state.use_cursor_pos,
            }
        }
    }
}
