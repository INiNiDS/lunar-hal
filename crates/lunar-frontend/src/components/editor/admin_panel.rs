use crate::assets::FONT_SANS;
use dioxus::prelude::*;
use lunar_game_backend::enemy::{Enemy, EnemyAction, EnemyType, Projectile};
use lunar_structures::ResponseStar;

fn action_label(action: &EnemyAction, stars: &[ResponseStar]) -> String {
    match action {
        EnemyAction::Nothing => String::from("idle"),
        EnemyAction::AttackingStar(coord) => {
            if let Some(star) = stars.iter().find(|s| {
                (s.x - coord.0).abs() < 1.0 && (s.y - coord.1).abs() < 1.0
            }) {
                format!("atk-star#{} ({:.0},{:.0})", star.id, coord.0, coord.1)
            } else {
                format!("atk-star ({:.0},{:.0})", coord.0, coord.1)
            }
        }
        EnemyAction::AttackingEnemy(eid) => format!("atk-enemy#{}", eid),
        EnemyAction::Flying(x, y) => format!("fly→{:.0},{:.0}", x, y),
        EnemyAction::Escaping { direction } => format!("esc ({:.0},{:.0})", direction.0, direction.1),
    }
}

fn type_tag(et: EnemyType) -> (&'static str, &'static str) {
    match et {
        EnemyType::Tank => ("TNK", "#ff4444"),
        EnemyType::Thief => ("THF", "#ffaa00"),
        EnemyType::Invisible => ("INV", "#aa66ff"),
        EnemyType::Scavenger => ("SCV", "#ff6644"),
        EnemyType::Backstabber => ("BKS", "#ff2288"),
        EnemyType::Coward => ("CWD", "#88ccff"),
    }
}

#[component]
pub fn AdminPanel(
    enemies: Vec<Enemy>,
    projectiles: Vec<Projectile>,
    sector_stars: Vec<ResponseStar>,
    on_close: EventHandler<()>,
) -> Element {
    let stars = use_memo(move || sector_stars.clone());

    rsx! {
        div {
            class: "fixed right-4 top-4 bottom-4 w-96 flex flex-col rounded-2xl bg-black/75 backdrop-blur-xl border border-white/[0.08] shadow-[0_8px_32px_rgba(0,0,0,0.8)] overflow-hidden z-50",
            style: "font-family: {FONT_SANS}",

            div { class: "px-5 pt-4 pb-3 flex items-center justify-between border-b border-white/[0.06]",
                div { class: "flex items-center gap-2",
                    div {
                        class: "w-1.5 h-1.5 rounded-full",
                        style: "background: #ef4444; box-shadow: 0 0 6px #ef4444",
                    }
                    h2 { class: "text-[10px] uppercase tracking-[0.2em] text-white/40 font-medium",
                        "Admin Console"
                    }
                    span { class: "text-[9px] text-white/20 tabular-nums ml-1",
                        "({enemies.len()} enemies, {projectiles.len()} proj)"
                    }
                }
                button {
                    class: "w-5 h-5 flex items-center justify-center rounded text-white/30 hover:text-white/70 hover:bg-white/10 transition-colors text-xs",
                    onclick: move |_| on_close.call(()),
                    "×"
                }
            }

            div { class: "flex-1 overflow-y-auto px-4 py-3 flex flex-col gap-2 scrollbar-thin",
                if enemies.is_empty() {
                    div { class: "text-[10px] text-white/20 italic text-center py-6",
                        "No enemies active"
                    }
                }
                for enemy in &enemies {
                    {enemy_row(enemy.clone(), stars())}
                }

                if !projectiles.is_empty() {
                    div { class: "pt-3 mt-1 border-t border-white/[0.06]" }
                    div { class: "text-[9px] uppercase tracking-widest text-white/25 mb-2",
                        "Projectiles ({projectiles.len()})"
                    }
                    for p in &projectiles {
                        {projectile_row(p.clone())}
                    }
                }
            }
        }
    }
}

fn hp_bar(hp: f32, max_hp: f32) -> Element {
    let ratio = (hp / max_hp).clamp(0.0, 1.0);
    let color = if ratio > 0.6 { "#22c55e" } else if ratio > 0.25 { "#eab308" } else { "#ef4444" };
    rsx! {
        div { class: "w-full h-1.5 rounded-full overflow-hidden bg-white/5",
            div {
                class: "h-full rounded-full transition-all duration-300",
                style: "
                    width: {ratio * 100.0}%;
                    background: {color};
                    box-shadow: 0 0 4px {color}40;
                ",
            }
        }
    }
}

fn enemy_row(enemy: Enemy, stars: Vec<ResponseStar>) -> Element {
    let (tag, color) = type_tag(enemy.enemy_type);
    let max_hp = enemy.enemy_type.hp();
    let label = action_label(&enemy.action, &stars);
    let vis_pct = (enemy.visibility * 100.0) as i32;
    let id_str = enemy.id.to_string();

    rsx! {
        div {
            class: "rounded-lg bg-white/[0.03] border border-white/[0.05] px-3 py-2.5 flex flex-col gap-1.5 hover:bg-white/[0.05] transition-colors",
            div { class: "flex items-center gap-2",
                span {
                    class: "text-[8px] font-bold px-1.5 py-0.5 rounded tracking-wider",
                    style: "background: {color}18; color: {color}; border: 1px solid {color}30",
                    "{tag}"
                }
                span { class: "text-[11px] text-white/70 font-medium tabular-nums", "#{id_str}" }
                span { class: "text-[9px] text-white/25 ml-auto", "({enemy.coordinates.0:.0},{enemy.coordinates.1:.0})" }
            }
            div { class: "text-[10px] text-white/40 font-mono", "{label}" }
            {hp_bar(enemy.hp, max_hp)}
            div { class: "flex items-center gap-3 text-[8px] text-white/25 mt-0.5",
                span { "HP {enemy.hp:.0}/{max_hp:.0}" }
                span { "vis {vis_pct}%" }
                span { "cd {enemy.attack_timer:.1}s" }
            }
        }
    }
}

fn projectile_row(p: Projectile) -> Element {
    rsx! {
        div { class: "rounded bg-white/[0.02] border border-white/[0.04] px-2.5 py-1.5 flex items-center gap-2",
            span { class: "text-[8px] text-amber-400/60 font-mono tabular-nums",
                "◈#{p.id}"
            }
            span { class: "text-[9px] text-white/35", "→star#{p.target_star_id}" }
            span { class: "text-[9px] text-white/25 ml-auto tabular-nums",
                "({p.coordinates.0:.0},{p.coordinates.1:.0})"
            }
        }
    }
}
