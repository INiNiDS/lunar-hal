use dioxus::prelude::*;
use lunar_stellar_core::enemy::Enemy as EnemyData;

fn enemy_type_color(type_name: &str) -> (&str, &str) {
    match type_name {
        "Tank" => ("#ff4444", "#cc0000"),
        "Thief" => ("#ffaa00", "#cc8800"),
        "Invisible" => ("#aa66ff", "#7733cc"),
        "Scavenger" => ("#ff6644", "#cc3300"),
        "Backstabber" => ("#ff2288", "#cc0066"),
        "Coward" => ("#88ccff", "#5599cc"),
        _ => ("#ff3333", "#cc0000"),
    }
}

#[component]
pub fn Enemy(
    enemy: EnemyData,
    center_x: f32,
    center_y: f32,
    px_per_pc: f32,
    on_click: EventHandler<EnemyData>,
) -> Element {
    let ex = (enemy.coordinates.0 - center_x) * px_per_pc;
    let ey = (enemy.coordinates.1 - center_y) * px_per_pc;
    let max_hp = enemy.enemy_type.hp();
    let hp_ratio = (enemy.hp / max_hp).clamp(0.0, 1.0);
    let (color, dark_color) = enemy_type_color(enemy.enemy_type.type_name());
    let size: f32 = (enemy.radius * 2.5).max(30.0).min(72.0);
    let hp_width = (size * hp_ratio).max(0.0);
    let opacity = enemy.visibility.clamp(0.0, 1.0);

    let angle = (enemy.id as f32 * 1.7).fract() * 6.28;
    let offset = (angle.cos() * 4.0, angle.sin() * 4.0);

    let e = enemy.clone();
    rsx! {
        style {
            "@keyframes enemy-rotate-{enemy.id} {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}"
            "@keyframes enemy-pulse-{enemy.id} {{ 0%, 100% {{ transform: scale(0.85); opacity: 0.7; }} 50% {{ transform: scale(1.15); opacity: 1; }} }}"
        }

        div {
            class: "absolute pointer-events-auto cursor-pointer",
            style: "
                left: {ex}px;
                top: {ey}px;
                width: {size}px;
                height: {size}px;
                transform: translate(-50%, -50%);
                opacity: {opacity};
                transition: opacity 0.3s;
            ",
            onclick: move |evt| {
                evt.stop_propagation();
                on_click.call(e.clone());
            },

            svg {
                view_box: "-30 -30 60 60",
                width: "{size}",
                height: "{size}",

                circle {
                    cx: "0",
                    cy: "0",
                    r: "22",
                    fill: "none",
                    stroke: "{color}",
                    stroke_width: "1",
                    stroke_dasharray: "6, 5",
                    opacity: "0.45",
                }

                g {
                    style: "filter: drop-shadow(0 0 5px {color}); animation: enemy-rotate-{enemy.id} 6s linear infinite; transform-origin: center;",

                    path {
                        d: "M 0,-15 L -11,0 L 0,15 L -3,0 Z",
                        fill: "none",
                        stroke: "{color}",
                        stroke_width: "1.5",
                    }
                    path {
                        d: "M 0,-15 L 11,0 L 0,15 L 3,0 Z",
                        fill: "none",
                        stroke: "{color}",
                        stroke_width: "1.5",
                    }
                }

                circle {
                    cx: "0",
                    cy: "0",
                    r: "3",
                    fill: "#ffffff",
                    style: "filter: drop-shadow(0 0 6px {color}); animation: enemy-pulse-{enemy.id} 1.8s ease-in-out infinite; transform-origin: center;",
                }

                match enemy.action {
                    lunar_stellar_core::enemy::EnemyAction::AttackingStar(_) => rsx! {
                        line {
                            x1: "{offset.0}",
                            y1: "{offset.1}",
                            x2: "{offset.0 * 3.0}",
                            y2: "{offset.1 * 3.0}",
                            stroke: "{color}",
                            stroke_width: "1.5",
                            opacity: "0.7",
                            stroke_dasharray: "3, 3",
                        }
                    },
                    lunar_stellar_core::enemy::EnemyAction::AttackingEnemy(_) => rsx! {
                        line {
                            x1: "{offset.0}",
                            y1: "{offset.1}",
                            x2: "{offset.0 * 3.5}",
                            y2: "{offset.1 * 3.5}",
                            stroke: "#ff0000",
                            stroke_width: "2",
                            opacity: "0.8",
                        }
                    },
                    _ => rsx! {}
                }
            }

            div {
                class: "absolute left-1/2 -translate-x-1/2 h-1 rounded-full overflow-hidden bg-white/10",
                style: "
                    top: {size + 2.0}px;
                    width: {size}px;
                    border: 0.5px solid {color}30;
                ",
                div {
                    class: "h-full rounded-full transition-all duration-300",
                    style: "
                        width: {hp_width}px;
                        background: linear-gradient(90deg, {dark_color}, {color});
                        box-shadow: 0 0 4px {color}60;
                    ",
                }
            }

            div {
                class: "text-[7px] text-white/40 font-mono tabular-nums text-center w-full absolute left-0",
                style: "top: {size + 6.0}px;",
                "{enemy.hp:.0}/{max_hp:.0}"
            }
        }
    }
}
