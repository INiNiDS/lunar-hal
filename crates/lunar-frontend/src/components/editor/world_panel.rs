use crate::assets::FONT_SANS;
use crate::game_state::use_game;
use dioxus::prelude::*;
use lunar_game_backend::Game;
use lunar_structures::{CreateWorldRequest, World, WorldSummary};

fn world_accent_color(id: &str) -> (String, String) {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in id.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let hue = (hash % 360) as f32;
    let sat = 65.0 + ((hash >> 8) % 25) as f32;
    let light = 55.0 + ((hash >> 16) % 15) as f32;
    let solid = format!("hsl({:.0}, {:.0}%, {:.0}%)", hue, sat, light);
    let glow = format!("hsla({:.0}, {:.0}%, {:.0}%, 0.55)", hue, sat, light + 10.0);
    (solid, glow)
}

#[component]
pub fn WorldPicker(
    worlds: Vec<WorldSummary>,
    loading: bool,
    on_select: EventHandler<String>,
    on_create: EventHandler<()>,
    on_delete: EventHandler<String>,
) -> Element {
    rsx! {
        div {
            class: "absolute inset-0 z-30 flex items-center justify-center bg-black/70 backdrop-blur-md",
            style: "font-family: {FONT_SANS};",

            div {
                class: "w-[640px] max-w-[92vw] max-h-[88vh] flex flex-col rounded-2xl bg-black/80 border border-white/10 shadow-[0_8px_48px_rgba(0,0,0,0.7)] overflow-hidden",

                div { class: "px-8 pt-7 pb-4 border-b border-white/[0.06]",
                    div { class: "flex items-center gap-2 mb-1",
                        div { class: "w-1.5 h-1.5 rounded-full bg-amber-400" }
                        span { class: "text-[10px] uppercase tracking-[0.25em] text-amber-300/80 font-semibold",
                            "Stellarium Archive"
                        }
                    }
                    h1 { class: "text-2xl font-bold text-white tracking-wide", "Select a World" }
                    p { class: "text-[11px] text-white/40 mt-1.5 tracking-wider uppercase",
                        "Each world is permanently catalogued with its own stellar population"
                    }
                }

                div { class: "flex-1 overflow-y-auto p-4 flex flex-col gap-2 scrollbar-thin",
                    if loading {
                        div { class: "flex flex-col items-center justify-center py-16 gap-3",
                            div { class: "w-5 h-5 border-2 border-white/10 border-t-white/60 rounded-full animate-spin" }
                            span { class: "text-[10px] uppercase tracking-widest text-white/30",
                                "Scanning the void..."
                            }
                        }
                    } else if worlds.is_empty() {
                        div { class: "flex flex-col items-center justify-center py-16 gap-3 text-center",
                            div { class: "text-[10px] uppercase tracking-widest text-white/30",
                                "No worlds found in the archive"
                            }
                            span { class: "text-xs text-white/40",
                                "Create the first sector to begin exploration."
                            }
                        }
                    } else {
                        for w in worlds.iter() {
                            WorldCard {
                                world: w.clone(),
                                on_select: move |id: String| on_select.call(id),
                                on_delete: move |id: String| on_delete.call(id),
                            }
                        }
                    }
                }

                div { class: "px-6 py-4 border-t border-white/[0.06] flex items-center justify-between bg-black/40",
                    span { class: "text-[10px] uppercase tracking-widest text-white/30",
                        "{worlds.len()} world(s) catalogued"
                    }
                    button {
                        class: "px-5 py-2.5 rounded-xl bg-amber-400/20 border border-amber-400/40 \
                               text-amber-200 text-[10px] font-bold uppercase tracking-[0.2em] \
                               hover:bg-amber-400/30 hover:text-amber-100 transition-colors cursor-pointer",
                        onclick: move |_| on_create.call(()),
                        "+ Forge New World"
                    }
                }
            }
        }
    }
}

#[component]
fn WorldCard(
    world: WorldSummary,
    on_select: EventHandler<String>,
    on_delete: EventHandler<String>,
) -> Element {
    let id = world.id.clone();
    let id_for_delete = world.id.clone();
    let (accent, glow) = world_accent_color(&world.id);
    let border_color = format!("{accent}55");
    let icon_bg = format!("linear-gradient(135deg, {accent}40, {accent}10)");
    let dot_shadow = format!("0 0 10px {glow}, 0 0 4px {accent}");
    rsx! {
        div {
            class: "group flex items-center gap-4 p-4 rounded-xl bg-white/[0.02] border transition-all cursor-pointer hover:bg-white/[0.04]",
            style: "border-color: {border_color}; box-shadow: 0 0 0 0 transparent; transition: border-color 200ms, box-shadow 200ms, background-color 200ms;",
            onclick: move |_| on_select.call(id.clone()),

            div {
                class: "w-12 h-12 rounded-lg border flex items-center justify-center shrink-0",
                style: "background: {icon_bg}; border-color: {accent}80;",
                div {
                    class: "w-2 h-2 rounded-full",
                    style: "background: {accent}; box-shadow: {dot_shadow};",
                }
            }

            div { class: "flex-1 min-w-0",
                div {
                    class: "text-sm font-semibold truncate",
                    style: "color: {accent};",
                    "{world.name}"
                }
                div { class: "flex items-center gap-2 text-[10px] text-white/40 mt-1 tracking-wider",
                    span { "[{world.center_x:.0}, {world.center_y:.0}, {world.center_z:.0}]" }
                    div { class: "w-1 h-1 rounded-full bg-white/20" }
                    span { "{world.star_count} stars" }
                }
            }

            button {
                class: "w-8 h-8 rounded-lg text-white/30 hover:text-red-400 hover:bg-red-400/10 \
                       flex items-center justify-center text-base opacity-0 group-hover:opacity-100 transition-all cursor-pointer",
                onclick: move |e| {
                    e.stop_propagation();
                    on_delete.call(id_for_delete.clone());
                },
                "×"
            }
        }
    }
}

#[component]
pub fn WorldCreator(on_cancel: EventHandler<()>, on_created: EventHandler<World>) -> Element {
    let game = use_game();
    let name = use_signal(String::new);

    // Initialize coordinate signals directly inside their closures
    let mut cx = use_signal(|| {
        let mut rng = 0x9E3779B97F4A7C15;
        let next_f32 = |state: &mut u64| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            (*state as f32) / (u64::MAX as f32)
        };
        (next_f32(&mut rng) * 2000.0) - 1000.0
    });
    let mut cy = use_signal(|| {
        let mut rng = 0x9E3779B97F4A7C15 ^ 0x12345;
        (next_f32(&mut rng) * 2000.0) - 1000.0
    });
    let mut cz = use_signal(|| {
        let mut rng = 0x9E3779B97F4A7C15 ^ 0x54321;
        (next_f32(&mut rng) * 1000.0) - 500.0
    });

    let mut temperature = use_signal(|| 0.7_f32);
    let mut submitting = use_signal(|| false);
    let mut error_msg = use_signal(|| Option::<String>::None);

    let mut randomize = move || {
        let mut rng = rng_seed();
        cx.set((next_f32(&mut rng) * 2000.0) - 1000.0);
        cy.set((next_f32(&mut rng) * 2000.0) - 1000.0);
        cz.set((next_f32(&mut rng) * 1000.0) - 500.0);
    };

    let submit = move |_| {
        let n = name().trim().to_string();
        if n.is_empty() {
            error_msg.set(Some("Name is required".into()));
            return;
        }
        submitting.set(true);
        error_msg.set(None);
        let req = CreateWorldRequest {
            name: n,
            center_x: cx(),
            center_y: cy(),
            center_z: cz(),
            temperature: temperature(),
        };
        spawn(async move {
            let g: Game = game.read().clone();
            match g.create_world(req).await {
                Ok(w) => on_created.call(w),
                Err(e) => {
                    error_msg.set(Some(e.to_string()));
                    submitting.set(false);
                }
            }
        });
    };

    rsx! {
        div {
            class: "absolute inset-0 z-40 flex items-center justify-center bg-black/80 backdrop-blur-md",
            style: "font-family: {FONT_SANS};",

            div {
                class: "w-[520px] max-w-[92vw] max-h-[90vh] flex flex-col rounded-2xl bg-black/85 border border-amber-300/20 shadow-[0_8px_48px_rgba(0,0,0,0.8)] overflow-hidden",

                div { class: "px-8 pt-7 pb-4 border-b border-white/[0.06]",
                    div { class: "flex items-center gap-2 mb-1",
                        div { class: "w-1.5 h-1.5 rounded-full bg-amber-400" }
                        span { class: "text-[10px] uppercase tracking-[0.25em] text-amber-300/80 font-semibold",
                            "Genesis Chamber"
                        }
                    }
                    h1 { class: "text-2xl font-bold text-white tracking-wide", "Forge New World" }
                    p { class: "text-[11px] text-white/40 mt-1.5 tracking-wider uppercase",
                        "Stars will be generated and crystallized into the archive forever"
                    }
                }

                div { class: "flex-1 overflow-y-auto p-6 flex flex-col gap-4 scrollbar-thin",
                    FieldInput { label: "Designation", value: name, placeholder: "e.g. Vela Rim" }

                    div { class: "grid grid-cols-3 gap-3",
                        NumberInput { label: "Center X (pc)", value: cx, step: 50.0 }
                        NumberInput { label: "Center Y (pc)", value: cy, step: 50.0 }
                        NumberInput { label: "Center Z (pc)", value: cz, step: 50.0 }
                    }

                    button {
                        class: "w-full py-2 rounded-lg bg-white/5 border border-white/10 text-white/60 text-[10px] \
                               uppercase tracking-[0.2em] font-semibold hover:bg-white/10 hover:text-white transition-colors cursor-pointer",
                        onclick: move |_| randomize(),
                        "↻ Randomize Coordinates"
                    }

                    EntropySlider {
                        temperature,
                        on_change: move |val| temperature.set(val),
                    }

                    div { class: "px-4 py-3 rounded-lg bg-amber-400/5 border border-amber-400/15 text-[10px] text-amber-200/70 leading-relaxed tracking-wider uppercase",
                        "Color and brightness crystallize randomly per star"
                    }

                    if let Some(err) = error_msg() {
                        div { class: "px-4 py-2.5 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-xs",
                            "{err}"
                        }
                    }
                }

                div { class: "px-6 py-4 border-t border-white/[0.06] flex items-center justify-end gap-3 bg-black/40",
                    button {
                        class: "px-5 py-2.5 rounded-xl text-white/60 text-[10px] font-bold uppercase tracking-[0.2em] \
                               hover:bg-white/5 hover:text-white transition-colors cursor-pointer",
                        onclick: move |_| on_cancel.call(()),
                        "Cancel"
                    }
                    button {
                        class: "px-6 py-2.5 rounded-xl bg-amber-400 text-black text-[10px] font-bold uppercase tracking-[0.2em] \
                               hover:bg-amber-300 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed",
                        disabled: submitting(),
                        onclick: submit,
                        if submitting() { "Forging..." } else { "✦ Crystallize World" }
                    }
                }
            }
        }
    }
}

#[component]
fn EntropySlider(temperature: Signal<f32>, on_change: EventHandler<f32>) -> Element {
    let (color, label, description) = if temperature() < 0.5 {
        (
            "#60a5fa",
            "Classical Cosmos",
            "Deterministic Newtonian physics. Perfectly circular orbits. Stable and predictable.",
        )
    } else if temperature() < 1.2 {
        (
            "#a78bfa",
            "Explorer Space",
            "Eccentric orbits, binary spirals. Unusual magnetic fields, crystalline coronas.",
        )
    } else {
        (
            "#f472b6",
            "Chaotic Multiverse",
            "Rogue stars, chrono-tears, Dyson relics. Physics bends at the edge of reality.",
        )
    };
    let temp_display = format!("{:.2}", temperature());

    rsx! {
        div { class: "flex flex-col gap-3 p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]",
            div { class: "flex items-center justify-between",
                label { class: "text-[10px] uppercase tracking-[0.15em] text-white/40 font-medium",
                    "Entropy"
                }
                span {
                    class: "text-sm font-bold tabular-nums",
                    style: "color: {color}",
                    "{temp_display}"
                }
            }

            input {
                r#type: "range",
                min: "0.0",
                max: "2.0",
                step: "0.01",
                value: "{temp_display}",
                class: "w-full h-1 rounded-full appearance-none cursor-pointer bg-white/10 accent-white",
                oninput: move |e| {
                    let val: f32 = e.value().parse().unwrap_or(0.7);
                    on_change.call(val);
                },
            }

            div { class: "flex justify-between text-[9px] text-white/25 tabular-nums",
                span { "0.0" }
                span { "1.0" }
                span { "2.0" }
            }

            div { class: "flex flex-col gap-0.5 pt-1",
                span {
                    class: "text-xs font-semibold",
                    style: "color: {color}",
                    "{label}"
                }
                span { class: "text-[11px] text-white/40 leading-relaxed",
                    "{description}"
                }
            }
        }
    }
}

#[component]
fn FieldInput(label: String, mut value: Signal<String>, placeholder: String) -> Element {
    rsx! {
        div { class: "flex flex-col gap-1.5",
            label { class: "text-[10px] uppercase tracking-[0.2em] text-white/40 font-semibold", "{label}" }
            input {
                r#type: "text",
                placeholder: "{placeholder}",
                value: "{value}",
                class: "w-full bg-white/5 border border-white/10 rounded-lg px-3.5 py-2.5 text-sm text-white \
                       placeholder:text-white/25 focus:outline-none focus:border-amber-300/40 focus:bg-white/[0.07] transition-colors",
                oninput: move |e| value.set(e.value()),
            }
        }
    }
}

#[component]
fn NumberInput(label: String, mut value: Signal<f32>, step: f32) -> Element {
    let display = format!("{:.2}", value());
    rsx! {
        div { class: "flex flex-col gap-1.5",
            label { class: "text-[10px] uppercase tracking-[0.2em] text-white/40 font-semibold", "{label}" }
            input {
                r#type: "number",
                step: "{step}",
                value: "{display}",
                class: "w-full bg-white/5 border border-white/10 rounded-lg px-3.5 py-2.5 text-sm text-white \
                       focus:outline-none focus:border-amber-300/40 focus:bg-white/[0.07] transition-colors tabular-nums",
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<f32>() {
                        value.set(v);
                    }
                },
            }
        }
    }
}

fn next_f32(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f32) / (u64::MAX as f32)
}

fn rng_seed() -> u64 {
    #[cfg(all(target_family = "wasm", not(target_os = "wasi")))]
    {
        let nanos = web_sys::window()
            .map(|w| {
                w.performance()
                    .map(|p| (p.now() * 1_000_000.0) as u64)
                    .unwrap_or(0x9E3779B97F4A7C15)
            })
            .unwrap_or(0x9E3779B97F4A7C15);
        nanos.wrapping_mul(0x9E3779B97F4A7C15)
    }
    #[cfg(not(all(target_family = "wasm", not(target_os = "wasi"))))]
    {
        match std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH) {
            Ok(d) => d.as_nanos() as u64,
            Err(_) => 0x9E3779B97F4A7C15,
        }
    }
}
