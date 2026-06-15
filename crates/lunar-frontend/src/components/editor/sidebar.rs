use crate::assets::FONT_SANS;
use dioxus::prelude::*;
use lunar_structures::{PinnResponse, StarLore};
use lunar_ui_shared::star_shader::StarShaderCanvas;

#[derive(Clone, Copy, PartialEq)]
pub enum EntropyLevel {
    Classical,
    Explorer,
    Multiverse,
}

impl EntropyLevel {
    pub fn from_temperature(t: f32) -> Self {
        if t < 0.5 {
            Self::Classical
        } else if t < 1.2 {
            Self::Explorer
        } else {
            Self::Multiverse
        }
    }

    pub fn label(&self) -> &str {
        match self {
            Self::Classical => "Classical Cosmos",
            Self::Explorer => "Explorer Space",
            Self::Multiverse => "Chaotic Multiverse",
        }
    }

    pub fn description(&self) -> &str {
        match self {
            Self::Classical => "Deterministic Newtonian physics. Perfectly circular orbits. Stable and predictable.",
            Self::Explorer => "Eccentric orbits, binary spirals. Unusual magnetic fields, crystalline coronas.",
            Self::Multiverse => "Rogue stars, chrono-tears, Dyson relics. Physics bends at the edge of reality.",
        }
    }

    pub fn color(&self) -> &str {
        match self {
            Self::Classical => "#60a5fa",
            Self::Explorer => "#a78bfa",
            Self::Multiverse => "#f472b6",
        }
    }
}

#[component]
pub fn StarSidebar(
    temperature: Signal<f32>,
    on_temperature_change: EventHandler<f32>,
    bp_rp: Signal<f32>,
    g_mag: Signal<f32>,
    selected: bool,
    pinn_data: Option<PinnResponse>,
    lore_data: Option<StarLore>,
    siren_texture_b64: Option<String>,
) -> Element {
    let level = EntropyLevel::from_temperature(temperature());

    rsx! {
        div {
            class: "fixed left-4 top-4 bottom-4 w-80 flex flex-col rounded-2xl bg-black/40 backdrop-blur-xl border border-white/[0.08] shadow-[0_8px_32px_rgba(0,0,0,0.6)] overflow-hidden",
            style: "font-family: {FONT_SANS}",

            div { class: "px-6 pt-6 pb-3",
                div { class: "flex items-center gap-2",
                    div {
                        class: "w-1.5 h-1.5 rounded-full",
                        style: "background: {level.color()}; box-shadow: 0 0 6px {level.color()}",
                    }
                    h2 { class: "text-[10px] uppercase tracking-[0.2em] text-white/40 font-medium",
                        "Stellar Profile"
                    }
                }
            }

            div { class: "flex-1 overflow-y-auto px-5 pb-6 flex flex-col gap-4 scrollbar-thin",

                if let Some(lore) = &lore_data {
                    div { class: "flex flex-col gap-2 px-1",
                        h3 {
                            class: "text-lg font-bold text-white/90 tracking-wide leading-snug",
                            "{lore.designated_name}"
                        }
                        span {
                            class: "text-[10px] font-semibold uppercase tracking-widest px-2.5 py-1 rounded-full w-fit",
                            style: "background: {level.color()}18; color: {level.color()}; border: 1px solid {level.color()}30",
                            "{lore.category}"
                        }
                    }
                }

                EntropySlider {
                    temperature,
                    on_change: move |val| on_temperature_change.call(val),
                }

                div { class: "flex flex-col gap-3 p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]",
                    div { class: "text-[10px] uppercase tracking-[0.15em] text-white/40 font-medium mb-0.5",
                        "Observation"
                    }

                    div { class: "flex flex-col gap-3",
                        div { class: "flex items-center justify-between",
                            label { class: "text-[10px] uppercase tracking-[0.15em] text-white/40 font-medium",
                                "Color (Bp\u{2212}Rp)"
                            }
                            span { class: "text-sm font-bold tabular-nums text-white/70",
                                "{bp_rp():.2}"
                            }
                        }

                        div { class: "flex items-center justify-between",
                            label { class: "text-[10px] uppercase tracking-[0.15em] text-white/40 font-medium",
                                "Brightness (G mag)"
                            }
                            span { class: "text-sm font-bold tabular-nums text-white/70",
                                "{g_mag():.1}"
                            }
                        }
                    }
                }

                if selected {
                    div { class: "flex flex-col gap-2 p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]",
                        div { class: "flex items-center justify-between mb-0.5",
                            span { class: "text-[10px] uppercase tracking-[0.15em] text-white/40 font-medium",
                                "Star Surface"
                            }
                            span { class: "text-[8px] uppercase tracking-widest text-emerald-400 font-bold px-1.5 py-0.5 rounded-sm bg-emerald-400/10",
                                "LIVE"
                            }
                        }
                        if let Some(pinn) = &pinn_data {
                            div { style: "width: 100%; aspect-ratio: 1; border-radius: 8px; overflow: hidden;",
                                StarShaderCanvas {
                                    width: 256,
                                    height: 256,
                                    teff: pinn.temperature_k as f64,
                                    bp_rp: bp_rp() as f64,
                                    noise_scale: 1.5,
                                    noise_speed: 0.3,
                                    contrast: 0.8,
                                }
                            }
                        }
                    }
                }

                if let Some(data) = &pinn_data {
                    div { class: "flex flex-col gap-2 p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]",
                        div { class: "text-[10px] uppercase tracking-[0.15em] text-white/40 font-medium mb-0.5",
                            "PINN Parameters"
                        }
                        {param_row("Temperature", &format!("{:.1} K", data.temperature_k), level.color())}
                        {param_row("Radius", &format_abs_radius(data.radius_solar), level.color())}
                        {param_row("Mass", &format_abs_mass(data.mass_solar), level.color())}
                        {param_row("Luminosity", &format_abs_luminosity(data.luminosity_solar), level.color())}
                    }
                } else if selected {
                    div { class: "flex flex-col items-center justify-center py-8 gap-3",
                        div { class: "w-5 h-5 border-2 border-white/10 border-t-white/40 rounded-full animate-spin" }
                        span { class: "text-[9px] uppercase tracking-widest text-white/30", "Running Models..." }
                    }
                }

                if let Some(lore) = &lore_data {
                    div { class: "flex flex-col gap-3",
                        div { class: "flex flex-col gap-1.5 p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]",
                            div { class: "text-[10px] uppercase tracking-[0.15em] text-white/40 font-medium",
                                "Visual Profile"
                            }
                            p { class: "text-xs text-white/65 leading-relaxed",
                                "{lore.visual_profile}"
                            }
                        }

                        div { class: "flex flex-col gap-1.5 p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]",
                            div { class: "text-[10px] uppercase tracking-[0.15em] text-white/40 font-medium",
                                "System Lore"
                            }
                            p { class: "text-xs text-white/65 leading-relaxed",
                                "{lore.system_lore}"
                            }
                        }

                        div { class: "flex flex-col gap-2 p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]",
                            div { class: "text-[10px] uppercase tracking-[0.15em] text-white/40 font-medium",
                                "Metadata"
                            }
                            {meta_row("Engine", &lore.metadata.simulation_engine)}
                            {meta_row("Source", &lore.metadata.data_source)}
                            {meta_row("Complexity", &lore.metadata.complexity_level)}
                        }
                    }
                }
            }
        }
    }
}

#[component]
pub fn EntropySlider(mut temperature: Signal<f32>, on_change: EventHandler<f32>) -> Element {
    let level = EntropyLevel::from_temperature(temperature());
    let temp_display = format!("{:.2}", temperature());

    rsx! {
        div { class: "flex flex-col gap-4 p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]",
            div { class: "flex items-center justify-between",
                label { class: "text-[10px] uppercase tracking-[0.15em] text-white/40 font-medium",
                    "Entropy"
                }
                span {
                    class: "text-sm font-bold tabular-nums",
                    style: "color: {level.color()}",
                    "{temp_display}"
                }
            }

            div { class: "relative w-full",
                input {
                    r#type: "range",
                    min: "0.0",
                    max: "2.0",
                    step: "0.01",
                    value: "{temp_display}",
                    class: "w-full h-1 rounded-full appearance-none cursor-pointer bg-white/10 accent-white",
                    oninput: move |e| {
                        let val: f32 = e.value().parse().unwrap_or(0.7);
                        temperature.set(val);
                        on_change.call(val);
                    },
                }
            }

            div { class: "flex justify-between text-[9px] text-white/25 tabular-nums",
                span { "0.0" }
                span { "1.0" }
                span { "2.0" }
            }

            div { class: "flex flex-col gap-0.5 pt-1",
                span {
                    class: "text-xs font-semibold",
                    style: "color: {level.color()}",
                    "{level.label()}"
                }
                span { class: "text-[11px] text-white/40 leading-relaxed",
                    "{level.description()}"
                }
            }
        }
    }
}

fn format_abs_radius(r_solar: f32) -> String {
    let km = r_solar * 695700.0;
    if km >= 1_000_000.0 {
        format!("{:.3} R☉ ({:.1}M km)", r_solar, km / 1_000_000.0)
    } else {
        format!("{:.3} R☉ ({:.0}k km)", r_solar, km / 1000.0)
    }
}

fn format_abs_mass(m_solar: f32) -> String {
    let kg = m_solar as f64 * 1.989e30;
    format!("{:.3} M☉ ({:.2e} kg)", m_solar, kg)
}

fn format_abs_luminosity(l_solar: f32) -> String {
    let w = l_solar as f64 * 3.828e26;
    format!("{:.3} L☉ ({:.2e} W)", l_solar, w)
}

fn param_row(label: &str, value: &str, color: &str) -> Element {
    rsx! {
        div { class: "flex items-center justify-between py-0.5",
            span { class: "text-[11px] text-white/35 uppercase tracking-wider",
                "{label}"
            }
            span {
                class: "text-[13px] font-semibold tabular-nums",
                style: "color: {color}",
                "{value}"
            }
        }
    }
}

fn meta_row(label: &str, value: &str) -> Element {
    rsx! {
        div { class: "flex items-center justify-between",
            span { class: "text-[10px] text-white/25 uppercase tracking-wider",
                "{label}"
            }
            span { class: "text-[11px] text-white/55",
                "{value}"
            }
        }
    }
}