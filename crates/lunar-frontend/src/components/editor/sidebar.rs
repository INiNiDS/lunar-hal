use crate::assets::FONT_SANS;
use dioxus::prelude::*;
use lunar_structures::{PinnResponse, StarLore};
use lunar_ui_shared::star_shader::StarShaderCanvas;

const ACCENT: &str = "#a78bfa";

#[component]
pub fn StarSidebar(
    selected: bool,
    selected_teff: f32,
    pinn_data: Option<PinnResponse>,
    lore_data: Option<StarLore>,
    siren_texture_b64: Option<String>,
    on_close: EventHandler<()>,
) -> Element {
    let _ = siren_texture_b64;

    rsx! {
        div {
            class: "fixed left-4 top-4 bottom-4 w-80 flex flex-col rounded-2xl bg-black/40 backdrop-blur-xl border border-white/[0.08] shadow-[0_8px_32px_rgba(0,0,0,0.6)] overflow-hidden",
            style: "font-family: {FONT_SANS}",

            div { class: "px-6 pt-6 pb-3",
                div { class: "flex items-center justify-between",
                    div { class: "flex items-center gap-2",
                        div {
                            class: "w-1.5 h-1.5 rounded-full",
                            style: "background: {ACCENT}; box-shadow: 0 0 6px {ACCENT}",
                        }
                        h2 { class: "text-[10px] uppercase tracking-[0.2em] text-white/40 font-medium",
                            "Stellar Profile"
                        }
                    }
                    button {
                        class: "w-6 h-6 flex items-center justify-center rounded-lg text-white/30 hover:text-white/70 hover:bg-white/10 transition-colors",
                        onclick: move |_| on_close.call(()),
                        svg {
                            class: "w-3.5 h-3.5",
                            view_box: "0 0 24 24",
                            fill: "none",
                            stroke: "currentColor",
                            stroke_width: "2",
                            stroke_linecap: "round",
                            path { d: "M18 6L6 18M6 6l12 12" }
                        }
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
                            style: "background: {ACCENT}18; color: {ACCENT}; border: 1px solid {ACCENT}30",
                            "{lore.category}"
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
                        if let Some(_pinn) = &pinn_data {
                            div { style: "width: 100%; aspect-ratio: 1; border-radius: 8px; overflow: hidden;",
                                StarShaderCanvas {
                                    width: 256,
                                    height: 256,
                                    teff: selected_teff as f64,
                                    bp_rp: 0.5,
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
                        {param_row("Temperature", &format!("{:.1} K", data.temperature_k), ACCENT)}
                        {param_row("Radius", &format_abs_radius(data.radius_solar), ACCENT)}
                        {param_row("Mass", &format_abs_mass(data.mass_solar), ACCENT)}
                        {param_row("Luminosity", &format_abs_luminosity(data.luminosity_solar), ACCENT)}
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
                    }
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
