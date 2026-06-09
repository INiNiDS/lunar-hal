use dioxus::prelude::*;
use crate::assets::FONT_SANS;

#[derive(Clone, Copy, PartialEq)]
pub enum EntropyLevel {
    Classical,
    Explorer,
    Multiverse,
}

impl EntropyLevel {
    fn from_temperature(t: f32) -> Self {
        if t < 0.5 {
            Self::Classical
        } else if t < 1.2 {
            Self::Explorer
        } else {
            Self::Multiverse
        }
    }

    fn label(&self) -> &str {
        match self {
            Self::Classical => "Classical Cosmos",
            Self::Explorer => "Explorer Space",
            Self::Multiverse => "Chaotic Multiverse",
        }
    }

    fn description(&self) -> &str {
        match self {
            Self::Classical => "Deterministic Newtonian physics. Perfectly circular orbits. Stable and predictable.",
            Self::Explorer => "Eccentric orbits, binary spirals. Unusual magnetic fields, crystalline coronas.",
            Self::Multiverse => "Rogue stars, chrono-tears, Dyson relics. Physics bends at the edge of reality.",
        }
    }

    fn color(&self) -> &str {
        match self {
            Self::Classical => "#60a5fa",
            Self::Explorer => "#a78bfa",
            Self::Multiverse => "#f472b6",
        }
    }
}

#[component]
pub fn EntropySlider(mut temperature: Signal<f32>, on_change: EventHandler<f32>) -> Element {
    let level = EntropyLevel::from_temperature(temperature());
    let temp_display = format!("{:.2}", temperature());

    rsx! {
        div { class: "flex flex-col gap-4 p-4 rounded-lg bg-black/40 backdrop-blur-sm border border-white/10",
            style: "font-family: {FONT_SANS}",

            div { class: "flex items-center justify-between",
                label { class: "text-xs uppercase tracking-widest text-white/60",
                    "Stellar Entropy"
                }
                span {
                    class: "text-sm font-semibold",
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
                    class: "w-full h-1 rounded-full appearance-none cursor-pointer bg-white/20 accent-white",
                    oninput: move |e| {
                        let val: f32 = e.value().parse().unwrap_or(0.7);
                        temperature.set(val);
                        on_change.call(val);
                    },
                }
            }

            div { class: "flex justify-between text-[10px] text-white/40",
                span { "T = 0.0" }
                span { "T = 1.0" }
                span { "T = 2.0" }
            }

            div { class: "flex flex-col gap-1",
                span {
                    class: "text-sm font-semibold",
                    style: "color: {level.color()}",
                    "{level.label()}"
                }
                span { class: "text-xs text-white/50 leading-relaxed",
                    "{level.description()}"
                }
            }
        }
    }
}