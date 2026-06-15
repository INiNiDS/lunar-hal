use crate::assets::FONT_SANS;
use crate::components::editor::sidebar::EntropySlider;
use dioxus::prelude::*;

#[component]
pub fn JourneyModal(on_continue: EventHandler<()>, on_close: EventHandler<()>) -> Element {
    let mut temperature = use_signal(|| 0.7_f32);

    rsx! {
        div { class: "fixed inset-0 z-50 flex items-center justify-center",
            div { class: "absolute inset-0 bg-black/70 backdrop-blur-sm",
                onclick: move |_| on_close.call(()),
            }

            div { class: "relative z-10 w-full max-w-md mx-4",
                style: "font-family: {FONT_SANS}",
                div { class: "flex flex-col gap-5 p-6 rounded-2xl \
                             bg-black/90 backdrop-blur-md border border-white/10 \
                             shadow-[0_0_60px_rgba(255,255,255,0.05)]",

                    div { class: "flex items-center justify-between",
                        h2 { class: "text-xs uppercase tracking-widest text-white/60",
                            "Stellar Entropy"
                        }
                        button {
                            class: "text-white/40 hover:text-white transition-colors cursor-pointer",
                            onclick: move |_| on_close.call(()),
                            "✕"
                        }
                    }

                    EntropySlider {
                        temperature,
                        on_change: move |val| temperature.set(val),
                    }

                    button {
                        class: "w-full rounded-full border border-white/80 \
                               bg-black/40 backdrop-blur-sm \
                               px-12 py-4 \
                               uppercase tracking-[0.2em] text-xs font-semibold text-white \
                               shadow-[0_0_15px_rgba(255,255,255,0.4)] \
                               hover:bg-white/10 hover:shadow-[0_0_25px_rgba(255,255,255,0.7)] \
                               transition-all duration-300 \
                               cursor-pointer select-none",
                        onclick: move |_| on_continue.call(()),
                        "CONTINUE"
                    }

                    button {
                        class: "w-full text-xs uppercase tracking-[0.15em] text-white/30 \
                               hover:text-white/60 transition-colors cursor-pointer",
                        onclick: move |_| on_close.call(()),
                        "Cancel"
                    }
                }
            }
        }
    }
}