use crate::assets::{BG_LUNAR_LANDSCAPE, BG_LUNAR_LANDSCAPE_PC};
use crate::components::{Divider, EntropySlider, GlowingButton, GlowingSubtitle, GlowingTitle, Header};
use dioxus::prelude::*;

#[component]
pub fn HeroSection() -> Element {
    let nav = use_navigator();
    let temperature = use_signal(|| 0.7_f32);

    rsx! {
        div { class: "h-screen w-full relative overflow-hidden bg-black",

            img {
                src: "{BG_LUNAR_LANDSCAPE}",
                class: "absolute inset-0 w-full h-full object-cover object-bottom -z50 md:hidden",
                alt: "lunar landscape mobile",
            }

            img {
                src: "{BG_LUNAR_LANDSCAPE_PC}",
                class: "absolute inset-0 w-full h-full object-cover object-center -z50 hidden md:block",
                alt: "lunar landscape pc",
            }

            div { class: "absolute inset-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent pointer-events-none -z-10" }

            div { class: "relative z-10 flex flex-col items-center justify-between h-full py-8",

                Header {}

                div { class: "flex flex-col items-center justify-center flex-grow plan-content",
                    GlowingTitle {}
                    Divider {}
                    GlowingSubtitle {}
                    GlowingButton {
                        on_click: move |_| {
                            nav.push("/about");
                        },
                    }
                }

                div { class: "absolute right-4 top-1/2 -translate-y-1/2 w-64 hidden lg:block",
                    EntropySlider {
                        temperature,
                        on_change: move |_t| {},
                    }
                }
            }
        }
    }
}
