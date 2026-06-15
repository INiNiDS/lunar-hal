use crate::assets::{BG_LUNAR_LANDSCAPE, BG_LUNAR_LANDSCAPE_PC};
use crate::components::{Divider, GlowingButton, GlowingSubtitle, GlowingTitle, Header, JourneyModal};
use dioxus::prelude::*;

#[component]
pub fn HeroSection() -> Element {
    let mut modal_open = use_signal(|| false);
    let nav = use_navigator();

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
                            modal_open.set(true);
                        },
                    }
                }
            }
        }

        if modal_open() {
            JourneyModal {
                on_continue: move |_| {
                    nav.push("/editor");
                },
                on_close: move |_| modal_open.set(false),
            }
        }
    }
}