use dioxus::prelude::*;
use crate::assets::BG_LUNAR_LANDSCAPE;
use crate::components::{GlowingTitle, Divider, GlowingSubtitle, GlowingButton, Header};

#[component]
pub fn HeroSection() -> Element {
    let nav = use_navigator();

    rsx! {
        div {
            class: "h-screen w-full relative overflow-hidden",
            style: "background-image: url('{BG_LUNAR_LANDSCAPE}'); background-size: cover; background-position: center;",

            img {
                src: "{BG_LUNAR_LANDSCAPE}",
                class: "absolute inset-0 w-full h-full object-cover object-center -z-30",
                alt: "lunar landscape",
            }

            div {
                class: "absolute inset-0 bg-gradient-to-t from-black/60 to-transparent pointer-events-none",
            }
            Header {}

            GlowingTitle {}
            Divider {}
            GlowingSubtitle {}
            GlowingButton {
                on_click: move |_| {
                    nav.push("/about");
                },
            }
        }
    }
}

