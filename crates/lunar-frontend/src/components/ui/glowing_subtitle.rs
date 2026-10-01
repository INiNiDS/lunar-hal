use crate::assets::FONT_SANS;
use dioxus::prelude::*;

#[component]
pub fn GlowingSubtitle() -> Element {
    rsx! {
        p {
            class: "absolute top-[47%] left-1/2 -translate-x-1/2 \
                   hero-subtitle-glow text-xs md:text-sm uppercase tracking-[0.35em] \
                   select-none whitespace-nowrap",
            style: "font-family: {FONT_SANS}",
            "YOUR JOURNEY STARTS HERE"
        }
    }
}
