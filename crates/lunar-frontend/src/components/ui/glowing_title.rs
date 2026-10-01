use crate::assets::FONT_SERIF;
use dioxus::prelude::*;

#[component]
pub fn GlowingTitle() -> Element {
    rsx! {
        h1 {
            class: "absolute top-[32%] left-1/2 -translate-x-1/2 \
                   hero-title-glow text-6xl md:text-7xl font-bold \
                   select-none whitespace-nowrap",
            style: "font-family: {FONT_SERIF}",
            "LUNAR-HAL"
        }
    }
}
