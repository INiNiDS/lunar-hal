use crate::assets::FONT_SERIF;
use dioxus::prelude::*;

#[component]
pub fn GlowingTitle() -> Element {
    rsx! {
        h1 {
            class: "absolute top-[32%] left-1/2 -translate-x-1/2 \
                   font-serif-glow text-6xl md:text-7xl text-white \
                   drop-shadow-[0_0_15px_rgba(255,255,255,0.8)] \
                   drop-shadow-[0_0_40px_rgba(255,255,255,0.4)] \
                   drop-shadow-[0_0_80px_rgba(255,255,255,0.2)] \
                   select-none whitespace-nowrap",
            style: "font-family: {FONT_SERIF}",
            "LUNAR-HAL"
        }
    }
}
