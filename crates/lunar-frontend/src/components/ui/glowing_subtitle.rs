use dioxus::prelude::*;
use crate::assets::FONT_SANS;

#[component]
pub fn GlowingSubtitle() -> Element {
    rsx! {
        p {
            class: "absolute top-[47%] left-1/2 -translate-x-1/2 \
                   text-xs uppercase tracking-[0.3em] text-white/70 \
                   drop-shadow-[0_0_8px_rgba(255,255,255,0.4)] \
                   select-none whitespace-nowrap",
            style: "font-family: {FONT_SANS}",
            "YOUR JOURNEY STARTS HERE",
        }
    }
}

