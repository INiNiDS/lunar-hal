use dioxus::prelude::*;
use crate::assets::DIVIDER_ICON_SVG;

#[component]
pub fn Divider() -> Element {
    rsx! {
        div {
            class: "absolute top-[42%] left-1/2 -translate-x-1/2 w-80 flex items-center",

            div { class: "flex-1 h-[1px] bg-white/40" }

            div {
                class: "mx-3 drop-shadow-[0_0_5px_rgba(255,255,255,0.5)]",
                dangerous_inner_html: "{DIVIDER_ICON_SVG}",
            }

            div { class: "flex-1 h-[1px] bg-white/40" }
        }
    }
}

