use crate::assets::DIVIDER_ICON_SVG;
use dioxus::prelude::*;

#[component]
pub fn Divider() -> Element {
    rsx! {
        div { class: "absolute top-[42%] left-1/2 -translate-x-1/2 w-80 flex items-center",

            div { class: "flex-1 hero-divider-line-left" }

            div {
                class: "mx-3 hero-divider-icon",
                dangerous_inner_html: "{DIVIDER_ICON_SVG}",
            }

            div { class: "flex-1 hero-divider-line-right" }
        }
    }
}
