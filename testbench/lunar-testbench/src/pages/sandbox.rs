use dioxus::prelude::*;

#[component]
pub fn Sandbox() -> Element {
    rsx! {
        div {
            class: "sandbox",
            h1 { "Sandbox" }
            p { "This is the sandbox page." }
        }
    }
}
