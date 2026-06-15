mod api;
mod components;
mod pages;

use components::Shell;
use dioxus::prelude::*;
use dioxus::document;

const MAIN_CSS: Asset = asset!("/assets/main.css");

fn main() {
    launch(App);
}

#[allow(non_snake_case)]
fn App() -> Element {
    rsx! {
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        Shell {}
    }
}
