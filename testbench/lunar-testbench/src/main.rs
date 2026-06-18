#![allow(non_snake_case)]

mod api;
mod components;
mod pages;

use components::Shell;
use dioxus::document;
use dioxus::prelude::*;

const MAIN_CSS: Asset = asset!("/assets/main.css");

fn main() {
    launch(App);
}

fn App() -> Element {
    rsx! {
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        Shell {}
    }
}
