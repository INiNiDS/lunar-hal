use dioxus::prelude::*;

mod assets;
mod components;
pub mod api;

use components::{AboutPage, ContactPage, EditorPage, HeroSection};

const MAIN_CSS: Asset = asset!("/assets/main.css");
const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");

#[derive(Routable, Clone, PartialEq)]
enum Route {
    #[route("/")]
    HomePage {},
    #[route("/about")]
    AboutPage {},
    #[route("/contact")]
    ContactPage {},
    #[route("/editor")]
    EditorPage {},
}

fn main() {
    launch(App);
}

#[component]
fn HomePage() -> Element {
    rsx! {
        HeroSection {}
    }
}

#[component]
fn App() -> Element {
    rsx! {
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        document::Link { rel: "stylesheet", href: TAILWIND_CSS }
        document::Link { rel: "preconnect", href: "https://fonts.googleapis.com" }
        document::Link { rel: "preconnect", href: "https://fonts.gstatic.com" }
        document::Link {
            rel: "stylesheet",
            href: "https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Space+Grotesk:wght@300..700&display=swap",
        }
        Router::<Route> {}
    }
}
