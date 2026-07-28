#![allow(non_snake_case)]

mod api;
mod components;
mod os;
mod pages;

use dioxus::document;
use dioxus::prelude::*;
use os::Room;

const MAIN_CSS: Asset = asset!("/assets/main.css");
// Compiled via `npm run build:css` / `npm run watch:css` from input.css + tailwind.config.js.
const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");

fn main() {
    launch(App);
}

fn App() -> Element {
    rsx! {
        document::Link {
            rel: "preconnect",
            href: "https://fonts.googleapis.com",
        }
        document::Link {
            rel: "preconnect",
            href: "https://fonts.gstatic.com",
            crossorigin: "true",
        }
        document::Link {
            rel: "stylesheet",
            href: "https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap",
        }
        // Tailwind first, then main.css so legacy admin-UI rules can still win
        // where pages haven't been migrated to the WebOS shell yet.
        document::Link { rel: "stylesheet", href: TAILWIND_CSS }
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        Room {}
    }
}
