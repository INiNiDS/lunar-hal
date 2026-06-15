use crate::pages::{
    backend_api::BackendApi, dashboard::Dashboard, datasets::Datasets, models::Models,
    pipeline::Pipeline, siren_gallery::SirenGallery, training::Training,
    validation::Validation,
};
use dioxus::prelude::*;

pub mod ui;

pub use lunar_ui_shared::star_shader::StarShaderCanvas;

#[derive(Routable, Clone, PartialEq)]
pub enum Route {
    #[layout(ShellLayout)]
    #[route("/")]
    Dashboard {},
    #[route("/models")]
    Models {},
    #[route("/training")]
    Training {},
    #[route("/validation")]
    Validation {},
    #[route("/siren")]
    SirenGallery {},
    #[route("/pipeline")]
    Pipeline {},
    #[route("/backend")]
    BackendApi {},
    #[route("/datasets")]
    Datasets {},
}

#[component]
pub fn Shell() -> Element {
    rsx! {
        Router::<Route> {}
    }
}

#[component]
fn ShellLayout() -> Element {
    let route = use_route::<Route>();
    rsx! {
        div { class: "app-shell",
            Sidebar { current: route.clone() }
            main { class: "main",
                Outlet::<Route> {}
            }
        }
    }
}

#[component]
fn Sidebar(current: Route) -> Element {
    let path: &'static str = match &current {
        Route::Dashboard {} => "/",
        Route::Models {} => "/models",
        Route::Training {} => "/training",
        Route::Validation {} => "/validation",
        Route::SirenGallery {} => "/siren",
        Route::Pipeline {} => "/pipeline",
        Route::BackendApi {} => "/backend",
        Route::Datasets {} => "/datasets",
    };
    let cls = |target: &'static str| -> &'static str {
        if path == target { "active" } else { "" }
    };

    rsx! {
        aside { class: "sidebar",
            div { class: "sidebar-brand",
                div { class: "brand-mark" }
                div { class: "brand-text",
                    div { class: "title", "LUNAR" }
                    div { class: "sub", "Testbench v0.1" }
                }
            }
            div { class: "nav-group",
                div { class: "nav-group-label", "Overview" }
                Link { to: Route::Dashboard {}, class: "nav-link {cls(\"/\")}",
                    span { class: "dot" }
                    span { "Dashboard" }
                }
                Link { to: Route::BackendApi {}, class: "nav-link {cls(\"/backend\")}",
                    span { class: "dot" }
                    span { "Backend API" }
                }
                Link { to: Route::Datasets {}, class: "nav-link {cls(\"/datasets\")}",
                    span { class: "dot" }
                    span { "Datasets" }
                }
            }
            div { class: "nav-group",
                div { class: "nav-group-label", "Models" }
                Link { to: Route::Models {}, class: "nav-link {cls(\"/models\")}",
                    span { class: "dot" }
                    span { "Run Models" }
                }
                Link { to: Route::Pipeline {}, class: "nav-link {cls(\"/pipeline\")}",
                    span { class: "dot" }
                    span { "Pipeline" }
                }
                Link { to: Route::SirenGallery {}, class: "nav-link {cls(\"/siren\")}",
                    span { class: "dot" }
                    span { "SIREN Gallery" }
                }
            }
            div { class: "nav-group",
                div { class: "nav-group-label", "Training" }
                Link { to: Route::Training {}, class: "nav-link {cls(\"/training\")}",
                    span { class: "dot" }
                    span { "Train" }
                }
                Link { to: Route::Validation {}, class: "nav-link {cls(\"/validation\")}",
                    span { class: "dot" }
                    span { "Validation" }
                }
            }
        }
    }
}
