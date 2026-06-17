use crate::assets::{BG_LUNAR_LANDSCAPE, BG_LUNAR_LANDSCAPE_PC};
use crate::components::{Divider, GlowingButton, GlowingSubtitle, GlowingTitle, Header, JourneyModal};
use dioxus::prelude::*;

#[cfg(feature = "web")]
async fn sleep_ms(ms: u32) {
    gloo_timers::future::TimeoutFuture::new(ms).await;
}

#[cfg(not(feature = "web"))]
async fn sleep_ms(ms: u32) {
    tokio::time::sleep(std::time::Duration::from_millis(ms as u64)).await;
}

#[component]
pub fn HeroSection() -> Element {
    let mut modal_open = use_signal(|| false);
    let mut is_diving = use_signal(|| false);
    let nav = use_navigator();

    let start_journey = move |_| {
        #[cfg(feature = "web")]
        {
            let _ = dioxus::document::eval(r#"
                if (!document.fullscreenElement) {
                    document.documentElement.requestFullscreen().catch(e => console.log(e));
                }
            "#);
        }
        #[cfg(not(feature = "web"))]
        let _ = ();

        is_diving.set(true);
        modal_open.set(false);

        spawn(async move {
            sleep_ms(1200).await;
            nav.push("/editor");
        });
    };

    let dive_class = if is_diving() { "is-diving" } else { "" };
    let bg_scale = if is_diving() { "scale-[3] blur-lg transition-all duration-[1500ms] ease-in" } else { "scale-100 transition-all duration-700" };
    let ui_opacity = if is_diving() { "opacity-0 scale-150 blur-xl transition-all duration-1000" } else { "opacity-100 scale-100" };

    rsx! {
        div {
            class: "h-screen w-full relative overflow-hidden bg-black transition-all duration-1000 {dive_class}",

            div { class: "absolute inset-0 z-0 {bg_scale}",
                img {
                    src: "{BG_LUNAR_LANDSCAPE}",
                    class: "absolute inset-0 w-full h-full object-cover object-bottom md:hidden",
                    alt: "lunar landscape mobile",
                }
                img {
                    src: "{BG_LUNAR_LANDSCAPE_PC}",
                    class: "absolute inset-0 w-full h-full object-cover object-center hidden md:block",
                    alt: "lunar landscape pc",
                }
                div { class: "absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent" }
            }

            div { class: "hard-glow-overlay" }

            if is_diving() {
                div { class: "absolute inset-0 z-30 pointer-events-none",
                    div { class: "warp-ray ray-1" }
                    div { class: "warp-ray ray-2" }
                    div { class: "warp-ray ray-3" }
                    div { class: "warp-ray ray-4" }
                    div { class: "warp-ray ray-5" }
                    div { class: "warp-ray ray-6" }
                }
            }

            div { class: "white-flash" }

            div { class: "relative z-40 flex flex-col items-center justify-between h-full py-8 {ui_opacity}",

                Header {}

                div { class: "flex flex-col items-center justify-center flex-grow plan-content",
                     GlowingTitle {}
                     Divider {}
                     GlowingSubtitle {}
                     GlowingButton {
                        on_click: move |_| {
                            modal_open.set(true);
                        },
                    }
                }
            }
        }

        if modal_open() {
            JourneyModal {
                on_continue: start_journey,
                on_close: move |_| modal_open.set(false),
            }
        }
    }
}