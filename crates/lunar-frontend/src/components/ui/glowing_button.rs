use dioxus::prelude::*;

#[component]
pub fn GlowingButton(on_click: EventHandler<()>) -> Element {
    rsx! {
        button {
            class: "absolute bottom-[35%] left-1/2 -translate-x-1/2 \
                   rounded-full border border-white/80 \
                   bg-black/40 backdrop-blur-sm \
                   px-12 py-4 \
                   uppercase tracking-[0.2em] text-xs font-semibold text-white \
                   shadow-[0_0_15px_rgba(255,255,255,0.4)] \
                   hover:bg-white/10 hover:shadow-[0_0_25px_rgba(255,255,255,0.7)] \
                   transition-all duration-300 \
                   cursor-pointer select-none",
            onclick: move |_| on_click.call(()),
            "START JOURNEY",
        }
    }
}

