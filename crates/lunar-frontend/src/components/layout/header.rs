use dioxus::prelude::*;

#[component]
pub fn Header() -> Element {
    rsx! {
        header {
            class: "flex items-center justify-between bg-black/60 backdrop-blur-md text-neutral-400 px-8 py-3.5 rounded-full border border-white/10 max-w-6xl mx-auto my-6 transition-all duration-300",

            div { class: "flex items-center gap-6 font-light text-[11px] tracking-widest uppercase",
                a {
                    href: "/",
                    class: "hover:text-white transition-colors duration-300",
                    "Home"
                }
                a {
                    href: "/about",
                    class: "hover:text-white transition-colors duration-300",
                    "About"
                }
                a {
                    href: "/contact",
                    class: "hover:text-white transition-colors duration-300",
                    "Contact"
                }
            }

            div { class: "flex items-center gap-5 text-[11px] tracking-widest uppercase font-light",
                a {
                    href: "https://github.com/ininids/lunar-hal",
                    class: "hover:text-white transition-colors duration-300",
                    target: "_blank",
                    "GitHub"
                }
            }
        }
    }
}

