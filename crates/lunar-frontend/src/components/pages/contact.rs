use crate::assets::{FONT_SANS, FONT_SERIF};
use crate::components::Header;
use dioxus::prelude::*;

#[component]
pub fn ContactPage() -> Element {
    rsx! {
        div { class: "min-h-screen w-full relative overflow-hidden bg-[#0f1116]",

            div { class: "absolute top-0 left-1/2 -translate-x-1/2 w-[900px] h-[700px] \
                       bg-white/[0.02] blur-[130px] rounded-full pointer-events-none" }
            div { class: "absolute bottom-0 right-1/4 w-[600px] h-[400px] \
                       bg-white/[0.015] blur-[100px] rounded-full pointer-events-none" }

            Header {}

            div { class: "relative z-10 flex flex-col items-center justify-center \
                       min-h-[calc(100vh-120px)] px-8 py-16",

                h1 {
                    class: "font-serif-glow text-5xl md:text-6xl text-white \
                           drop-shadow-[0_0_15px_rgba(255,255,255,0.7)] \
                           drop-shadow-[0_0_35px_rgba(255,255,255,0.3)] \
                           select-none tracking-[0.15em] mb-16",
                    style: "font-family: {FONT_SERIF}",
                    "CONTACT"
                }

                div { class: "grid grid-cols-1 md:grid-cols-3 gap-5 max-w-4xl w-full",

                    // GitHub
                    div { class: "group bg-white/[0.003] backdrop-blur-sm border border-white/[0.008] \
                               rounded-xl p-7 text-center \
                               hover:bg-white/[0.025] hover:border-white/[0.05] \
                               hover:shadow-[0_0_40px_rgba(255,255,255,0.02)] \
                               transition-all duration-500",

                        div { class: "mb-5 inline-flex items-center justify-center w-14 h-14 \
                                   rounded-full bg-white/[0.015] border border-white/[0.03] \
                                   group-hover:bg-white/[0.06] group-hover:border-white/15 \
                                   group-hover:shadow-[0_0_20px_rgba(255,255,255,0.08)] \
                                   transition-all duration-500",

                            svg {
                                class: "w-6 h-6 text-white/50 group-hover:text-white/90 \
                                       transition-colors duration-500",
                                view_box: "0 0 24 24",
                                fill: "currentColor",
                                path {
                                    d: "M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 \
                                       11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 \
                                       1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 \
                                       3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 \
                                       1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 \
                                       5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 \
                                       2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 \
                                       5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 \
                                       24 12c0-6.627-5.373-12-12-12z",
                                }
                            }
                        }

                        h3 {
                            class: "text-white/70 group-hover:text-white/90 text-[11px] uppercase tracking-[0.3em] mb-3 \
                                   font-semibold transition-all duration-500",
                            style: "font-family: {FONT_SANS}",
                            "GitHub"
                        }
                        p {
                            class: "text-white/30 group-hover:text-white/50 text-[13px] leading-relaxed mb-5 \
                                   transition-all duration-500",
                            style: "font-family: {FONT_SANS}",
                            "Explore the source code, open issues, or contribute to the project."
                        }
                        a {
                            class: "inline-block text-white/50 text-[11px] uppercase tracking-[0.15em] \
                                   border border-white/15 rounded-full px-5 py-2 \
                                   hover:text-white hover:border-white/30 \
                                   hover:shadow-[0_0_15px_rgba(255,255,255,0.15)] \
                                   transition-all duration-300",
                            style: "font-family: {FONT_SANS}",
                            href: "https://github.com/ininids/lunar-hal",
                            target: "_blank",
                            "Open Repository"
                        }
                    }

                    // Email
                    div { class: "group bg-white/[0.003] backdrop-blur-sm border border-white/[0.008] \
                               rounded-xl p-7 text-center \
                               hover:bg-white/[0.025] hover:border-white/[0.05] \
                               hover:shadow-[0_0_40px_rgba(255,255,255,0.02)] \
                               transition-all duration-500",

                        div { class: "mb-5 inline-flex items-center justify-center w-14 h-14 \
                                   rounded-full bg-white/[0.015] border border-white/[0.03] \
                                   group-hover:bg-white/[0.06] group-hover:border-white/15 \
                                   group-hover:shadow-[0_0_20px_rgba(255,255,255,0.08)] \
                                   transition-all duration-500",

                            svg {
                                class: "w-6 h-6 text-white/50 group-hover:text-white/90 \
                                       transition-colors duration-500",
                                view_box: "0 0 24 24",
                                fill: "none",
                                stroke: "currentColor",
                                stroke_width: "1.5",
                                stroke_linecap: "round",
                                stroke_linejoin: "round",
                                path { d: "M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 \
                                       0-2-.9-2-2V6c0-1.1.9-2 2-2z" }
                                polyline { points: "22,6 12,13 2,6" }
                            }
                        }

                        h3 {
                            class: "text-white/70 group-hover:text-white/90 text-[11px] uppercase tracking-[0.3em] mb-3 \
                                   font-semibold transition-all duration-500",
                            style: "font-family: {FONT_SANS}",
                            "Email"
                        }
                        p {
                            class: "text-white/30 group-hover:text-white/50 text-[13px] leading-relaxed mb-5 \
                                   transition-all duration-500",
                            style: "font-family: {FONT_SANS}",
                            "Have questions or want to collaborate? Get in touch directly."
                        }
                        a {
                            class: "inline-block text-white/50 text-[11px] uppercase tracking-[0.15em] \
                                   border border-white/15 rounded-full px-5 py-2 \
                                   hover:text-white hover:border-white/30 \
                                   hover:shadow-[0_0_15px_rgba(255,255,255,0.15)] \
                                   transition-all duration-300",
                            style: "font-family: {FONT_SANS}",
                            href: "mailto:ininids@ininids.in.rs",
                            "Send Message"
                        }
                    }

                    div { class: "group bg-white/[0.003] backdrop-blur-sm border border-white/[0.008] \
                               rounded-xl p-7 text-center \
                               hover:bg-white/[0.025] hover:border-white/[0.05] \
                               hover:shadow-[0_0_40px_rgba(255,255,255,0.02)] \
                               transition-all duration-500",

                        div { class: "mb-5 inline-flex items-center justify-center w-14 h-14 \
                                   rounded-full bg-white/[0.015] border border-white/[0.03] \
                                   group-hover:bg-white/[0.06] group-hover:border-white/15 \
                                   group-hover:shadow-[0_0_20px_rgba(255,255,255,0.08)] \
                                   transition-all duration-500",

                            svg {
                                class: "w-6 h-6 text-white/50 group-hover:text-white/90 \
                                       transition-colors duration-500",
                                view_box: "0 0 24 24",
                                fill: "none",
                                stroke: "currentColor",
                                stroke_width: "1.5",
                                stroke_linecap: "round",
                                stroke_linejoin: "round",
                                circle { cx: "12", cy: "12", r: "10" }
                                line {
                                    x1: "2",
                                    y1: "12",
                                    x2: "22",
                                    y2: "12",
                                }
                                path { d: "M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 \
                                       15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z" }
                            }
                        }

                        h3 {
                            class: "text-white/70 group-hover:text-white/90 text-[11px] uppercase tracking-[0.3em] mb-3 \
                                   font-semibold transition-all duration-500",
                            style: "font-family: {FONT_SANS}",
                            "Connect"
                        }
                        p {
                            class: "text-white/30 group-hover:text-white/50 text-[13px] leading-relaxed mb-5 \
                                   transition-all duration-500",
                            style: "font-family: {FONT_SANS}",
                            "Follow the mission. Updates, milestones, and deep-space engineering logs."
                        }
                        a {
                            class: "inline-block text-white/50 text-[11px] uppercase tracking-[0.15em] \
                                   border border-white/15 rounded-full px-5 py-2 \
                                   hover:text-white hover:border-white/30 \
                                   hover:shadow-[0_0_15px_rgba(255,255,255,0.15)] \
                                   transition-all duration-300",
                            style: "font-family: {FONT_SANS}",
                            href: "https://github.com/ininids/lunar-hal/discussions",
                            target: "_blank",
                            "Join Discussions"
                        }
                    }
                }
            }
        }
    }
}