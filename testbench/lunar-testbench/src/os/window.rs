use dioxus::prelude::*;

use crate::os::app_host::AppHost;
use crate::os::manifest::AppIcon;
use crate::os::{DragKind, OsState, WindowState, use_os_state, viewport_size};

#[component]
pub fn Window(win: WindowState) -> Element {
    let mut os = use_os_state();

    let id = win.id;
    let mut style = format!(
        "left: {}px; top: {}px; width: {}px; height: {}px; z-index: {};",
        win.x, win.y, win.width, win.height, win.z
    );

    if win.minimized {
        style.push_str(" display: none; visibility: hidden; pointer-events: none;");
    }

    let title = win.title.clone();
    let app_id = win.app_id.clone();

    let radius = if win.maximized {
        "absolute pointer-events-auto glass-strong flex flex-col overflow-hidden animate-window-open"
    } else {
        "absolute pointer-events-auto glass-strong rounded-xl flex flex-col overflow-hidden animate-window-open"
    };

    rsx! {
        div {
            class: "{radius}",
            style: "{style}",
            "data-app-id": "{app_id}",
            "data-testid": "webos-window",
            aria_hidden: if win.minimized { "true" } else { "false" },
            onmousedown: move |_| os.focus_window(id),

            div {
                class: "flex items-center h-9 pl-3 gap-2.5 shrink-0 border-b border-white/[0.08] bg-gradient-to-b from-white/[0.09] to-white/[0.02] cursor-grab active:cursor-grabbing",
                "data-testid": "window-titlebar",
                onmousedown: move |e| {
                    let p = e.client_coordinates();
                    os.begin_drag(id, DragKind::Move, p.x, p.y);
                },
                ondoubleclick: move |_| {
                    let (vw, vh) = viewport_size();
                    os.toggle_maximize_window(id, vw, vh);
                },

                div { class: "w-3.5 h-3.5 shrink-0 text-white/55",
                    AppIcon { app_id: app_id.clone() }
                }
                span { class: "flex-1 min-w-0 truncate font-grotesk text-[12px] tracking-wide text-white/80",
                    "{title}"
                }

                div { class: "flex items-stretch h-full shrink-0",
                    button {
                        class: "w-11 grid place-items-center text-white/55 hover:text-white hover:bg-white/10 transition-colors",
                        title: "Minimize",
                        onmousedown: move |e| e.stop_propagation(),
                        onclick: move |_| os.minimize_window(id),
                        svg { class: "w-2.5 h-2.5", view_box: "0 0 10 10", stroke: "currentColor", stroke_width: "1", path { d: "M0 5h10" } }
                    }
                    button {
                        class: "w-11 grid place-items-center text-white/55 hover:text-white hover:bg-white/10 transition-colors",
                        title: "Maximize",
                        onmousedown: move |e| e.stop_propagation(),
                        onclick: move |_| {
                            let (vw, vh) = viewport_size();
                            os.toggle_maximize_window(id, vw, vh);
                        },
                        svg { class: "w-2.5 h-2.5", view_box: "0 0 10 10", fill: "none", stroke: "currentColor", stroke_width: "1", rect { x: "0.5", y: "0.5", width: "9", height: "9" } }
                    }
                    button {
                        class: "w-11 grid place-items-center rounded-tr-xl text-white/55 hover:text-white hover:bg-err transition-colors",
                        title: "Close",
                        onmousedown: move |e| e.stop_propagation(),
                        onclick: move |_| os.close_window(id),
                        svg { class: "w-2.5 h-2.5", view_box: "0 0 10 10", stroke: "currentColor", stroke_width: "1", stroke_linecap: "round", path { d: "M0.5 0.5l9 9M9.5 0.5l-9 9" } }
                    }
                }
            }

            AppHost {
                window_id: id,
                app_id: app_id.clone(),
                minimized: win.minimized,
                title: title,
            }

            {resize_handle(os, id, "absolute right-0 top-1.5 bottom-1.5 w-1.5 cursor-ew-resize", DragKind::ResizeE)}
            {resize_handle(os, id, "absolute left-0 top-1.5 bottom-1.5 w-1.5 cursor-ew-resize", DragKind::ResizeW)}
            {resize_handle(os, id, "absolute left-1.5 right-1.5 bottom-0 h-1.5 cursor-ns-resize", DragKind::ResizeS)}
            {resize_handle(os, id, "absolute left-1.5 right-1.5 top-0 h-1.5 cursor-ns-resize", DragKind::ResizeN)}
            {resize_handle(os, id, "absolute right-0 bottom-0 w-3 h-3 cursor-nwse-resize", DragKind::ResizeSE)}
            {resize_handle(os, id, "absolute left-0 bottom-0 w-3 h-3 cursor-nesw-resize", DragKind::ResizeSW)}
            {resize_handle(os, id, "absolute right-0 top-0 w-3 h-3 cursor-nesw-resize", DragKind::ResizeNE)}
            {resize_handle(os, id, "absolute left-0 top-0 w-3 h-3 cursor-nwse-resize", DragKind::ResizeNW)}
        }
    }
}

fn resize_handle(mut os: OsState, id: u64, class: &'static str, kind: DragKind) -> Element {
    rsx! {
        div {
            class: "{class}",
            onmousedown: move |e| {
                e.stop_propagation();
                let p = e.client_coordinates();
                os.begin_drag(id, kind, p.x, p.y);
            },
        }
    }
}
