use dioxus::prelude::*;

#[component]
pub fn Enemy(
    x: f64,
    y: f64,
    on_mouse_down: EventHandler<MouseEvent>,
    on_click: EventHandler<MouseEvent>,
) -> Element {
    rsx! {
        style {
            r#"
            .enemy-container {{
                position: absolute;
                width: 60px;
                height: 60px;
                display: flex;
                justify-content: center;
                align-items: center;
                pointer-events: auto;
                cursor: grab;
            }}
            .enemy-container:active {{
                cursor: grabbing;
            }}
            .rotate-clockwise {{
                animation: spin-cw 10s linear infinite;
                transform-origin: center;
            }}
            .rotate-counter {{
                animation: spin-ccw 6s linear infinite;
                transform-origin: center;
            }}
            .pulse-core {{
                animation: pulse 1.8s ease-in-out infinite;
                transform-origin: center;
            }}
            @keyframes spin-cw {{
                from {{ transform: rotate(0deg); }}
                to {{ transform: rotate(360deg); }}
            }}
            @keyframes spin-ccw {{
                from {{ transform: rotate(0deg); }}
                to {{ transform: rotate(-360deg); }}
            }}
            @keyframes pulse {{
                0%, 100% {{ transform: scale(0.85); opacity: 0.7; }}
                50% {{ transform: scale(1.15); opacity: 1; }}
            }}
            "#
        }

        div {
            class: "enemy-container",
            style: "left: {x}px; top: {y}px; transform: translate(-50%, -50%);",
            onmousedown: move |e| on_mouse_down.call(e),
            onclick: move |e| on_click.call(e),

            svg {
                view_box: "-30 -30 60 60",
                width: "60",
                height: "60",

                circle {
                    class: "rotate-counter",
                    cx: "0",
                    cy: "0",
                    r: "22",
                    fill: "none",
                    stroke: "rgba(255, 51, 51, 0.45)",
                    stroke_width: "1",
                    stroke_dasharray: "6, 5",
                }

                g {
                    class: "rotate-clockwise",
                    style: "filter: drop-shadow(0 0 5px rgba(255, 51, 51, 0.8));",

                    path {
                        d: "M 0,-15 L -11,0 L 0,15 L -3,0 Z",
                        fill: "none",
                        stroke: "#ff3333",
                        stroke_width: "1.5",
                    }
                    path {
                        d: "M 0,-15 L 11,0 L 0,15 L 3,0 Z",
                        fill: "none",
                        stroke: "#ff3333",
                        stroke_width: "1.5",
                    }
                }

                circle {
                    class: "pulse-core",
                    cx: "0",
                    cy: "0",
                    r: "3",
                    fill: "#ffffff",
                    style: "filter: drop-shadow(0 0 6px #ff3333);",
                }
            }
        }
    }
}
