use crate::os::category_lamp::CategoryLamp;
use crate::os::manifest::{AppCategory, AppIcon, apps_for_category};
use crate::os::use_os_state;
use dioxus::prelude::*;

#[component]
pub fn Desktop() -> Element {
    let mut os = use_os_state();
    rsx! {
        div { class: "desktop-surface",
            for category in AppCategory::ALL {
                section { class: "desktop-category", key: "{category.service()}",
                    CategoryLamp { category }
                    div { class: "desktop-icon-grid",
                        for (index, app) in apps_for_category(category)
                            .into_iter()
                            .filter(|app| os.is_app_visible(app.id) || os.has_window(app.id))
                            .enumerate()
                        {
                            {
                                let available = os.is_app_available(app.id);
                                let restorable = os.has_window(app.id);
                                let enabled = available || restorable;
                                let app_id = app.id;
                                let title = app.title;
                                let epoch = os.service_reveal_epoch(category.service());
                                rsx! {
                                    button {
                                        key: "{app_id}-{epoch}",
                                        class: if enabled {
                                            "desktop-app desktop-app-reveal group"
                                        } else {
                                            "desktop-app desktop-app-disabled"
                                        },
                                        "data-testid": "desktop-app-{app_id}",
                                        disabled: !enabled,
                                        style: "--app-index: {index};",
                                        title: if available {
                                            title.to_string()
                                        } else if restorable {
                                            format!("Restore {title} in paused mode")
                                        } else {
                                            format!("{title}: required services are not running")
                                        },
                                        // `open_window` restores an existing window before it
                                        // checks whether a new instance may be launched.
                                        onclick: move |_| os.open_window(app_id, title),
                                        div { class: "desktop-app-icon",
                                            AppIcon { app_id: app_id.to_string() }
                                        }
                                        span { class: "desktop-app-label", "{title}" }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
