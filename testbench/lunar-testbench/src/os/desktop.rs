use crate::os::category_lamp::CategoryLamp;
use crate::os::manifest::{AppCategory, AppIcon, apps_for_category};
use crate::os::use_os_state;
use dioxus::prelude::*;

#[component]
pub fn Desktop() -> Element {
    let mut os = use_os_state();
    rsx! { div { class: "desktop-surface",
        for category in AppCategory::ALL {
            section { class: "desktop-category", key: "{category.service()}",
                CategoryLamp { category }
                div { class: "desktop-icon-grid",
                    for (index, app) in apps_for_category(category).into_iter().filter(|app| os.is_app_visible(app.id)).enumerate() {
                        { let available=os.is_app_available(app.id); let app_id=app.id; let title=app.title; let epoch=os.service_reveal_epoch(category.service()); rsx! {
                            button { key: "{app_id}-{epoch}", class: if available { "desktop-app desktop-app-reveal group" } else { "desktop-app desktop-app-disabled" }, disabled: !available, style: "--app-index: {index};", title: if available { title.to_string() } else { format!("{title}: required services are not running") },
                                onclick: move |_| if os.is_app_available(app_id) { os.open_window(app_id,title); },
                                div { class: "desktop-app-icon", AppIcon { app_id: app_id.to_string() } }
                                span { class: "desktop-app-label", "{title}" }
                            }
                        }}
                    }
                }
            }
        }
    }}
}
