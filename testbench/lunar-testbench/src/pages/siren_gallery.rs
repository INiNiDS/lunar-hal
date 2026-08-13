//! Persistent SIREN Gallery backed by `lunar-backend`.
//!
//! Unlike the old one-render preview, this page is a searchable catalog of
//! Gallery records. Creating a record asks the backend to render and atomically
//! save the texture before it appears here.

use dioxus::prelude::*;

use crate::api::{
    self, CreateGalleryStarRequest, GallerySource, GalleryStar, ResponseStar, StarModelInputs,
    UpdateGalleryStarRequest,
};
use crate::os::state::{is_window_lifecycle_visible, use_window_lifecycle};
use crate::os::{use_os_state, WindowLifecycle};

fn gallery_request_id() -> String {
    format!("gallery-ui-{}", js_sys::Date::now())
}

fn display_name(record: &GalleryStar) -> String {
    record
        .name
        .clone()
        .unwrap_or_else(|| record.star.name.clone())
}

fn source_label(source: &GallerySource) -> &'static str {
    match source {
        GallerySource::AdminGenerated => "Admin generated",
        GallerySource::SceneSaved => "Scene saved",
        GallerySource::Imported => "Imported",
    }
}

fn parse_tags(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|tag| !tag.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

#[component]
pub fn SirenGallery() -> Element {
    let mut os = use_os_state();
    let mut records = use_signal(Vec::<GalleryStar>::new);
    let mut selected = use_signal(|| None::<GalleryStar>);
    let mut query = use_signal(String::new);
    let mut sort = use_signal(|| "updated_desc".to_string());
    let mut refresh_tick = use_signal(|| 0_u32);
    let mut status = use_signal(|| None::<String>);
    let mut busy = use_signal(|| false);
    let lifecycle = use_window_lifecycle();
    let is_blocked = lifecycle
        .map(|signal| *signal.read() == WindowLifecycle::Blocked)
        .unwrap_or(false);

    let mut name = use_signal(|| "Generated star".to_string());
    let mut bp_rp = use_signal(|| 0.85_f32);
    let mut g_mag = use_signal(|| 4.83_f32);
    let mut temperature = use_signal(|| 5778.0_f32);
    let mut tags = use_signal(|| "generated, siren".to_string());

    let mut detail_name = use_signal(String::new);
    let mut detail_tags = use_signal(String::new);
    let mut detail_notes = use_signal(String::new);

    let listing_lifecycle = lifecycle;
    use_resource(move || {
        let search = query();
        let ordering = sort();
        let tick = refresh_tick();
        let visible = is_window_lifecycle_visible(listing_lifecycle);
        async move {
            let _ = tick;
            if !visible {
                return;
            }
            match api::list_gallery_stars(None, 72, Some(&ordering), Some(&search)).await {
                Ok(list) => records.set(list.stars),
                Err(error) => status.set(Some(error)),
            }
        }
    });

    let mut choose = move |record: GalleryStar| {
        detail_name.set(display_name(&record));
        detail_tags.set(record.tags.join(", "));
        detail_notes.set(record.notes.clone().unwrap_or_default());
        selected.set(Some(record));
    };

    let generate_and_save = move |_| {
        busy.set(true);
        status.set(None);
        let star = ResponseStar {
            id: 0,
            x: 10.0,
            y: 0.0,
            z: 0.0,
            temperature_k: temperature().max(1.0),
            radius: 1.0,
            mass: 1.0,
            luminosity: 1.0,
            description: "SIREN Gallery generation".into(),
            name: name().trim().to_string(),
            type_hint: "generated".into(),
            velocity_vector: [0.0; 3],
        };
        let request = CreateGalleryStarRequest {
            request_id: gallery_request_id(),
            source: GallerySource::AdminGenerated,
            inputs: StarModelInputs {
                x_pc: star.x,
                y_pc: star.y,
                z_pc: star.z,
                bp_rp: bp_rp(),
                g_mag: g_mag(),
                entropy_temperature: None,
            },
            star,
            pinn: None,
            metadata: None,
            name: Some(name().trim().to_string()),
            tags: parse_tags(&tags()),
            notes: None,
        };
        spawn(async move {
            match api::create_gallery_star(&request).await {
                Ok(record) => {
                    detail_name.set(display_name(&record));
                    detail_tags.set(record.tags.join(", "));
                    detail_notes.set(record.notes.clone().unwrap_or_default());
                    selected.set(Some(record));
                    refresh_tick.set(refresh_tick().wrapping_add(1));
                    status.set(Some("Texture generated and saved to the persistent Gallery.".into()));
                }
                Err(error) => status.set(Some(error)),
            }
            busy.set(false);
        });
    };

    let save_details = move |_| {
        let Some(record) = selected() else { return };
        busy.set(true);
        let id = record.id.clone();
        let request = UpdateGalleryStarRequest {
            name: Some(detail_name().trim().to_string()),
            tags: Some(parse_tags(&detail_tags())),
            notes: Some(detail_notes().trim().to_string()),
        };
        spawn(async move {
            match api::update_gallery_star(&id, &request).await {
                Ok(updated) => {
                    selected.set(Some(updated));
                    refresh_tick.set(refresh_tick().wrapping_add(1));
                    status.set(Some("Gallery metadata saved.".into()));
                }
                Err(error) => status.set(Some(error)),
            }
            busy.set(false);
        });
    };

    let delete_selected = move |_| {
        let Some(record) = selected() else { return };
        busy.set(true);
        let id = record.id;
        spawn(async move {
            match api::delete_gallery_star(&id).await {
                Ok(()) => {
                    selected.set(None);
                    refresh_tick.set(refresh_tick().wrapping_add(1));
                    status.set(Some("Gallery record deleted.".into()));
                }
                Err(error) => status.set(Some(error)),
            }
            busy.set(false);
        });
    };

    use_effect(move || {
        if let Some(lifecycle) = lifecycle {
            if *lifecycle.read() == WindowLifecycle::Visible {
                refresh_tick.set(refresh_tick().wrapping_add(1));
            }
        }
    });

    let controls_disabled = is_blocked || busy();
    let open_sandbox = move |_| os.open_window("sandbox", "Sandbox");
    let selected_thumbnail = selected()
        .as_ref()
        .and_then(|record| record.thumbnail_url.as_ref())
        .map(|_| api::gallery_thumbnail_url(&selected().expect("selected record").id));

    rsx! {
        div { class: "page space-y-4",
            div { class: "flex flex-wrap items-end justify-between gap-3",
                div { h1 { class: "text-xl font-semibold text-white", "SIREN Gallery" } p { class: "mt-1 max-w-2xl text-sm text-white/55", "Persistent stellar records, model inputs, metadata, and backend-rendered previews." } }
                if busy() { span { class: "rounded-lg bg-amber-400/15 px-3 py-2 text-xs text-amber-200", "Saving…" } }
            }
            if let Some(message) = status() { p { class: "rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-white/70", "{message}" } }

            div { class: "gallery-layout",
                // Generation panel: a real POST /gallery/stars flow, not a temporary PNG.
                section { class: "card gallery-builder space-y-3",
                    h2 { class: "card-title", "Generate & save" }
                    label { class: "block text-xs text-white/55", "Name" input { class: "mt-1 w-full rounded-lg bg-black/35 px-2 py-1.5 text-sm", value: "{name()}", oninput: move |event| name.set(event.value()) } }
                    div { class: "grid grid-cols-2 gap-2",
                        label { class: "text-xs text-white/55", "Bp–Rp" input { class: "mt-1 w-full rounded-lg bg-black/35 px-2 py-1.5 text-sm", r#type: "number", step: "0.05", value: "{bp_rp()}", oninput: move |event| if let Ok(value) = event.value().parse() { bp_rp.set(value); } } }
                        label { class: "text-xs text-white/55", "G magnitude" input { class: "mt-1 w-full rounded-lg bg-black/35 px-2 py-1.5 text-sm", r#type: "number", step: "0.1", value: "{g_mag()}", oninput: move |event| if let Ok(value) = event.value().parse() { g_mag.set(value); } } }
                    }
                    label { class: "block text-xs text-white/55", "Temperature (K)" input { class: "mt-1 w-full rounded-lg bg-black/35 px-2 py-1.5 text-sm", r#type: "number", value: "{temperature()}", oninput: move |event| if let Ok(value) = event.value().parse() { temperature.set(value); } } }
                    label { class: "block text-xs text-white/55", "Tags" input { class: "mt-1 w-full rounded-lg bg-black/35 px-2 py-1.5 text-sm", value: "{tags()}", oninput: move |event| tags.set(event.value()) } }
                    button { class: "w-full rounded-lg bg-violet-500/35 px-3 py-2 text-sm font-medium text-white hover:bg-violet-500/55", disabled: controls_disabled, onclick: generate_and_save, "Generate & save" }
                    p { class: "text-[11px] leading-relaxed text-white/40", "The backend produces the SIREN PNG, writes metadata atomically, and returns a durable Gallery URL." }
                }

                section { class: "gallery-content",
                    div { class: "flex flex-wrap items-center gap-2",
                        input { class: "min-w-40 flex-1 rounded-lg border border-white/10 bg-black/35 px-3 py-2 text-sm", placeholder: "Search name, tag, or id", value: "{query()}", oninput: move |event| query.set(event.value()) }
                        select { class: "rounded-lg border border-white/10 bg-black/35 px-2 py-2 text-xs", value: "{sort()}", onchange: move |event| sort.set(event.value()),
                            option { value: "updated_desc", "Recently updated" }
                            option { value: "created_asc", "Oldest first" }
                            option { value: "name", "Name" }
                        }
                    }
                    if records().is_empty() {
                        div { class: "rounded-2xl border border-dashed border-white/15 p-10 text-center text-sm text-white/45", "No saved stars match this view." }
                    } else {
                        div { class: "gallery-grid",
                            for record in records() {
                                {
                                    let label = display_name(&record);
                                    let id = record.id.clone();
                                    let thumbnail = record.thumbnail_url.as_ref().map(|_| api::gallery_thumbnail_url(&id));
                                    let source = source_label(&record.source);
                                    rsx! {
                                        button { class: "group overflow-hidden rounded-xl border border-white/10 bg-white/[0.03] text-left focus:outline-none focus:ring-2 focus:ring-violet-400/70", onclick: move |_| choose(record.clone()),
                                            div { class: "aspect-square bg-gradient-to-br from-violet-500/20 via-slate-900 to-cyan-500/10",
                                                if let Some(src) = thumbnail { img { class: "h-full w-full object-cover", src: "{src}", alt: "Texture preview for {label}" } }
                                            }
                                            div { class: "space-y-1 p-2", p { class: "truncate text-sm font-medium text-white/85", "{label}" } p { class: "truncate text-[10px] uppercase tracking-wider text-white/40", "{source}" } }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                aside { class: "card gallery-detail min-h-64 space-y-3",
                    h2 { class: "card-title", "Details" }
                    if let Some(record) = selected() {
                        if let Some(src) = selected_thumbnail { img { class: "aspect-square w-full rounded-xl border border-white/10 object-cover", src: "{src}", alt: "Selected texture preview" } }
                        label { class: "block text-xs text-white/55", "Name" input { class: "mt-1 w-full rounded-lg bg-black/35 px-2 py-1.5 text-sm", value: "{detail_name()}", oninput: move |event| detail_name.set(event.value()) } }
                        label { class: "block text-xs text-white/55", "Tags" input { class: "mt-1 w-full rounded-lg bg-black/35 px-2 py-1.5 text-sm", value: "{detail_tags()}", oninput: move |event| detail_tags.set(event.value()) } }
                        label { class: "block text-xs text-white/55", "Notes" textarea { class: "mt-1 min-h-20 w-full rounded-lg bg-black/35 px-2 py-1.5 text-sm", value: "{detail_notes()}", oninput: move |event| detail_notes.set(event.value()) } }
                        div { class: "text-[11px] text-white/45", p { "{record.star.temperature_k:.0} K · {record.star.type_hint}" } p { "ID {record.id}" } }
                        div { class: "flex flex-wrap gap-2", button { class: "rounded-lg bg-sky-500/25 px-2 py-1.5 text-xs hover:bg-sky-500/40", disabled: controls_disabled, onclick: save_details, "Save" } button { class: "rounded-lg bg-emerald-500/20 px-2 py-1.5 text-xs hover:bg-emerald-500/35", disabled: controls_disabled, onclick: open_sandbox, "Open in Sandbox" } button { class: "rounded-lg bg-red-500/20 px-2 py-1.5 text-xs hover:bg-red-500/35", disabled: controls_disabled, onclick: delete_selected, "Delete" } }
                        p { class: "text-[10px] leading-relaxed text-white/35", "Keyboard/touch fallback: select this card and use Open in Sandbox; shared drag payloads are reserved for the following drag-and-drop stage." }
                    } else {
                        p { class: "text-sm text-white/45", "Select a saved star to inspect its texture and edit its metadata." }
                    }
                }
            }
        }
    }
}
