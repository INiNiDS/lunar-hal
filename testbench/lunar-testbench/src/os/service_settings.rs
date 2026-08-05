use crate::api::{FieldType, ServiceConfigField, ServiceStatus};
use crate::os::use_os_state;
use dioxus::prelude::*;
fn advanced(field: &ServiceConfigField) -> bool {
    matches!(
        field.key.as_str(),
        "LUNAR_ENV" | "EXTRA_ARGS" | "BUILD_ARGS"
    ) || field.is_build_param
}
#[component]
pub fn ServiceSettings() -> Element {
    let mut os = use_os_state();
    let Some(state) = os.service_settings.read().clone() else {
        return rsx! {};
    };
    let running = os
        .service_status(&state.service)
        .as_ref()
        .is_some_and(ServiceStatus::is_running);
    let meta = os.service_meta(&state.service);
    let title = meta
        .as_ref()
        .map(|m| m.title.clone())
        .unwrap_or_else(|| state.service.clone());
    let description = meta
        .as_ref()
        .map(|m| m.description.clone())
        .unwrap_or_default();
    let busy = state.busy();
    rsx! {div{class:"fixed inset-0 z-[1200] grid place-items-center bg-black/70 px-4 py-8 backdrop-blur-sm",onclick:move |_|if !busy{os.close_service_settings();},
     div{class:"glass-strong flex max-h-full w-full max-w-2xl flex-col overflow-hidden rounded-2xl",onclick:move|e|e.stop_propagation(),
      header{class:"flex items-start justify-between border-b border-white/[0.08] px-6 py-5",div{p{class:"font-mono text-[9px] uppercase tracking-[0.25em] text-white/30","Service configuration"}h2{class:"mt-1 font-display text-lg tracking-[0.08em] text-white/90","{title}"}p{class:"mt-1 text-[11px] leading-relaxed text-white/35","{description}"}if running{p{class:"mt-2 font-mono text-[10px] text-warn/80","Running — effective configuration is read-only. Stop before changing it."}}}button{class:"h-8 w-8 rounded-lg text-white/35 hover:bg-white/10 hover:text-white",disabled:busy,onclick:move |_|os.close_service_settings(),"×"}}
      div{class:"scrollbar-thin flex-1 overflow-auto px-6 py-5",
       if state.loading{div{class:"py-16 text-center font-mono text-xs text-white/35 animate-pulse","Loading schema and configuration…"}}
       else if let Some(schema)=&state.schema{
        div{class:"grid grid-cols-1 gap-4 sm:grid-cols-2",for field in schema.fields.iter().filter(|f|!advanced(f)){SettingsField{key:"{field.key}",field:field.clone(),value:state.form_values.get(&field.key).cloned().unwrap_or_default(),error:state.client_field_errors.get(&field.key).or_else(||state.server_field_errors.get(&field.key)).cloned(),disabled:running||busy}}}
        if schema.fields.iter().any(advanced){details{class:"mt-6 rounded-xl border border-white/[0.08] bg-black/15",summary{class:"cursor-pointer px-4 py-3 font-mono text-[10px] uppercase tracking-[0.18em] text-white/45","Advanced"}div{class:"grid grid-cols-1 gap-4 border-t border-white/[0.06] p-4",for field in schema.fields.iter().filter(|f|advanced(f)){SettingsField{key:"{field.key}",field:field.clone(),value:state.form_values.get(&field.key).cloned().unwrap_or_default(),error:state.client_field_errors.get(&field.key).or_else(||state.server_field_errors.get(&field.key)).cloned(),disabled:running||busy}}}}}
       }
       if let Some(error)=&state.error{div{class:"mt-5 rounded-lg border border-err/25 bg-err/10 px-4 py-3 font-mono text-[11px] text-err/90","{error}"}}
      }
      footer{class:"flex items-center justify-between gap-3 border-t border-white/[0.08] px-6 py-4",button{class:"rounded-lg px-3 py-2 font-mono text-[10px] uppercase tracking-[0.12em] text-white/35 hover:bg-white/[0.06] disabled:opacity-30",disabled:busy||running||state.schema.is_none(),onclick:move |_|os.reset_service_settings(),"Reset defaults"}div{class:"flex gap-2",button{class:"rounded-lg px-4 py-2 text-xs text-white/50 hover:bg-white/[0.07] disabled:opacity-30",disabled:busy,onclick:move |_|os.close_service_settings(),if running{"Close"}else{"Cancel"}}if !running{button{class:"rounded-lg border border-accent/35 bg-accent/15 px-5 py-2 text-xs font-medium text-accent hover:bg-accent/25 disabled:cursor-wait disabled:opacity-40",disabled:busy||state.schema.is_none(),onclick:move |_|os.validate_and_start_service(),if state.validating{"Validating…"}else if state.saving{"Saving…"}else if state.starting{"Starting…"}else{"Save & start"}}}}}
     }
    }}
}
#[component]
fn SettingsField(
    field: ServiceConfigField,
    value: String,
    error: Option<String>,
    disabled: bool,
) -> Element {
    let mut os = use_os_state();
    let key = field.key.clone();
    let key_input = field.key.clone();
    let key_toggle = field.key.clone();
    let readonly = disabled || field.read_only;
    let field_class = if error.is_some() {
        "w-full rounded-lg border border-err/50 bg-black/30 px-3 py-2 font-mono text-[11px] text-white/85 outline-none"
    } else {
        "w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 font-mono text-[11px] text-white/85 outline-none focus:border-accent/50"
    };
    rsx! {
        label { class: "block min-w-0",
            span { class: "mb-1.5 flex items-center gap-2 text-[11px] text-white/65",
                "{field.label}"
                if field.required { span { class: "text-accent/70", "*" } }
                if field.read_only { span { class: "font-mono text-[8px] uppercase text-white/25", "read only" } }
            }
            match &field.field_type {
                FieldType::Select { options } => rsx! {
                    select { class: "{field_class}", disabled: readonly, value: "{value}", oninput: move |event| os.update_service_setting(&key_input, event.value()),
                        for option in options { option { value: "{option}", selected: *option == value, "{option}" } }
                    }
                },
                FieldType::Boolean => { let enabled = value == "true"; rsx! {
                    button { r#type: "button", class: if enabled { "flex w-full justify-between rounded-lg border border-ok/30 bg-ok/10 px-3 py-2 font-mono text-[11px] text-ok" } else { "flex w-full justify-between rounded-lg border border-white/10 bg-black/30 px-3 py-2 font-mono text-[11px] text-white/40" }, disabled: readonly, onclick: move |_| os.update_service_setting(&key_toggle, (!enabled).to_string()),
                        span { if enabled { "Enabled" } else { "Disabled" } }
                        span { class: if enabled { "h-4 w-7 rounded-full bg-ok/50 p-0.5" } else { "h-4 w-7 rounded-full bg-white/10 p-0.5" }, span { class: if enabled { "block h-3 w-3 translate-x-3 rounded-full bg-white" } else { "block h-3 w-3 rounded-full bg-white/50" } } }
                    }
                }},
                FieldType::StringList => rsx! { textarea { class: "{field_class} min-h-20 resize-y", disabled: readonly, value: "{value}", placeholder: "One argument per line", oninput: move |event| os.update_service_setting(&key, event.value()) } },
                _ => rsx! { input { class: "{field_class}", disabled: readonly, r#type: if matches!(field.field_type, FieldType::Port) { "number" } else { "text" }, min: field.min.map(|value| value.to_string()), max: field.max.map(|value| value.to_string()), value: "{value}", oninput: move |event| os.update_service_setting(&key_input, event.value()) } },
            }
            if let Some(description) = &field.description { span { class: "mt-1 block text-[10px] leading-relaxed text-white/25", "{description}" } }
            span { class: "mt-1 block font-mono text-[9px] text-white/20", "{field.key} · default: {field.default_value}" }
            if let Some(error) = &error { span { class: "mt-1 block font-mono text-[10px] text-err", "{error}" } }
        }
    }
}
