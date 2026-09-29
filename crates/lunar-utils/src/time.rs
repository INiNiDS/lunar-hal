pub fn current_time_ms() -> u64 {
    #[cfg(all(target_family = "wasm", not(target_os = "wasi")))]
    {
        js_sys::Date::now() as u64
    }
    #[cfg(not(all(target_family = "wasm", not(target_os = "wasi"))))]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

pub fn current_time_nanos() -> u128 {
    #[cfg(all(target_family = "wasm", not(target_os = "wasi")))]
    {
        (js_sys::Date::now() * 1_000_000.0) as u128
    }
    #[cfg(not(all(target_family = "wasm", not(target_os = "wasi"))))]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    }
}
