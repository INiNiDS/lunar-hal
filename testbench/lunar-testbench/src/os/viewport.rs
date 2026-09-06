use dioxus::prelude::*;

use crate::os::use_os_state;

#[cfg(target_arch = "wasm32")]
use std::{cell::RefCell, rc::Rc};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, closure::Closure};

/// Retains browser listeners for the room lifetime and removes them when the
/// room unmounts. Resize events are debounced; global release/focus events make
/// sure a lost mouseup can never leave the drag overlay blocking the desktop.
#[cfg(target_arch = "wasm32")]
struct ViewportListeners {
    window: web_sys::Window,
    visual_viewport: Option<web_sys::VisualViewport>,
    window_resize: Closure<dyn FnMut(web_sys::Event)>,
    orientation_change: Closure<dyn FnMut(web_sys::Event)>,
    viewport_resize: Option<Closure<dyn FnMut(web_sys::Event)>>,
    global_mouse_up: Closure<dyn FnMut(web_sys::Event)>,
    window_blur: Closure<dyn FnMut(web_sys::Event)>,
    pointer_cancel: Closure<dyn FnMut(web_sys::Event)>,
}

#[cfg(target_arch = "wasm32")]
impl Drop for ViewportListeners {
    fn drop(&mut self) {
        let _ = self.window.remove_event_listener_with_callback(
            "resize",
            self.window_resize.as_ref().unchecked_ref(),
        );
        let _ = self.window.remove_event_listener_with_callback(
            "orientationchange",
            self.orientation_change.as_ref().unchecked_ref(),
        );
        let _ = self.window.remove_event_listener_with_callback(
            "mouseup",
            self.global_mouse_up.as_ref().unchecked_ref(),
        );
        let _ = self
            .window
            .remove_event_listener_with_callback("blur", self.window_blur.as_ref().unchecked_ref());
        let _ = self.window.remove_event_listener_with_callback(
            "pointercancel",
            self.pointer_cancel.as_ref().unchecked_ref(),
        );
        if let (Some(viewport), Some(listener)) = (&self.visual_viewport, &self.viewport_resize) {
            let _ = viewport
                .remove_event_listener_with_callback("resize", listener.as_ref().unchecked_ref());
        }
    }
}

/// Raw browser closures run outside the Dioxus runtime, so they must only
/// touch signals here — calling `spawn` from them panics in
/// `Runtime::current_scope_id`. The debounced reflow task is spawned by the
/// effect watching `generation`, which always runs inside the component scope.
#[cfg(target_arch = "wasm32")]
fn resize_listener(mut generation: Signal<u64>) -> Closure<dyn FnMut(web_sys::Event)> {
    Closure::wrap(Box::new(move |_event: web_sys::Event| {
        generation.with_mut(|g| *g += 1);
    }) as Box<dyn FnMut(web_sys::Event)>)
}

/// Debounces the raw resize events into a single reflow 75ms after the last one.
/// Must run inside a component scope so `spawn` has a current scope.
#[cfg(target_arch = "wasm32")]
fn use_debounced_reflow(
    mut os: crate::os::OsState,
    generation: Signal<u64>,
    mut last_viewport: Signal<(f64, f64)>,
) {
    use_effect(move || {
        let ticket = generation();
        if ticket == 0 {
            return;
        }
        spawn(async move {
            gloo_timers::future::TimeoutFuture::new(75).await;
            if *generation.peek() != ticket {
                return;
            }
            let size = crate::os::viewport_size();
            let previous = *last_viewport.peek();
            if (previous.0 - size.0).abs() < 1.0 && (previous.1 - size.1).abs() < 1.0 {
                return;
            }
            last_viewport.set(size);
            os.reflow_windows(size.0, size.1);
        });
    });
}

#[cfg(target_arch = "wasm32")]
fn cancel_drag_listener(mut os: crate::os::OsState) -> Closure<dyn FnMut(web_sys::Event)> {
    Closure::wrap(Box::new(move |_event: web_sys::Event| {
        os.cancel_drag();
    }) as Box<dyn FnMut(web_sys::Event)>)
}

#[cfg(target_arch = "wasm32")]
fn install_viewport_listeners(
    os: crate::os::OsState,
    generation: Signal<u64>,
) -> Rc<RefCell<Option<ViewportListeners>>> {
    let retained = Rc::new(RefCell::new(None));
    let Some(window) = web_sys::window() else {
        return retained;
    };

    let window_resize = resize_listener(generation);
    let orientation_change = resize_listener(generation);
    let global_mouse_up = cancel_drag_listener(os);
    let window_blur = cancel_drag_listener(os);
    let pointer_cancel = cancel_drag_listener(os);

    let _ =
        window.add_event_listener_with_callback("resize", window_resize.as_ref().unchecked_ref());
    let _ = window.add_event_listener_with_callback(
        "orientationchange",
        orientation_change.as_ref().unchecked_ref(),
    );
    let _ = window
        .add_event_listener_with_callback("mouseup", global_mouse_up.as_ref().unchecked_ref());
    let _ = window.add_event_listener_with_callback("blur", window_blur.as_ref().unchecked_ref());
    let _ = window
        .add_event_listener_with_callback("pointercancel", pointer_cancel.as_ref().unchecked_ref());

    let visual_viewport = window.visual_viewport();
    let viewport_resize = visual_viewport
        .as_ref()
        .map(|_| resize_listener(generation));
    if let (Some(viewport), Some(listener)) = (&visual_viewport, &viewport_resize) {
        let _ =
            viewport.add_event_listener_with_callback("resize", listener.as_ref().unchecked_ref());
    }

    *retained.borrow_mut() = Some(ViewportListeners {
        window,
        visual_viewport,
        window_resize,
        orientation_change,
        viewport_resize,
        global_mouse_up,
        window_blur,
        pointer_cancel,
    });
    retained
}

/// Reflow floating windows when browser or visual-viewport dimensions change.
/// The initial effect also clamps any restored geometry before the room becomes
/// interactive. On non-WASM checks there is no browser event target, so only the
/// initial reflow is meaningful.
pub fn use_viewport_resize() {
    let os = use_os_state();
    let mut initial_os = os;
    use_effect(move || {
        let (width, height) = crate::os::viewport_size();
        initial_os.reflow_windows(width, height);
    });

    #[cfg(target_arch = "wasm32")]
    {
        let generation = use_signal(|| 0_u64);
        let last_viewport = use_signal(crate::os::viewport_size);
        use_debounced_reflow(os, generation, last_viewport);
        let _listeners = use_hook(move || install_viewport_listeners(os, generation));
        let _ = _listeners;
    }
}
