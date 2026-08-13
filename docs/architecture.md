# Architecture

Lunar-HAL is a technical stellar generation and visualization platform. It has no gameplay runtime, combat state, or compatibility layer for the previous world/game API.

## Data ownership

- **`lunar-backend`** owns live scenes, generated stars, SIREN artifacts, Gallery records, and the scene event stream. It is the only source of truth for mutations.
- **`lunar-stellar-core`** is a framework-agnostic client read model for camera state, selection, sector caching, validation, snapshots, and scene events.
- **`lunar-frontend`** renders a scene and consumes snapshot/SSE data. Its embedded mode is a renderer, not an administrative command receiver.
- **`lunar-testbench`** is the WebOS host and administrative overlay. The Sandbox is an iframe of the real web frontend; overlay mutations call the backend directly.
- **`lunar-start` and `lunar-start-backend`** validate service-specific configuration, build platform-aware frontend commands, and publish runtime status including frontend platform and public URL.

## Scene flow

1. WebOS opens a Sandbox window only when backend is running and the managed frontend reports `platform: web` with a public URL.
2. Sandbox embeds `<public_url>/editor?embedded=sandbox&scene_id=<id>`.
3. The overlay submits create, edit, delete, import, or clear requests to scene/gallery endpoints.
4. Backend persists the mutation and publishes a scene event.
5. The iframe frontend applies that event to its local read model. The parent does not send mutation commands through `postMessage`.

## Window lifecycle

A WebOS `WindowState` remains mounted until the user closes it. `Minimized` hides the DOM subtree, and `Blocked` draws a dependency overlay over the mounted content. Only `Close` removes the window and its ephemeral component state. See [WebOS lifecycle](webos-lifecycle.md).

## Responsive boundary

Every app window body is a named CSS container (`app-window`). App pages use container queries rather than browser-width media queries. A manifest entry declares each app's preferred/minimum size and `Scroll` or `Fill` body mode. This keeps iframe/canvas hosts full-size while document pages scroll internally.
