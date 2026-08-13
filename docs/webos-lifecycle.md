# WebOS window lifecycle

A WebOS window has three visible runtime states:

- **Visible** — the window is interactive and all required services are available.
- **Minimized** — the window subtree stays mounted but is hidden with `display: none`, `visibility: hidden`, `aria-hidden`, and no pointer events.
- **Blocked** — the app subtree stays mounted while a dependency overlay prevents interaction and names missing services.

`Close` is the only operation that removes `WindowState` and unmounts the app subtree. Reopening after Close creates a fresh session; restoring after Minimize or a temporary Blocked state preserves form values, selected tabs, results, filters, and iframe navigation.

## App manifest

`src/os/manifest.rs` is the source for app dependencies, preferred/minimum window size, category, and body mode:

- `Scroll` apps own an internal vertical scrollbar.
- `Fill` apps use a clipped, relative client area for iframe/canvas content.

Floating windows are clamped on viewport/visual-viewport resize. First-open windows are centered, later windows cascade only while visible, and maximized/snap rects reserve dock safe area.

New pages must not create independent global service polling. Read lifecycle context when pausing expensive page-local refreshes, but leave shell-level `/health`, `/services`, and log-stream handling in `use_os_runtime`.
