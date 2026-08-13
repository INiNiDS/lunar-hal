# Frontend platforms

The managed `frontend` service is one platform-aware service, not separate web and desktop services.

| Platform | Launch shape | Public URL | Sandbox availability |
| --- | --- | --- | --- |
| `web` | `dx serve --platform web --port <port>` | `http://<host>:<port>` | visible when backend is running |
| `desktop` | `dx serve --platform desktop` | none | hidden |
| `android` | `dx serve --platform android` | none | hidden |

## Configuration

Use the Service Settings form or `lunar-start-backend` schema endpoints. The managed values include:

- `LUNAR_FRONTEND_PLATFORM` (`web`, `desktop`, or `android`)
- `LUNAR_FRONTEND_PORT` (web only)
- `LUNAR_DX_BIN`
- `LUNAR_BACKEND_URL`
- service-specific build and extra arguments

Do not add `--platform` or `--port` through free-form arguments: the typed launcher owns those flags and prevents duplicate or conflicting values.

## Build checks

```bash
rustup target add wasm32-unknown-unknown
cargo check -p lunar-frontend --target wasm32-unknown-unknown --no-default-features --features web
cargo check -p lunar-frontend --no-default-features --features desktop
dx build --platform web
```

Android additionally needs the SDK, NDK, `adb`, and an emulator/device. Android emulators reach a host backend at `http://10.0.2.2:25255`; physical devices need an explicit reachable host URL. The WebOS Sandbox intentionally does not appear for desktop or Android because it requires an iframe URL.
