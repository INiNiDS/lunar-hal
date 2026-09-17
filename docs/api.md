# HTTP API overview

The API is local-first. Examples assume `http://127.0.0.1:25255` for `lunar-backend` and `http://127.0.0.1:16181` for `lunar-start-backend`. All mutation inputs are validated server-side; clients should render field errors rather than relying only on browser validation.

## Scene and Gallery API (`lunar-backend`)

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/scenes` | List saved scenes |
| `POST` | `/scenes/create` | Create a scene |
| `GET`, `DELETE` | `/scenes/{id}` | Fetch or delete a scene snapshot |
| `GET` | `/scenes/{id}/events` | SSE stream of ordered scene events |
| `POST` | `/scenes/{id}/stars/generate` | Generate one or more stars into a scene |
| `POST` | `/scenes/{id}/stars` | Add a custom or Gallery-backed star |
| `PATCH`, `DELETE` | `/scenes/{id}/stars/{star_id}` | Modify or remove a scene star |
| `POST` | `/scenes/{id}/clear` | Clear the live scene |
| `GET`, `POST` | `/gallery/stars` | List or create Gallery records |
| `GET`, `PATCH`, `DELETE` | `/gallery/stars/{id}` | Read, edit metadata, or delete a Gallery record |
| `GET` | `/gallery/stars/{id}/texture.png` | Return full texture content |
| `GET` | `/gallery/stars/{id}/thumbnail` | Return a preview asset |
| `GET` | `/version` | Service identity + exact artifact version per registry entry |
| `POST` | `/models/reload` | Controlled model reload (409 + zero state change on invalid bundles) |

The model/pipeline endpoints (`/pinn`, `/gnn`, `/sector/stars`, `/pipeline`, `/siren/*`) remain available for technical tools. Scene SSE uses a scene ID and monotonic event/revision data; clients reconnect by fetching a fresh snapshot before applying later events.

## Managed service API (`lunar-start-backend`)

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Lightweight shell readiness probe |
| `GET` | `/services` | Runtime service status, frontend platform, and public URL |
| `GET` | `/services/meta` | Service metadata for the WebOS desktop |
| `GET` | `/services/{name}/config/schema` | Form schema and defaults |
| `GET`, `PUT` | `/services/{name}/config` | Read or save a stopped service's configuration |
| `POST` | `/services/{name}/validate` | Trusted filesystem/port/executable validation |
| `POST` | `/services/{name}/start` | Spawn from the saved validated configuration |
| `POST` | `/services/{name}/restart` | Validate and restart with saved configuration |
| `POST` | `/services/{name}/stop` | Stop the managed service |
| `GET` | `/services/{name}/logs`, `/services/{name}/stats` | Log tail and counters |
| `GET` | `/logs` | Aggregate SSE log stream |

Field validation errors are returned as a structured `field_errors` object. Requests that conflict with runtime state or ports use a conflict response; failed spawn operations are server errors. Legacy GET start/stop/restart routes exist only as temporary compatibility wrappers and are not the preferred client contract.
