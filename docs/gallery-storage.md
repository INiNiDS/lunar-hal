# Gallery storage

`lunar-backend` stores Gallery records in the platform data directory by default. Set `LUNAR_GALLERY_DIR` to override the root, for example for an isolated test run.

Each record is stored under a stable ID directory:

```text
gallery/
  <gallery-id>/
    metadata.json
    texture.png
    thumb.webp   # or a PNG thumbnail when applicable
```

Writes use a temporary file followed by rename so a partially written record is never published as valid. Gallery CRUD is durable across backend restarts. A broken single record must not prevent neighboring records from loading.

Scene-to-Gallery and Gallery-to-scene transfers have copy semantics. A repeated request must use a request ID/idempotency key so retrying a drop does not create duplicates. Old world/game data is not migrated or read by this storage layer.
