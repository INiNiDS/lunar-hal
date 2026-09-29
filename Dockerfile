# Stage 1: Build Rust backend binaries and WebAssembly frontend
FROM rust:1.85-slim-bookworm AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    curl \
    ca-certificates \
    tar \
    && rm -rf /var/lib/apt/lists/*

RUN rustup target add wasm32-unknown-unknown

# Download prebuilt dx CLI for ultra-fast compilation
RUN curl -sL https://github.com/DioxusLabs/dioxus/releases/download/v0.7.10/dx-x86_64-unknown-linux-gnu.tar.gz | tar -xz -C /usr/local/bin && chmod +x /usr/local/bin/dx

COPY . .

# Build release backend and service manager binaries
RUN cargo build --release --bin lunar-backend --bin lunar-start-backend

# Build release WASM frontend
WORKDIR /app/testbench/lunar-testbench
RUN dx build --platform web --release

# Stage 2: Minimal runtime image with Nginx
FROM nginx:alpine

WORKDIR /app

RUN apk add --no-cache gettext ca-certificates libgcc libssl3

# Copy compiled backend binaries
COPY --from=builder /app/target/release/lunar-backend /app/lunar-backend
COPY --from=builder /app/target/release/lunar-start-backend /app/lunar-start-backend

# Copy AI model weights and manifests
COPY --from=builder /app/models /app/models

# Copy compiled WebOS frontend static files
COPY --from=builder /app/target/dx/lunar-testbench/release/web/public /usr/share/nginx/html

# Copy configuration and entrypoint
COPY deploy/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY deploy/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PORT=8080
EXPOSE 8080

CMD ["/app/entrypoint.sh"]
