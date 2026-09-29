# Runtime image with Nginx and Mesa Vulkan on Ubuntu 24.04 (matching host glibc 2.39)
FROM ubuntu:24.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    gettext-base \
    ca-certificates \
    libssl3 \
    libvulkan1 \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/sites-enabled/default \
    && mkdir -p /etc/nginx/templates /usr/share/nginx/html /usr/share/nginx/html/testbench

# Copy compiled backend binaries from deploy/dist
COPY deploy/dist/lunar-backend /app/lunar-backend
COPY deploy/dist/lunar-start-backend /app/lunar-start-backend

# Copy AI model weights and manifests
COPY models /app/models

# Copy compiled Lunar Frontend (root /) and WebOS Testbench (/testbench)
COPY deploy/dist/frontend /usr/share/nginx/html
COPY deploy/dist/testbench /usr/share/nginx/html/testbench

# Copy configuration and entrypoint
COPY deploy/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY deploy/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PORT=8080
EXPOSE 8080

CMD ["/app/entrypoint.sh"]
