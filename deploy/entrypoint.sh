#!/bin/sh
set -e

export PORT=${PORT:-8080}
export LUNAR_ALLOW_UNAPPROVED_MODELS=1
export LUNAR_MODELS_DIR=${LUNAR_MODELS_DIR:-/app/models}
export LUNAR_BACKEND_PORT=25255
export LUNAR_BACKEND_HOST=127.0.0.1
export LUNAR_START_PORT=16181
export LUNAR_START_HOST=127.0.0.1

echo "[entrypoint] Starting lunar-start-backend on :16181..."
/app/lunar-start-backend &

echo "[entrypoint] Starting lunar-backend on :25255..."
/app/lunar-backend &

echo "[entrypoint] Configuring Nginx port ${PORT}..."
envsubst '$PORT' < /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf

echo "[entrypoint] Starting Nginx on 0.0.0.0:${PORT}..."
exec nginx -g 'daemon off;'
