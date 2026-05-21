#!/bin/sh
# Container entrypoint: fetch bigdata from GCS, then start the API server.
# GCS download is skipped for each file already on disk (warm instance reuse).
set -eu

python3 /app/download_bigdata.py

exec uvicorn service:app \
    --host 0.0.0.0 \
    --port "${PORT:-8080}" \
    --workers 1 \
    --timeout-keep-alive 75
