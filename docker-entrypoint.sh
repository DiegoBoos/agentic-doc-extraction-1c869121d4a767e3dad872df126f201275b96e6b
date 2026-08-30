#!/bin/sh
set -eu

ROLE="${APP_ROLE:-api}"

if [ "$ROLE" = "worker" ]; then
  exec python -m app.worker
fi

exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 5090 \
  --workers "${WEB_CONCURRENCY:-2}" \
  --limit-concurrency "${UVICORN_LIMIT_CONCURRENCY:-16}" \
  --timeout-keep-alive "${UVICORN_TIMEOUT_KEEP_ALIVE:-10}"
