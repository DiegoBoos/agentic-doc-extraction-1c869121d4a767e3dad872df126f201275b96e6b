# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

FastAPI backend that parses medical-authorization documents (PDF/image) via OCR and extracts
structured fields with an LLM. Spanish-language domain (Colombian medical authorizations —
"autorizaciones médicas"); user-facing error messages and the LLM system prompts are in Spanish.

## Commands

```bash
uv sync                                # install deps (Python 3.12+, managed by uv)
uv run uvicorn app.main:app --reload   # run the API locally
APP_ROLE=worker uv run python -m app.worker   # run the queue worker locally

uv run python -m pytest                        # run all tests
uv run python -m pytest tests/test_api.py -k test_health   # run a single test
uv run python -m ruff check .                  # lint
uv run python -m ruff format --check .         # format check
uv run python -m ruff format .                 # auto-format
```

There is no `DATABASE_URL` required for tests/local dev to boot — DB and Redis access degrade
gracefully to no-ops when unconfigured (see Architecture below).

## Architecture

### Request flow

Two POST endpoints ([app/api/routes/parsing.py](app/api/routes/parsing.py) and
[app/api/routes/patient.py](app/api/routes/patient.py)) share the same shape:

1. Validate + save the upload ([app/services/file_ingest.py](app/services/file_ingest.py)) —
   extension/content-type allowlist, size cap, SHA-256 content hash for dedup.
2. Page-count gate: reject PDFs over `max_upload_pages` with a structured 400 JSON body
   *before* touching OCR/LLM (cheap rejection).
3. Hand off to [app/services/processing_pipeline.py](app/services/processing_pipeline.py),
   either directly or via the job queue (see below).
4. OCR via [app/services/document_parser.py](app/services/document_parser.py)
   (`DocumentParserRouter` dispatches to Azure Document Intelligence or Google Cloud Vision).
5. Extraction via [app/services/openai_extractor.py](app/services/openai_extractor.py) using
   OpenAI structured outputs (`client.responses.parse`), with automatic chunking for large
   documents ([app/services/authorization_chunking.py](app/services/authorization_chunking.py)).
6. Billing metadata persisted to Postgres with a retry loop
   (`persist_billing_metadata_with_retry`); best-effort when no `DATABASE_URL` is set, strict
   (raises) when it is.
7. Response normalized/deduped (`normalize_authorizations`) and returned as the same JSON either
   way — sync or queued.

### Sync HTTP contract over an async job queue

The client always calls `POST /api/v1/parse` (or `/extract-patient`) and gets the final JSON in
that same request/response — even when `JOB_QUEUE_MODE=redis` is enabled. Internally:

- API saves the file, creates a `processing_jobs` row, pushes `job_id` onto a Redis list
  ([app/services/redis_queue.py](app/services/redis_queue.py)), then polls Postgres
  (`JobCoordinator.wait_for_result`) until the job is `succeeded`/`failed` or times out
  (`JOB_QUEUE_WAIT_TIMEOUT_SECONDS` → HTTP 504).
- A separate worker process (`APP_ROLE=worker`, [app/worker.py](app/worker.py)) BLPOPs job IDs
  and runs the same pipeline functions used by the direct/sync path
  ([app/services/job_orchestrator.py](app/services/job_orchestrator.py)).
- `JOB_QUEUE_MODE=direct` (default) skips Redis/Postgres entirely and runs the pipeline inline
  in the request handler — this is the mode used in tests and simple deployments.
- The same Docker image serves both roles; `APP_ROLE` (`api` | `worker`) picks the entrypoint
  in [docker-entrypoint.sh](docker-entrypoint.sh). In Dokploy this is deployed as two services
  from the same image/branch.

### Runtime construction

[app/services/runtime.py](app/services/runtime.py) builds a `RuntimeServices` bundle (parsers,
extractor, processing limiter, optional job queue) once at startup
(`lifespan` in [app/main.py](app/main.py)) and stores it on `app.state`. Route handlers pull
individual pieces back out via FastAPI `Depends` in
[app/dependencies/services.py](app/dependencies/services.py). Both parser providers are
constructed independently and failures are non-fatal (a missing/misconfigured provider just
logs a warning and is left out of the router's `parsers` dict) — `parse_first_page` (used for
patient extraction) is hardcoded to the `"azure"` provider regardless of `OCR_PROVIDER`.

### Config

All settings live in a single `pydantic-settings` class:
[app/core/config.py](app/core/config.py). Most fields accept either a bare env var
(`REDIS_HOST`) or a `DOC_EXTRACTION_`-prefixed one via `AliasChoices` — check this file for the
authoritative list of env vars rather than relying on the README, which only documents a subset.

### Database layer

[app/db/connectiondb.py](app/db/connectiondb.py) uses a lazily-initialized
`ThreadedConnectionPool` and idempotent `CREATE TABLE IF NOT EXISTS` / `ADD COLUMN IF NOT EXISTS`
migrations run on first use (`ensure_billing_schema`, called from both API and worker startup).
There is no separate migrations tool/directory — schema changes are made by editing
`ensure_billing_schema` directly. All DB calls are sync psycopg2 wrapped in
`asyncio.to_thread` at call sites.

### Extraction schemas & normalization

- [app/schemas/authorization.py](app/schemas/authorization.py) defines the structured-output
  contract for the LLM (`AuthorizationResponse` / `AuthorizationExtraction`) with Pydantic
  validators that normalize dates, `vigencia` text, and enforce `numero_autorizacion` as exactly
  14 digits (non-conforming authorizations are silently dropped, not errored).
  `ServiciosAutorizados` auto-corrects `ubicacion_paciente`/`grupo_servicio` if the LLM swaps
  them.
- [app/schemas/patient.py](app/schemas/patient.py) is the parallel contract for the
  `/extract-patient` endpoint; [app/services/fhir_mapper.py](app/services/fhir_mapper.py) maps
  that patient data to a FHIR-shaped payload included in the response under `fhir`.

### Test conventions

Tests use `TestClient(app)` and monkeypatch service instances directly on `client.app.state`
(e.g. `client.app.state.parser.parse_document = lambda **_: ...`,
`client.app.state.extractor = _MockExtractor()`) rather than FastAPI dependency overrides — the
default `JOB_QUEUE_MODE=direct` means no Redis/Postgres is required to run the suite.
