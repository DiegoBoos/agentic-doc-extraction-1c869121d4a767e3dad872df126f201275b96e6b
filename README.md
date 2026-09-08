# agentic-doc-extraction

Backend en **FastAPI** para parsear documentos y extraer datos estructurados de autorizaciones médicas.

Actualmente soporta dos proveedores de OCR/parseo:

- **Azure AI Document Intelligence** (`prebuilt-read`)

## Endpoints activos

- `GET /health`
- `POST /api/v1/parse`

## Ejecutar

```bash
uv sync
uv run uvicorn app.main:app --reload
```

## Variables de entorno

### Concurrencia y robustez

```bash
export PROCESSING_MAX_CONCURRENT_DOCUMENTS=2
export PROCESSING_LARGE_DOCUMENT_PAGE_THRESHOLD=50
export OPENAI_MAX_INPUT_TOKENS=45000
export OPENAI_CHUNK_TARGET_TOKENS=24000
export OPENAI_CHUNK_MAX_PAGES=12
export OPENAI_CHUNK_OVERLAP_PAGES=1
export DATABASE_POOL_MIN=1
export DATABASE_POOL_MAX=8
export WEB_CONCURRENCY=2
export UVICORN_LIMIT_CONCURRENCY=16
```

### Jobs / cola interna

```bash
# modo clásico (sin cola)
export JOB_QUEUE_MODE=direct

# modo escalable con Redis + worker
export JOB_QUEUE_MODE=redis
export REDIS_HOST=127.0.0.1
export REDIS_PORT=6379
export REDIS_PASSWORD=""
export REDIS_DB=0
export JOB_QUEUE_NAME=agentic-doc-extraction:jobs
export JOB_QUEUE_WAIT_TIMEOUT_SECONDS=900
export JOB_QUEUE_POLL_INTERVAL_MS=500
```

### Azure Document Intelligence

```bash
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://<tu-resource>.cognitiveservices.azure.com/"
export AZURE_DOCUMENT_INTELLIGENCE_KEY="<tu-key>"
# opcional
export DOC_EXTRACTION_AZURE_DOCUMENT_INTELLIGENCE_MODEL=prebuilt-read
```

## Qué hace cada endpoint

### `POST /api/v1/parse`

Recibe un archivo, lo guarda localmente, lo parsea con el proveedor seleccionado y luego extrae campos útiles de autorizaciones médica.

Para soportar documentos grandes y mayor concurrencia, la extracción ahora:

- limita la cantidad de documentos pesados que se procesan en paralelo,
- mueve el OCR bloqueante fuera del event loop principal,
- fragmenta documentos grandes antes de enviarlos al LLM,
- reutiliza conexiones a PostgreSQL mediante pool,
- y puede correr en modo **job/cola** sin romper el contrato síncrono del endpoint.

### Modo job/cola sin romper el contrato HTTP

El cliente sigue llamando a:

```http
POST /api/v1/parse
```

Y sigue recibiendo el **JSON final en la misma request**.

La diferencia es interna:

1. la API guarda el archivo,
2. crea un `processing_job` en PostgreSQL,
3. encola el `job_id` en Redis,
4. un worker separado procesa OCR + LLM,
5. la API espera el resultado del job y responde el mismo JSON final.

Esto permite escalar workers por separado sin bloquear el proceso HTTP con OCR/LLM pesado.

Campos extraídos:

- `authorization_numbers`
- `primary_authorization_number`
- `solicitud_numbers`
- `patient_name`
- `patient_document_id`
- `eps`
- `ips`
- `cups_codes`
- `service_description`
- `observacion`
- `vigencia`

## Archivos generados

- uploads: `data/uploads/`
- parse outputs: `data/parsed/<document_id>/`
- markdown: `data/parsed/<document_id>/<archivo>.md`

## Worker

La misma imagen sirve para API o worker.

### API

```bash
export APP_ROLE=api
uv run uvicorn app.main:app --reload
```

### Worker

```bash
export APP_ROLE=worker
uv run python -m app.worker
```

En Dokploy puedes desplegar dos servicios desde la misma rama/imagen:
- uno con `APP_ROLE=api`
- otro con `APP_ROLE=worker`

## Tests y calidad

```bash
uv run python -m pytest
uv run python -m ruff check .
uv run python -m ruff format --check .
```
# agentic-doc-extraction
