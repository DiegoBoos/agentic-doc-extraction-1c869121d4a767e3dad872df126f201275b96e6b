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
- y reutiliza conexiones a PostgreSQL mediante pool.

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

## Tests y calidad

```bash
uv run python -m pytest
uv run python -m ruff check .
uv run python -m ruff format --check .
```
# agentic-doc-extraction
