import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import health, jobs, parsing, patient, payment_file
from app.db.connectiondb import ensure_billing_schema
from app.logging_setup import configure_logging
from app.services.runtime import build_runtime

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    runtime = await build_runtime()
    ensure_billing_schema()

    logger.info(
        "Runtime ready | queue_enabled=%s max_concurrent_documents=%s large_document_threshold=%s",
        runtime.settings.queue_enabled,
        runtime.settings.processing_max_concurrent_documents,
        runtime.settings.processing_large_document_page_threshold,
    )

    app.state.settings = runtime.settings
    app.state.parser = runtime.parser
    app.state.extractor = runtime.extractor
    app.state.processing_limiter = runtime.processing_limiter
    app.state.job_queue = runtime.job_queue
    yield

    if runtime.job_queue is not None:
        await runtime.job_queue.close()


app = FastAPI(
    title="agentic-doc-extraction",
    version="0.1.0",
    description="FastAPI backend for parsing and extracting structured data from documents.",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled exception | %s %s",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


app.include_router(health.router)
app.include_router(parsing.router, prefix="/api/v1")
app.include_router(patient.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(payment_file.router, prefix="/api/v1")
