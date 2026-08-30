import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import health, parsing, patient
from app.core.config import Settings
from app.db.connectiondb import ensure_billing_schema
from app.services.document_parser import (
    AzureDocumentIntelligenceParser,
    DocumentParserRouter,
    GoogleCloudVisionParser,
)
from app.services.openai_extractor import OpenAIExtractorService
from app.services.processing_limiter import ProcessingLimiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()

    parsers = {}

    try:
        parsers["google_vision"] = GoogleCloudVisionParser(
            output_root=settings.parse_output_dir,
            credentials_file=settings.google_vision_credentials_file,
        )
    except Exception as exc:
        print(f"WARNING: Google Cloud Vision could not be initialized: {exc}")

    if settings.azure_document_intelligence_endpoint and settings.azure_document_intelligence_key:
        parsers["azure"] = AzureDocumentIntelligenceParser(
            output_root=settings.parse_output_dir,
            endpoint=settings.azure_document_intelligence_endpoint,
            key=settings.azure_document_intelligence_key,
            model=settings.azure_document_intelligence_model,
        )
    else:
        logger.warning("Azure Document Intelligence is not configured.")

    default_provider = (
        settings.ocr_provider
        if settings.ocr_provider in parsers
        else next(iter(parsers), "google_vision")
    )

    parser = DocumentParserRouter(
        default_provider=default_provider,
        parsers=parsers,
    )

    extractor = OpenAIExtractorService(settings=settings)
    processing_limiter = ProcessingLimiter(settings.processing_max_concurrent_documents)
    ensure_billing_schema()

    logger.info(
        "Processing limiter ready | max_concurrent_documents=%s large_document_threshold=%s",
        settings.processing_max_concurrent_documents,
        settings.processing_large_document_page_threshold,
    )

    app.state.settings = settings
    app.state.parser = parser
    app.state.extractor = extractor
    app.state.processing_limiter = processing_limiter
    yield


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
