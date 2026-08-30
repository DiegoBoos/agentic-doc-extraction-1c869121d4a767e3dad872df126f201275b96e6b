from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.config import Settings
from app.services.document_parser import (
    AzureDocumentIntelligenceParser,
    DocumentParserRouter,
    GoogleCloudVisionParser,
)
from app.services.openai_extractor import OpenAIExtractorService
from app.services.processing_limiter import ProcessingLimiter
from app.services.redis_queue import RedisJobQueue

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeServices:
    settings: Settings
    parser: DocumentParserRouter
    extractor: OpenAIExtractorService
    processing_limiter: ProcessingLimiter
    job_queue: RedisJobQueue | None


async def build_runtime(settings: Settings | None = None) -> RuntimeServices:
    settings = settings or Settings()

    parsers: dict[str, object] = {}

    try:
        parsers["google_vision"] = GoogleCloudVisionParser(
            output_root=settings.parse_output_dir,
            credentials_file=settings.google_vision_credentials_file,
        )
    except Exception as exc:
        logger.warning("Google Cloud Vision could not be initialized: %s", exc)

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

    job_queue = None
    if settings.queue_enabled:
        job_queue = RedisJobQueue(
            redis_url=settings.resolved_redis_url,
            queue_name=settings.job_queue_name,
        )
        await job_queue.ping()
        logger.info("Redis job queue ready | queue=%s", settings.job_queue_name)

    return RuntimeServices(
        settings=settings,
        parser=parser,
        extractor=extractor,
        processing_limiter=processing_limiter,
        job_queue=job_queue,
    )
