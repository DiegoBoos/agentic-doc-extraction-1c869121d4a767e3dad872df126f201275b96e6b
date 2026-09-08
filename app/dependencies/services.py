from fastapi import Header, HTTPException, Request

from app.core.config import Settings
from app.services.document_parser import DocumentParserRouter
from app.services.job_orchestrator import JobCoordinator
from app.services.openai_extractor import OpenAIExtractorService
from app.services.processing_limiter import ProcessingLimiter
from app.services.redis_queue import RedisJobQueue


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    settings: Settings = request.app.state.settings
    if settings.api_key is None:
        return
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def get_parser(request: Request) -> DocumentParserRouter:
    return request.app.state.parser


def get_extractor(request: Request) -> OpenAIExtractorService:
    extractor = request.app.state.extractor
    if extractor is None:
        raise HTTPException(status_code=500, detail="OpenAI extractor is not configured")
    return extractor


def get_processing_limiter(request: Request) -> ProcessingLimiter:
    limiter = getattr(request.app.state, "processing_limiter", None)
    if limiter is None:
        raise HTTPException(status_code=500, detail="Processing limiter is not configured")
    return limiter


def get_job_coordinator(request: Request) -> JobCoordinator | None:
    settings: Settings = request.app.state.settings
    queue: RedisJobQueue | None = getattr(request.app.state, "job_queue", None)
    if not settings.queue_enabled or queue is None:
        return None
    return JobCoordinator(settings=settings, queue=queue)
