from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import HTTPException, status

from app.core.config import Settings
from app.db.connectiondb import (
    complete_processing_job,
    create_processing_job,
    fail_processing_job,
    get_processing_job,
    mark_processing_job_running,
)
from app.services.file_ingest import SavedUpload
from app.services.processing_pipeline import (
    JOB_TYPE_PARSE,
    JOB_TYPE_PATIENT,
    build_http_exception,
    cleanup_processing_artifacts,
    map_processing_exception,
    run_parse_pipeline,
    run_patient_pipeline,
)
from app.services.redis_queue import RedisJobQueue
from app.services.runtime import RuntimeServices

logger = logging.getLogger(__name__)


class JobCoordinator:
    def __init__(self, *, settings: Settings, queue: RedisJobQueue) -> None:
        self.settings = settings
        self.queue = queue

    async def submit(self, *, job_type: str, saved: SavedUpload) -> str:
        started_at = time.perf_counter()
        metadata = {
            "filename": saved.filename,
            "content_type": saved.content_type,
        }
        logger.info(
            "Creating processing job | job_id=%s job_type=%s filename=%s size_bytes=%s",
            saved.id,
            job_type,
            saved.filename,
            saved.size_bytes,
        )
        await asyncio.to_thread(
            create_processing_job,
            job_id=saved.id,
            job_type=job_type,
            document_id=saved.id,
            file_hash=saved.file_hash,
            filename=saved.filename,
            content_type=saved.content_type,
            size_bytes=saved.size_bytes,
            stored_path=saved.stored_path,
            created_at=saved.created_at,
            metadata=metadata,
        )
        await self.queue.enqueue(saved.id)
        logger.info(
            "Queued processing job | job_id=%s job_type=%s elapsed_ms=%.2f",
            saved.id,
            job_type,
            (time.perf_counter() - started_at) * 1000,
        )
        return saved.id

    async def wait_for_result(self, job_id: str) -> dict[str, Any]:
        timeout_seconds = max(1, self.settings.job_queue_wait_timeout_seconds)
        poll_interval_seconds = max(0.1, self.settings.job_queue_poll_interval_ms / 1000)
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        deadline = started_at + timeout_seconds
        next_progress_log = started_at + 15
        poll_count = 0
        last_status: str | None = None

        while True:
            poll_count += 1
            job = await asyncio.to_thread(get_processing_job, job_id)
            if job is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Processing job not found: {job_id}",
                )

            status_value = job.get("status")
            elapsed_seconds = loop.time() - started_at
            if status_value != last_status:
                logger.info(
                    "Processing job status changed | job_id=%s status=%s elapsed_s=%.2f",
                    job_id,
                    status_value,
                    elapsed_seconds,
                )
                last_status = status_value
            elif loop.time() >= next_progress_log:
                logger.info(
                    "Still waiting for processing job | job_id=%s status=%s elapsed_s=%.2f polls=%s",
                    job_id,
                    status_value,
                    elapsed_seconds,
                    poll_count,
                )
                next_progress_log = loop.time() + 15

            if status_value == "succeeded":
                payload = job.get("result_payload")
                if isinstance(payload, dict):
                    logger.info(
                        "Processing job succeeded | job_id=%s elapsed_s=%.2f polls=%s",
                        job_id,
                        elapsed_seconds,
                        poll_count,
                    )
                    return payload
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Processing job completed without payload.",
                )

            if status_value == "failed":
                logger.warning(
                    "Processing job failed while waiting | job_id=%s error_type=%s elapsed_s=%.2f",
                    job_id,
                    job.get("error_type"),
                    elapsed_seconds,
                )
                raise self._build_failed_job_exception(job)

            if loop.time() >= deadline:
                logger.warning(
                    "Processing job wait timed out | job_id=%s elapsed_s=%.2f polls=%s",
                    job_id,
                    elapsed_seconds,
                    poll_count,
                )
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=(
                        "El procesamiento sigue en curso y superó el tiempo de espera "
                        f"sincrónico. job_id={job_id}"
                    ),
                )

            await asyncio.sleep(poll_interval_seconds)

    @staticmethod
    def _build_failed_job_exception(job: dict[str, Any]) -> HTTPException:
        error_type = job.get("error_type")
        error_message = job.get("error_message") or "Processing job failed."
        mapping = {
            "llm_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
            "validation_error": status.HTTP_400_BAD_REQUEST,
            "document_too_large": status.HTTP_400_BAD_REQUEST,
            "internal_error": status.HTTP_500_INTERNAL_SERVER_ERROR,
        }
        return HTTPException(
            status_code=mapping.get(error_type, status.HTTP_500_INTERNAL_SERVER_ERROR),
            detail=error_message,
        )


def saved_upload_from_job(job: dict[str, Any]) -> SavedUpload:
    return SavedUpload(
        id=job["document_id"],
        filename=job["filename"],
        content_type=job.get("content_type"),
        size_bytes=job["size_bytes"],
        stored_path=job["stored_path"],
        created_at=job["created_at"],
        file_hash=job.get("file_hash") or "",
    )


async def process_job(runtime: RuntimeServices, job_id: str) -> None:
    started_at = time.perf_counter()
    job = await asyncio.to_thread(get_processing_job, job_id)
    if job is None:
        logger.warning("Processing job %s not found", job_id)
        return

    saved = saved_upload_from_job(job)
    logger.info(
        "Worker picked up job | job_id=%s job_type=%s filename=%s size_bytes=%s",
        job_id,
        job.get("job_type"),
        saved.filename,
        saved.size_bytes,
    )
    await asyncio.to_thread(mark_processing_job_running, job_id)
    logger.info("Marked job as running | job_id=%s", job_id)

    try:
        if job["job_type"] == JOB_TYPE_PARSE:
            payload = await run_parse_pipeline(
                saved=saved,
                parser=runtime.parser,
                settings=runtime.settings,
                extractor=runtime.extractor,
                limiter=runtime.processing_limiter,
            )
        elif job["job_type"] == JOB_TYPE_PATIENT:
            payload = await run_patient_pipeline(
                saved=saved,
                parser=runtime.parser,
                settings=runtime.settings,
                extractor=runtime.extractor,
                limiter=runtime.processing_limiter,
            )
        else:
            raise ValueError(f"Unsupported job type: {job['job_type']}")

        await asyncio.to_thread(complete_processing_job, job_id, payload)
        logger.info(
            "Worker completed job | job_id=%s job_type=%s elapsed_ms=%.2f",
            job_id,
            job.get("job_type"),
            (time.perf_counter() - started_at) * 1000,
        )
    except Exception as exc:
        _, error_type, error_message = map_processing_exception(exc)
        await asyncio.to_thread(
            fail_processing_job,
            job_id,
            error_type=error_type,
            error_message=error_message,
        )
        logger.exception(
            "Processing job failed | job_id=%s elapsed_ms=%.2f",
            job_id,
            (time.perf_counter() - started_at) * 1000,
        )
    finally:
        await asyncio.to_thread(cleanup_processing_artifacts, saved, runtime.settings)
        logger.info("Cleaned up job artifacts | job_id=%s", job_id)


async def run_direct_parse(
    *,
    saved: SavedUpload,
    runtime: RuntimeServices,
) -> dict[str, Any]:
    try:
        return await run_parse_pipeline(
            saved=saved,
            parser=runtime.parser,
            settings=runtime.settings,
            extractor=runtime.extractor,
            limiter=runtime.processing_limiter,
        )
    except Exception as exc:
        raise build_http_exception(exc) from exc
    finally:
        await asyncio.to_thread(cleanup_processing_artifacts, saved, runtime.settings)


async def run_direct_patient(
    *,
    saved: SavedUpload,
    runtime: RuntimeServices,
) -> dict[str, Any]:
    try:
        return await run_patient_pipeline(
            saved=saved,
            parser=runtime.parser,
            settings=runtime.settings,
            extractor=runtime.extractor,
            limiter=runtime.processing_limiter,
        )
    except Exception as exc:
        raise build_http_exception(exc) from exc
    finally:
        await asyncio.to_thread(cleanup_processing_artifacts, saved, runtime.settings)
