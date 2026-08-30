from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.config import Settings
from app.dependencies.services import (
    get_extractor,
    get_job_coordinator,
    get_parser,
    get_processing_limiter,
    get_settings,
    require_api_key,
)
from app.services.document_parser import DocumentParserRouter
from app.services.file_ingest import FileTooLargeError, save_upload
from app.services.job_orchestrator import JobCoordinator, run_direct_patient
from app.services.openai_extractor import OpenAIExtractorService
from app.services.processing_limiter import ProcessingLimiter
from app.services.processing_pipeline import (
    JOB_TYPE_PATIENT,
    _count_pdf_pages,
    cleanup_processing_artifacts,
)
from app.services.runtime import RuntimeServices

router = APIRouter(prefix="/extract-patient", tags=["patient"])

ParserDep = Annotated[DocumentParserRouter, Depends(get_parser)]
ExtractorDep = Annotated[OpenAIExtractorService, Depends(get_extractor)]
LimiterDep = Annotated[ProcessingLimiter, Depends(get_processing_limiter)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
UploadFileDep = Annotated[UploadFile, File(...)]
JobCoordinatorDep = Annotated[JobCoordinator | None, Depends(get_job_coordinator)]


@router.post(
    "",
    status_code=status.HTTP_200_OK,
)
async def extract_patient(
    file: UploadFileDep,
    parser: ParserDep,
    settings: SettingsDep,
    extractor: ExtractorDep,
    limiter: LimiterDep,
    coordinator: JobCoordinatorDep,
    _auth: None = Depends(require_api_key),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required")

    try:
        saved = await save_upload(
            file=file,
            upload_dir=settings.upload_dir,
            max_size_bytes=settings.max_upload_size_mb * 1024 * 1024,
            allowed_extensions={".pdf"},
            allowed_content_types={"application/pdf"},
        )
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # --- Page-count gate: reject before Azure/LLM ---
    from pathlib import Path

    max_pages = settings.max_upload_pages
    if max_pages > 0:
        page_count = _count_pdf_pages(Path(saved.stored_path))
        if page_count is not None and page_count > max_pages:
            await asyncio.to_thread(cleanup_processing_artifacts, saved, settings)
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": True,
                    "error_type": "document_too_large",
                    "detail": (
                        f"El documento '{saved.filename}' tiene {page_count} páginas "
                        f"y excede el máximo permitido de {max_pages}."
                    ),
                    "filename": saved.filename,
                    "pages": page_count,
                    "max_pages": max_pages,
                    "document_id": saved.id,
                },
            )

    if coordinator is not None:
        try:
            await coordinator.submit(job_type=JOB_TYPE_PATIENT, saved=saved)
        except Exception:
            await asyncio.to_thread(cleanup_processing_artifacts, saved, settings)
            raise
        return await coordinator.wait_for_result(saved.id)

    runtime = RuntimeServices(
        settings=settings,
        parser=parser,
        extractor=extractor,
        processing_limiter=limiter,
        job_queue=None,
    )
    return await run_direct_patient(saved=saved, runtime=runtime)
