from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import Settings
from app.db.connectiondb import save_billing_metadata
from app.dependencies.services import (
    get_extractor,
    get_parser,
    get_processing_limiter,
    get_settings,
    require_api_key,
)
from app.schemas.patient import PatientDataResponse
from app.services.document_parser import DocumentParserRouter
from app.services.fhir_mapper import patient_data_to_fhir
from app.services.file_ingest import FileTooLargeError, save_upload
from app.services.openai_extractor import LLMConnectionError, OpenAIExtractorService
from app.services.processing_limiter import ProcessingLimiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extract-patient", tags=["patient"])

ParserDep = Annotated[DocumentParserRouter, Depends(get_parser)]
ExtractorDep = Annotated[OpenAIExtractorService, Depends(get_extractor)]
LimiterDep = Annotated[ProcessingLimiter, Depends(get_processing_limiter)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
UploadFileDep = Annotated[UploadFile, File(...)]


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

    started_at = time.perf_counter()

    try:
        async with limiter.acquire() as slot:
            logger.info(
                "Accepted patient extraction job | document_id=%s file=%s "
                "wait_ms=%.2f active_jobs=%s",
                saved.id,
                saved.filename,
                slot.wait_ms,
                slot.active_jobs,
            )

            parsed = await asyncio.to_thread(
                parser.parse_first_page,
                document_path=Path(saved.stored_path),
                document_id=saved.id,
            )
            extraction, tokens_input, tokens_output = await extractor.extract_patient_data(
                parsed.markdown
            )

            extracted_pages = _count_extracted_pages(parsed.json_output_path)
            await asyncio.to_thread(
                save_billing_metadata,
                document_id=saved.id,
                filename=file.filename,
                extracted_pages=extracted_pages,
                azure_model_id=parsed.model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                processed_authorizations=0,
            )

            flat_response = PatientDataResponse(
                document_id=saved.id,
                filename=saved.filename,
                content_type=saved.content_type,
                size_bytes=saved.size_bytes,
                created_at=saved.created_at,
                provider=parsed.provider,
                model=parsed.model,
                extracted_pages=extracted_pages,
                chunk_count=len(parsed.chunks),
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                patient=extraction.patient,
            )
            fhir_patient = patient_data_to_fhir(extraction.patient)

            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "Completed patient extraction job | document_id=%s file=%s "
                "tokens_in=%s tokens_out=%s elapsed_ms=%.2f",
                saved.id,
                saved.filename,
                tokens_input,
                tokens_output,
                elapsed_ms,
            )

            return {
                **flat_response.model_dump(mode="json"),
                "fhir": fhir_patient,
            }

    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Archivo demasiado grande para procesar.",
        ) from exc
    except LLMConnectionError as exc:
        logger.error("LLM unavailable for patient extraction %s: %s", saved.id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio de extracción LLM no disponible. Intente de nuevo más tarde.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "Unhandled error extracting patient data | file=%s document_id=%s",
            file.filename,
            saved.id,
        )
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(exc)}") from exc
    finally:
        try:
            Path(saved.stored_path).unlink(missing_ok=True)
            shutil.rmtree(settings.parse_output_dir / saved.id, ignore_errors=True)
        except Exception:
            pass


def _count_extracted_pages(json_output_path: str) -> int:
    try:
        with Path(json_output_path).open("r", encoding="utf-8") as handle:
            azure_data = json.load(handle)
        pages = azure_data.get("pages", [])
        return len(pages)
    except Exception:
        return 0
