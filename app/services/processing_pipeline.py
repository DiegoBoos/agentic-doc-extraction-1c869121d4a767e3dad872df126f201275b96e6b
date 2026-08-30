from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from app.core.config import Settings
from app.db.connectiondb import save_billing_metadata
from app.schemas.patient import PatientDataResponse
from app.services.document_parser import DocumentParserRouter
from app.services.fhir_mapper import patient_data_to_fhir
from app.services.file_ingest import SavedUpload
from app.services.openai_extractor import LLMConnectionError, OpenAIExtractorService
from app.services.processing_limiter import ProcessingLimiter
from app.services.utils import normalize_nit

logger = logging.getLogger(__name__)

JOB_TYPE_PARSE = "parse"
JOB_TYPE_PATIENT = "extract_patient"


@asynccontextmanager
async def _optional_processing_slot(limiter: ProcessingLimiter | None):
    if limiter is None:
        yield None
        return

    async with limiter.acquire() as slot:
        yield slot


async def run_parse_pipeline(
    *,
    saved: SavedUpload,
    parser: DocumentParserRouter,
    settings: Settings,
    extractor: OpenAIExtractorService,
    limiter: ProcessingLimiter | None,
) -> dict[str, Any]:
    started_at = time.perf_counter()

    async with _optional_processing_slot(limiter) as slot:
        logger.info(
            "Accepted parse pipeline | document_id=%s file=%s size_bytes=%s wait_ms=%s",
            saved.id,
            saved.filename,
            saved.size_bytes,
            f"{slot.wait_ms:.2f}" if slot else "0.00",
        )

        parsed = await asyncio.to_thread(
            parser.parse_document,
            document_path=Path(saved.stored_path),
            document_id=saved.id,
        )
        extracted_pages = count_extracted_pages(parsed.json_output_path)

        if extracted_pages >= settings.processing_large_document_page_threshold:
            logger.warning(
                "Large document detected | document_id=%s file=%s pages=%s threshold=%s",
                saved.id,
                saved.filename,
                extracted_pages,
                settings.processing_large_document_page_threshold,
            )

        extraction, tokens_input, tokens_output = await extractor.extract_authorization(
            parsed.markdown,
            parsed.chunks,
        )
        processed_authorizations = count_processed_authorizations(extraction)

        await asyncio.to_thread(
            save_billing_metadata,
            document_id=saved.id,
            filename=saved.filename,
            extracted_pages=extracted_pages,
            azure_model_id=parsed.model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            processed_authorizations=processed_authorizations,
        )

        payload = normalize_authorizations(extraction.model_dump())

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "Completed parse pipeline | document_id=%s file=%s pages=%s tokens_in=%s "
            "tokens_out=%s authorizations=%s elapsed_ms=%.2f",
            saved.id,
            saved.filename,
            extracted_pages,
            tokens_input,
            tokens_output,
            processed_authorizations,
            elapsed_ms,
        )
        return payload


async def run_patient_pipeline(
    *,
    saved: SavedUpload,
    parser: DocumentParserRouter,
    settings: Settings,
    extractor: OpenAIExtractorService,
    limiter: ProcessingLimiter | None,
) -> dict[str, Any]:
    started_at = time.perf_counter()

    async with _optional_processing_slot(limiter) as slot:
        logger.info(
            "Accepted patient pipeline | document_id=%s file=%s wait_ms=%s",
            saved.id,
            saved.filename,
            f"{slot.wait_ms:.2f}" if slot else "0.00",
        )

        parsed = await asyncio.to_thread(
            parser.parse_first_page,
            document_path=Path(saved.stored_path),
            document_id=saved.id,
        )
        extraction, tokens_input, tokens_output = await extractor.extract_patient_data(
            parsed.markdown
        )

        extracted_pages = count_extracted_pages(parsed.json_output_path)
        await asyncio.to_thread(
            save_billing_metadata,
            document_id=saved.id,
            filename=saved.filename,
            extracted_pages=extracted_pages,
            azure_model_id=parsed.model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            processed_authorizations=0,
        )

        payload = PatientDataResponse(
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
        ).model_dump(mode="json")
        payload["fhir"] = patient_data_to_fhir(extraction.patient)

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "Completed patient pipeline | document_id=%s file=%s tokens_in=%s "
            "tokens_out=%s elapsed_ms=%.2f",
            saved.id,
            saved.filename,
            tokens_input,
            tokens_output,
            elapsed_ms,
        )
        return payload


def cleanup_processing_artifacts(saved: SavedUpload, settings: Settings) -> None:
    try:
        Path(saved.stored_path).unlink(missing_ok=True)
        shutil.rmtree(settings.parse_output_dir / saved.id, ignore_errors=True)
    except Exception:
        logger.exception("Failed cleaning processing artifacts | document_id=%s", saved.id)


def count_extracted_pages(json_output_path: str) -> int:
    try:
        with Path(json_output_path).open("r", encoding="utf-8") as handle:
            parsed_payload = json.load(handle)
        pages = parsed_payload.get("pages", [])
        if isinstance(pages, list) and pages:
            return len(pages)
        if "responses" in parsed_payload and isinstance(parsed_payload["responses"], list):
            return len(parsed_payload["responses"])
    except Exception:
        return 0
    return 0


def count_processed_authorizations(extraction: Any) -> int:
    if hasattr(extraction, "authorizations") and extraction.authorizations:
        return len(extraction.authorizations)
    if hasattr(extraction, "authorizations_list") and extraction.authorizations_list:
        return len(extraction.authorizations_list)
    if isinstance(extraction, list):
        return len(extraction)
    return 0


def normalize_authorizations(data: dict[str, Any]) -> dict[str, Any]:
    try:
        auth_list = data.get("authorizations") if "authorizations" in data else [data]

        for auth_item in auth_list:
            prestador = auth_item.get("prestador_autorizado") or {}
            if "numero_identificacion_o_nit" in prestador:
                prestador["numero_identificacion_o_nit"] = normalize_nit(
                    prestador.get("numero_identificacion_o_nit")
                )

            datos_paciente = auth_item.get("datos_paciente") or {}
            if "numero_identificacion" in datos_paciente:
                datos_paciente["numero_identificacion"] = normalize_nit(
                    datos_paciente.get("numero_identificacion")
                )

        filtered = [
            auth_item
            for auth_item in auth_list
            if auth_item.get("numero_autorizacion") not in (None, "")
        ]

        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for auth_item in filtered:
            numero = auth_item.get("numero_autorizacion", "")
            if numero not in seen:
                seen.add(numero)
                deduped.append(auth_item)

        if "authorizations" in data:
            data["authorizations"] = deduped
        elif not filtered:
            data = {"authorizations": []}
    except Exception:
        return data

    return data


def map_processing_exception(exc: Exception) -> tuple[int, str, str]:
    if isinstance(exc, LLMConnectionError):
        return (
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "llm_unavailable",
            "Servicio de extracción LLM no disponible. Intente de nuevo más tarde.",
        )
    if isinstance(exc, ValueError):
        return status.HTTP_400_BAD_REQUEST, "validation_error", str(exc)
    return status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", str(exc)


def build_http_exception(exc: Exception) -> HTTPException:
    status_code, _, detail = map_processing_exception(exc)
    return HTTPException(status_code=status_code, detail=detail)
