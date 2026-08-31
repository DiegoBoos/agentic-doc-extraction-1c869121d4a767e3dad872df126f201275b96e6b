from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from app.core.config import Settings
from app.db.connectiondb import (
    BILLING_STATUS_FAILED,
    BILLING_STATUS_SUCCEEDED,
    BILLING_STATUS_SYNCING,
    save_billing_metadata,
    update_processing_job_billing,
)
from app.schemas.patient import PatientDataResponse
from app.services.document_parser import DocumentParserRouter
from app.services.fhir_mapper import patient_data_to_fhir
from app.services.file_ingest import SavedUpload
from app.services.openai_extractor import LLMConnectionError, OpenAIExtractorService
from app.services.processing_limiter import ProcessingLimiter
from app.services.utils import normalize_nit


class DocumentTooManyPagesError(ValueError):
    """Raised when a PDF exceeds the maximum allowed page count."""

    def __init__(self, *, pages: int, max_pages: int, filename: str) -> None:
        super().__init__(
            f"El documento '{filename}' tiene {pages} páginas y excede el "
            f"máximo permitido de {max_pages}."
        )
        self.pages = pages
        self.max_pages = max_pages


def _count_pdf_pages(file_path: Path) -> int | None:
    """Count pages in a PDF using pypdf.  Returns None for non-PDF files."""
    if file_path.suffix.lower() != ".pdf":
        return None
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        return len(reader.pages)
    except Exception:
        logger.warning("Could not count pages for %s", file_path)
        return None

logger = logging.getLogger(__name__)

JOB_TYPE_PARSE = "parse"
JOB_TYPE_PATIENT = "extract_patient"


class BillingPersistenceError(RuntimeError):
    pass


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
    job_id: str | None = None,
    file_hash: str | None = None,
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

        # --- Page-count gate: reject before calling Azure/LLM ---
        max_pages = settings.max_upload_pages
        if max_pages > 0:
            page_count = await asyncio.to_thread(_count_pdf_pages, Path(saved.stored_path))
            if page_count is not None and page_count > max_pages:
                logger.warning(
                    "Document rejected (too many pages) | document_id=%s file=%s pages=%s max=%s",
                    saved.id,
                    saved.filename,
                    page_count,
                    max_pages,
                )
                raise DocumentTooManyPagesError(
                    pages=page_count, max_pages=max_pages, filename=saved.filename
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

        await persist_billing_metadata_with_retry(
            settings=settings,
            job_id=job_id,
            document_id=saved.id,
            filename=saved.filename,
            extracted_pages=extracted_pages,
            azure_model_id=parsed.model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            processed_authorizations=processed_authorizations,
            file_hash=file_hash or saved.file_hash,
        )

        payload = normalize_authorizations(
            extraction.model_dump(),
            markdown=parsed.markdown,
        )

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
    job_id: str | None = None,
    file_hash: str | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()

    async with _optional_processing_slot(limiter) as slot:
        logger.info(
            "Accepted patient pipeline | document_id=%s file=%s wait_ms=%s",
            saved.id,
            saved.filename,
            f"{slot.wait_ms:.2f}" if slot else "0.00",
        )

        # --- Page-count gate: reject before calling Azure/LLM ---
        max_pages = settings.max_upload_pages
        if max_pages > 0:
            page_count = await asyncio.to_thread(_count_pdf_pages, Path(saved.stored_path))
            if page_count is not None and page_count > max_pages:
                logger.warning(
                    "Document rejected (too many pages) | document_id=%s file=%s pages=%s max=%s",
                    saved.id,
                    saved.filename,
                    page_count,
                    max_pages,
                )
                raise DocumentTooManyPagesError(
                    pages=page_count, max_pages=max_pages, filename=saved.filename
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
        await persist_billing_metadata_with_retry(
            settings=settings,
            job_id=job_id,
            document_id=saved.id,
            filename=saved.filename,
            extracted_pages=extracted_pages,
            azure_model_id=parsed.model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            processed_authorizations=0,
            file_hash=file_hash or saved.file_hash,
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


def _has_database_url() -> bool:
    import os

    return bool(os.getenv("DATABASE_URL") or os.getenv("DOC_EXTRACTION_DATABASE_URL"))


async def persist_billing_metadata_with_retry(
    *,
    settings: Settings,
    job_id: str | None,
    document_id: str,
    filename: str,
    extracted_pages: int,
    azure_model_id: str,
    tokens_input: int,
    tokens_output: int,
    processed_authorizations: int,
    file_hash: str | None = None,
) -> None:
    strict = _has_database_url()
    max_retries = max(1, settings.billing_max_retries)
    retry_delay_seconds = max(0, settings.billing_retry_delay_ms) / 1000
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            if job_id is not None:
                await asyncio.to_thread(
                    update_processing_job_billing,
                    job_id,
                    billing_status=BILLING_STATUS_SYNCING,
                    billing_attempts=attempt,
                    billing_error_message=None,
                )

            await asyncio.to_thread(
                save_billing_metadata,
                document_id=document_id,
                filename=filename,
                extracted_pages=extracted_pages,
                azure_model_id=azure_model_id,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                processed_authorizations=processed_authorizations,
                strict=strict,
                file_hash=file_hash,
            )

            if job_id is not None:
                await asyncio.to_thread(
                    update_processing_job_billing,
                    job_id,
                    billing_status=BILLING_STATUS_SUCCEEDED,
                    billing_attempts=attempt,
                    billing_error_message=None,
                )
            return
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Billing persistence failed | document_id=%s attempt=%s/%s error=%s",
                document_id,
                attempt,
                max_retries,
                exc,
            )
            if job_id is not None:
                await asyncio.to_thread(
                    update_processing_job_billing,
                    job_id,
                    billing_status=BILLING_STATUS_FAILED,
                    billing_attempts=attempt,
                    billing_error_message=str(exc),
                )
            if attempt < max_retries and retry_delay_seconds > 0:
                await asyncio.sleep(retry_delay_seconds)

    if strict:
        raise BillingPersistenceError(
            "No fue posible guardar la facturación del documento tras "
            f"{max_retries} intento(s): {last_error}"
        )
    logger.error(
        "Billing persistence exhausted retries (non-strict) | document_id=%s error=%s",
        document_id,
        last_error,
    )


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


_LOCATION_CANONICAL = {
    "ambulatorio": "Ambulatorio",
    "hospitalario": "Hospitalario",
    "urgencias": "Urgencias",
    "domiciliario": "Domiciliario",
}

_GROUP_CANONICAL = {
    "consulta externa": "Consulta externa",
    "hospitalización": "Hospitalización",
    "hospitalizacion": "Hospitalización",
    "cirugía": "Cirugía",
    "cirugia": "Cirugía",
    "apoyo diagnóstico": "Apoyo diagnóstico",
    "apoyo diagnostico": "Apoyo diagnóstico",
}


def _normalize_date_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    match = re.fullmatch(r"(\d{4})[-/](\d{2})[-/](\d{2})", text)
    if match:
        return f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
    match = re.fullmatch(r"(\d{2})[-/](\d{2})[-/](\d{4})", text)
    if match:
        return f"{match.group(3)}/{match.group(2)}/{match.group(1)}"
    return text


def _normalize_generic_number(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str):
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) < 3:
        return None
    return digits


def _extract_document_vigencia(markdown: str | None) -> str | None:
    if not markdown:
        return None
    patterns = [
        r"vigencia[^\n:]{0,40}[:\-]?\s*(\d{1,3})\s*d[ií]as?",
        r"(\d{1,3})\s*d[ií]as?\s+de\s+vigencia",
        r"vigencia[^\n]{0,80}?(\d{1,3})\s*d[ií]as?",
    ]
    for pattern in patterns:
        match = re.search(pattern, markdown, flags=re.IGNORECASE)
        if match:
            return f"{match.group(1)} dias"
    return None


def _extract_document_solicitud_numbers(markdown: str | None) -> list[str]:
    if not markdown:
        return []

    patterns = [
        r"no\.?\s*solicitud\s*[:\-#]?\s*(\d{3,20})",
        r"solicitud\s+no\.?\s*[:\-#]?\s*(\d{3,20})",
    ]
    found: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, markdown, flags=re.IGNORECASE):
            digits = _normalize_generic_number(match.group(1))
            if digits and digits not in seen:
                seen.add(digits)
                found.append(digits)
    return found


def _normalize_vigencia_value(value: Any, *, document_vigencia: str | None = None) -> Any:
    if document_vigencia:
        return document_vigencia
    if not isinstance(value, str):
        return value
    text = " ".join(value.strip().split())
    match = re.search(r"(\d{1,3})\s*d[ií]as?", text, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)} dias"
    return text or None


def _canonical_location(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"no especificada", "no especificado", "n/a", "na", "null"}:
        return None
    if lowered in _GROUP_CANONICAL:
        return None
    return _LOCATION_CANONICAL.get(lowered)


def _canonical_group(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    if not text:
        return None
    for token in re.split(r"[;|,/]", text):
        lowered = token.strip().lower()
        if lowered in _GROUP_CANONICAL:
            return _GROUP_CANONICAL[lowered]
    if text.lower() in _LOCATION_CANONICAL:
        return None
    return None


def _extract_document_location(markdown: str | None) -> str | None:
    if not markdown:
        return None
    lowered = markdown.lower()
    for key, canonical in _LOCATION_CANONICAL.items():
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            return canonical
    return None


def _extract_document_group(markdown: str | None) -> str | None:
    if not markdown:
        return None
    lowered = markdown.lower()
    for key, canonical in _GROUP_CANONICAL.items():
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            return canonical
    return None


def normalize_authorizations(
    data: dict[str, Any], *, markdown: str | None = None
) -> dict[str, Any]:
    try:
        auth_list = data.get("authorizations") if "authorizations" in data else [data]
        document_vigencia = _extract_document_vigencia(markdown)
        document_location = _extract_document_location(markdown)
        document_group = _extract_document_group(markdown)
        document_solicitudes = _extract_document_solicitud_numbers(markdown)

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

            auth_item["fecha_autorizacion"] = _normalize_date_text(
                auth_item.get("fecha_autorizacion")
            )
            auth_item["numero_solicitud"] = _normalize_generic_number(
                auth_item.get("numero_solicitud")
            )
            auth_item["vigencia"] = _normalize_vigencia_value(
                auth_item.get("vigencia"),
                document_vigencia=document_vigencia,
            )

            servicios = auth_item.get("servicios_autorizados") or {}
            grupo = _canonical_group(servicios.get("grupo_servicio"))
            ubicacion = _canonical_location(servicios.get("ubicacion_paciente"))

            if ubicacion is None and grupo is None:
                grupo = document_group
                ubicacion = document_location
            else:
                grupo = grupo or document_group
                ubicacion = ubicacion or document_location

            servicios["grupo_servicio"] = grupo
            servicios["ubicacion_paciente"] = ubicacion
            auth_item["servicios_autorizados"] = servicios

        if not auth_list and document_solicitudes:
            auth_list = [
                {
                    "numero_autorizacion": None,
                    "numero_solicitud": solicitud,
                    "fecha_autorizacion": None,
                    "prestador_autorizado": {},
                    "datos_paciente": {},
                    "servicios_autorizados": {
                        "ubicacion_paciente": document_location,
                        "grupo_servicio": document_group,
                        "items": [],
                    },
                    "vigencia": document_vigencia,
                }
                for solicitud in document_solicitudes
            ]

        filtered = [
            auth_item
            for auth_item in auth_list
            if auth_item.get("numero_autorizacion") not in (None, "")
            or auth_item.get("numero_solicitud") not in (None, "")
        ]

        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for auth_item in filtered:
            numero = auth_item.get("numero_autorizacion") or ""
            solicitud = auth_item.get("numero_solicitud") or ""
            identity = f"auth:{numero}" if numero else f"sol:{solicitud}"
            if identity not in seen:
                seen.add(identity)
                deduped.append(auth_item)

        if "authorizations" in data:
            data["authorizations"] = deduped
        elif not filtered:
            data = {"authorizations": []}
    except Exception:
        return data

    return data


def map_processing_exception(exc: Exception) -> tuple[int, str, str]:
    if isinstance(exc, DocumentTooManyPagesError):
        return (
            status.HTTP_400_BAD_REQUEST,
            "document_too_large",
            str(exc),
        )
    if isinstance(exc, LLMConnectionError):
        return (
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "llm_unavailable",
            "Servicio de extracción LLM no disponible. Intente de nuevo más tarde.",
        )
    if isinstance(exc, BillingPersistenceError):
        return (
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "billing_persistence_error",
            "No fue posible guardar la facturación del documento.",
        )
    if isinstance(exc, ValueError):
        return status.HTTP_400_BAD_REQUEST, "validation_error", str(exc)
    return status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", str(exc)


def build_http_exception(exc: Exception) -> HTTPException:
    status_code, _, detail = map_processing_exception(exc)
    return HTTPException(status_code=status_code, detail=detail)
