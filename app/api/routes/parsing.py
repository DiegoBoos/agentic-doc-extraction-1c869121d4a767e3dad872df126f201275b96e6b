from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Annotated

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
from app.services.document_parser import DocumentParserRouter
from app.services.file_ingest import FileTooLargeError, save_upload
from app.services.openai_extractor import LLMConnectionError, OpenAIExtractorService
from app.services.processing_limiter import ProcessingLimiter
from app.services.utils import normalize_nit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/parse", tags=["parse"])

ParserDep = Annotated[DocumentParserRouter, Depends(get_parser)]
ExtractorDep = Annotated[OpenAIExtractorService, Depends(get_extractor)]
LimiterDep = Annotated[ProcessingLimiter, Depends(get_processing_limiter)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
UploadFileDep = Annotated[UploadFile, File(...)]


@router.post(
    "",
    status_code=status.HTTP_200_OK,
)
async def parse_document(
    file: UploadFileDep,
    parser: ParserDep,
    settings: SettingsDep,
    extractor: ExtractorDep,
    limiter: LimiterDep,
    _auth: None = Depends(require_api_key),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required")

    try:
        saved = await save_upload(
            file=file,
            upload_dir=settings.upload_dir,
            max_size_bytes=settings.max_upload_size_mb * 1024 * 1024,
            allowed_extensions=set(settings.allowed_upload_extensions),
            allowed_content_types=set(settings.allowed_upload_content_types),
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
                "Accepted parse job | document_id=%s file=%s size_bytes=%s "
                "wait_ms=%.2f active_jobs=%s",
                saved.id,
                saved.filename,
                saved.size_bytes,
                slot.wait_ms,
                slot.active_jobs,
            )

            parsed = await asyncio.to_thread(
                parser.parse_document,
                document_path=Path(saved.stored_path),
                document_id=saved.id,
            )
            extracted_pages = _count_extracted_pages(parsed.json_output_path)

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

            processed_authorizations = _count_processed_authorizations(extraction)

            await asyncio.to_thread(
                save_billing_metadata,
                document_id=saved.id,
                filename=file.filename,
                extracted_pages=extracted_pages,
                azure_model_id=parsed.model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                processed_authorizations=processed_authorizations,
            )

            data = _normalize_authorizations(extraction.model_dump())

            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "Completed parse job | document_id=%s file=%s pages=%s "
                "tokens_in=%s tokens_out=%s authorizations=%s elapsed_ms=%.2f",
                saved.id,
                saved.filename,
                extracted_pages,
                tokens_input,
                tokens_output,
                processed_authorizations,
                elapsed_ms,
            )
            return data

    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Archivo demasiado grande para procesar.",
        ) from exc
    except LLMConnectionError as exc:
        logger.error("LLM unavailable for document %s: %s", saved.id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio de extracción LLM no disponible. Intente de nuevo más tarde.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "Unhandled error processing document | file=%s document_id=%s",
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
        with open(json_output_path, "r", encoding="utf-8") as handle:
            parsed_payload = json.load(handle)
        pages = parsed_payload.get("pages", [])
        if isinstance(pages, list) and pages:
            return len(pages)
        if "responses" in parsed_payload and isinstance(parsed_payload["responses"], list):
            return len(parsed_payload["responses"])
    except Exception:
        return 0
    return 0


def _count_processed_authorizations(extraction) -> int:
    if hasattr(extraction, "authorizations") and extraction.authorizations:
        return len(extraction.authorizations)
    if hasattr(extraction, "authorizations_list") and extraction.authorizations_list:
        return len(extraction.authorizations_list)
    if isinstance(extraction, list):
        return len(extraction)
    return 0


def _normalize_authorizations(data: dict) -> dict:
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
        deduped: list[dict] = []
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
