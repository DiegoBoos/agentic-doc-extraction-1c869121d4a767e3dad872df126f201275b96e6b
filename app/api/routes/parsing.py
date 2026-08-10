import json
import logging
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import Settings
from app.db.connectiondb import save_billing_metadata
from app.dependencies.services import get_extractor, get_parser, get_settings, require_api_key
from app.services.document_parser import DocumentParserRouter
from app.services.file_ingest import FileTooLargeError, save_upload
from app.services.openai_extractor import LLMConnectionError, OpenAIExtractorService
from app.services.utils import normalize_nit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/parse", tags=["parse"])

ParserDep = Annotated[DocumentParserRouter, Depends(get_parser)]
ExtractorDep = Annotated[OpenAIExtractorService, Depends(get_extractor)]
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
    except FileTooLargeError as e:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(e),
        ) from e
    except ValueError as e:
        # Unsupported extension or content-type — client error, not a 500
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    try:
        parsed = parser.parse_document(
            document_path=Path(saved.stored_path),
            document_id=saved.id,
        )
        extraction, tokens_input, tokens_output = await extractor.extract_authorization(
            parsed.markdown
        )

        # Extraction and recording of billing metadata
        azure_model_id = parsed.model
        extracted_pages = 0
        try:
            with open(parsed.json_output_path, "r", encoding="utf-8") as f:
                azure_data = json.load(f)
            pages = azure_data.get("pages", [])
            extracted_pages = len(pages)
        except Exception:
            pass

        processed_authorizations = 0
        if hasattr(extraction, "authorizations") and extraction.authorizations:
            processed_authorizations = len(extraction.authorizations)
        elif hasattr(extraction, "authorizations_list") and extraction.authorizations_list:
            processed_authorizations = len(extraction.authorizations_list)
        elif isinstance(extraction, list):
            processed_authorizations = len(extraction)

        save_billing_metadata(
            document_id=saved.id,
            filename=file.filename,
            extracted_pages=extracted_pages,
            azure_model_id=azure_model_id,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            processed_authorizations=processed_authorizations,
        )

        # Normalize any NIT/identification numbers returned by the extractor
        data = extraction.model_dump()

        try:
            # Handle both single AuthorizationExtraction (old) and AuthorizationResponse (new)
            auth_list = data.get("authorizations") if "authorizations" in data else [data]

            for auth_item in auth_list:
                # prestador autorizado NIT
                pa = auth_item.get("prestador_autorizado") or {}
                if "numero_identificacion_o_nit" in pa:
                    pa["numero_identificacion_o_nit"] = normalize_nit(
                        pa.get("numero_identificacion_o_nit")
                    )

                dp = auth_item.get("datos_paciente") or {}
                if "numero_identificacion" in dp:
                    dp["numero_identificacion"] = normalize_nit(dp.get("numero_identificacion"))

            # Filter out authorizations without numero_autorizacion
            filtered = [
                auth_item
                for auth_item in auth_list
                if auth_item.get("numero_autorizacion") not in (None, "")
            ]

            # Deduplicate by numero_autorizacion (14-digit), keeping first occurrence
            seen: set[str] = set()
            deduped: list[dict] = []
            for auth_item in filtered:
                num = auth_item.get("numero_autorizacion", "")
                if num not in seen:
                    seen.add(num)
                    deduped.append(auth_item)

            if "authorizations" in data:
                data["authorizations"] = deduped
            else:
                if not filtered:
                    data = {"authorizations": []}
        except Exception:
            # defensive: if normalization fails, return raw extraction
            return extraction.model_dump()

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
