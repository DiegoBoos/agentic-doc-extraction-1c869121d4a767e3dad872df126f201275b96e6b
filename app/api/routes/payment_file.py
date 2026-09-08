from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import Settings
from app.dependencies.services import get_settings, require_api_key
from app.schemas.payment_file_report import PaymentFileReport
from app.services.file_ingest import FileTooLargeError, save_upload
from app.services.payment_file_parser import PaymentFileParseError, parse_payment_file_report

router = APIRouter(prefix="/payment-files", tags=["payment-files"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
UploadFileDep = Annotated[UploadFile, File(...)]


@router.post("/parse", status_code=status.HTTP_200_OK, response_model=PaymentFileReport)
async def parse_payment_file(
    file: UploadFileDep,
    settings: SettingsDep,
    _auth: None = Depends(require_api_key),
) -> PaymentFileReport:
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

    try:
        return parse_payment_file_report(Path(saved.stored_path))
    except PaymentFileParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    finally:
        Path(saved.stored_path).unlink(missing_ok=True)
