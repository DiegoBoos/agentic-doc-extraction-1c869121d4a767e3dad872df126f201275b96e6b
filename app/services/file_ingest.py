from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


class FileTooLargeError(ValueError):
    def __init__(self, *, size_bytes: int, max_size_bytes: int) -> None:
        super().__init__(f"File too large: {size_bytes} bytes exceeds max {max_size_bytes} bytes")
        self.size_bytes = size_bytes
        self.max_size_bytes = max_size_bytes


@dataclass(slots=True)
class SavedUpload:
    id: str
    filename: str
    content_type: str | None
    size_bytes: int
    stored_path: str
    created_at: datetime
    file_hash: str


async def save_upload(
    *,
    file: UploadFile,
    upload_dir: Path,
    max_size_bytes: int,
    allowed_extensions: set[str],
    allowed_content_types: set[str],
) -> SavedUpload:
    upload_dir.mkdir(parents=True, exist_ok=True)

    upload_id = uuid4().hex
    safe_name = Path(file.filename or "upload.bin").name
    extension = Path(safe_name).suffix.lower()

    if extension not in allowed_extensions:
        raise ValueError(f"Unsupported file extension: {extension}")

    if not file.content_type or file.content_type not in allowed_content_types:
        raise ValueError(f"Unsupported content type: {file.content_type}")

    destination = upload_dir / f"{upload_id}-{safe_name}"

    size = 0
    digest = hashlib.sha256()
    with destination.open("wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_size_bytes:
                handle.close()
                destination.unlink(missing_ok=True)
                raise FileTooLargeError(size_bytes=size, max_size_bytes=max_size_bytes)
            handle.write(chunk)
            digest.update(chunk)

    return SavedUpload(
        id=upload_id,
        filename=safe_name,
        content_type=file.content_type,
        size_bytes=size,
        stored_path=str(destination),
        created_at=datetime.now(UTC),
        file_hash=digest.hexdigest(),
    )
