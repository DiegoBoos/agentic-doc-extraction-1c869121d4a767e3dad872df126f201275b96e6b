from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies.services import require_api_key
from app.db.connectiondb import get_processing_job
from app.services.job_orchestrator import serialize_processing_job

router = APIRouter(tags=["jobs"])


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    _auth: None = Depends(require_api_key),
) -> dict[str, Any]:
    job = await asyncio.to_thread(get_processing_job, job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Processing job not found: {job_id}",
        )
    return serialize_processing_job(job)
