from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass(slots=True)
class ProcessingSlot:
    active_jobs: int
    wait_ms: float


class ProcessingLimiter:
    def __init__(self, max_concurrent_jobs: int) -> None:
        self.max_concurrent_jobs = max(1, int(max_concurrent_jobs))
        self._semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        self._active_jobs = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self) -> ProcessingSlot:
        wait_started = time.perf_counter()
        await self._semaphore.acquire()
        wait_ms = (time.perf_counter() - wait_started) * 1000

        async with self._lock:
            self._active_jobs += 1
            active_jobs = self._active_jobs

        try:
            yield ProcessingSlot(active_jobs=active_jobs, wait_ms=wait_ms)
        finally:
            async with self._lock:
                self._active_jobs -= 1
            self._semaphore.release()
