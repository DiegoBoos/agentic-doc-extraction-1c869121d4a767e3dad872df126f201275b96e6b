import asyncio
from types import SimpleNamespace

import pytest
from redis.exceptions import ResponseError

from app import worker as worker_module


@pytest.mark.asyncio
async def test_worker_reconnects_after_redis_failover(monkeypatch) -> None:
    processed: list[str] = []

    class _FakeQueue:
        def __init__(self) -> None:
            self.dequeue_calls = 0
            self.reconnect_calls = 0
            self.close_calls = 0

        async def dequeue(self, *, timeout_seconds: int = 5) -> str | None:
            self.dequeue_calls += 1
            if self.dequeue_calls == 1:
                raise ResponseError(
                    "UNBLOCKED force unblock from blocking operation, instance state changed"
                )
            if self.dequeue_calls == 2:
                return "job-1"
            raise asyncio.CancelledError()

        async def reconnect(self) -> None:
            self.reconnect_calls += 1

        async def close(self) -> None:
            self.close_calls += 1

    queue = _FakeQueue()
    runtime = SimpleNamespace(
        settings=SimpleNamespace(
            processing_max_concurrent_documents=1,
            job_queue_name="agentic-doc-extraction:jobs",
        ),
        job_queue=queue,
    )

    async def _fake_build_runtime():
        return runtime

    async def _fake_process_job(_runtime, job_id: str) -> None:
        processed.append(job_id)

    monkeypatch.setattr(worker_module, "build_runtime", _fake_build_runtime)
    monkeypatch.setattr(worker_module, "ensure_billing_schema", lambda: None)
    monkeypatch.setattr(worker_module, "process_job", _fake_process_job)

    with pytest.raises(asyncio.CancelledError):
        await worker_module.run_worker()

    assert queue.reconnect_calls == 1
    assert queue.close_calls == 1
    assert processed == ["job-1"]
