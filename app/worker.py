from __future__ import annotations

import asyncio
import logging

from redis.exceptions import RedisError

from app.db.connectiondb import ensure_billing_schema
from app.logging_setup import configure_logging
from app.services.job_orchestrator import process_job
from app.services.runtime import build_runtime

configure_logging()
logger = logging.getLogger(__name__)


async def run_worker() -> None:
    configure_logging()
    runtime = await build_runtime()
    ensure_billing_schema()

    if runtime.job_queue is None:
        raise RuntimeError("Worker requires JOB_QUEUE_MODE=redis and a valid Redis connection.")

    max_in_flight = max(1, runtime.settings.processing_max_concurrent_documents)
    reconnect_backoff_seconds = 2
    in_flight: set[asyncio.Task] = set()

    logger.info(
        "Worker ready | queue=%s max_in_flight=%s",
        runtime.settings.job_queue_name,
        max_in_flight,
    )

    try:
        while True:
            while len(in_flight) >= max_in_flight:
                done, pending = await asyncio.wait(
                    in_flight,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                in_flight = set(pending)
                for task in done:
                    task.result()

            try:
                job_id = await runtime.job_queue.dequeue(timeout_seconds=5)
            except RedisError as exc:
                logger.warning(
                    "Redis queue dequeue interrupted | queue=%s error=%s",
                    runtime.settings.job_queue_name,
                    exc,
                )
                try:
                    await runtime.job_queue.reconnect()
                except RedisError as reconnect_exc:
                    logger.warning(
                        "Redis queue reconnect failed | queue=%s error=%s",
                        runtime.settings.job_queue_name,
                        reconnect_exc,
                    )
                    await asyncio.sleep(reconnect_backoff_seconds)
                else:
                    logger.info(
                        "Redis queue reconnected | queue=%s",
                        runtime.settings.job_queue_name,
                    )
                continue

            if job_id is None:
                continue

            logger.info(
                "Dequeued job | job_id=%s in_flight=%s/%s",
                job_id,
                len(in_flight),
                max_in_flight,
            )
            task = asyncio.create_task(process_job(runtime, job_id))
            in_flight.add(task)
            logger.info(
                "Dispatched job to worker task | job_id=%s in_flight=%s/%s",
                job_id,
                len(in_flight),
                max_in_flight,
            )
            task.add_done_callback(in_flight.discard)
    finally:
        if runtime.job_queue is not None:
            await runtime.job_queue.close()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
