from __future__ import annotations

from redis.asyncio import Redis


class RedisJobQueue:
    def __init__(self, *, redis_url: str, queue_name: str) -> None:
        self.queue_name = queue_name
        # socket_timeout must exceed the longest BLPOP timeout to avoid
        # the client killing the connection while Redis is still blocking.
        self.client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=30.0,
            socket_connect_timeout=10.0,
        )

    async def ping(self) -> None:
        await self.client.ping()

    async def enqueue(self, job_id: str) -> None:
        await self.client.rpush(self.queue_name, job_id)

    async def dequeue(self, *, timeout_seconds: int = 5) -> str | None:
        result = await self.client.blpop(self.queue_name, timeout=timeout_seconds)
        if result is None:
            return None
        _, job_id = result
        return job_id

    async def close(self) -> None:
        await self.client.aclose()
