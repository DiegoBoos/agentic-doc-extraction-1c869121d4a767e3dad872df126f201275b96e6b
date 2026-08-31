from __future__ import annotations

import logging


class _HealthCheckAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if '"GET /health HTTP/1.1" 200' in message:
            return False
        return True


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.core").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    access_logger = logging.getLogger("uvicorn.access")
    access_logger.setLevel(logging.INFO)

    has_filter = any(isinstance(f, _HealthCheckAccessFilter) for f in access_logger.filters)
    if not has_filter:
        health_filter = _HealthCheckAccessFilter()
        access_logger.addFilter(health_filter)
        for handler in access_logger.handlers:
            handler.addFilter(health_filter)
