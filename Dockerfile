# Multi-stage build for production deployment using uv
FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies and uv without relying on GHCR side images
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && pip install --no-cache-dir uv \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
RUN uv sync --frozen --no-dev --no-install-project

# Production stage
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY ./app ./app
COPY README.md ./
COPY docker-entrypoint.sh ./docker-entrypoint.sh

# Set PATH to use venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV APP_ROLE=api
ENV WEB_CONCURRENCY=2
ENV UVICORN_LIMIT_CONCURRENCY=16
ENV UVICORN_TIMEOUT_KEEP_ALIVE=10

# Create non-root user
RUN chmod +x /app/docker-entrypoint.sh \
    && useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5090

# Health check — for workers (APP_ROLE=worker) verify the process is alive;
# for the API role fall back to the HTTP endpoint.
HEALTHCHECK --interval=30s --timeout=20s --start-period=45s --retries=5 \
    CMD if [ "$APP_ROLE" = "worker" ]; then pgrep -f "app.worker" > /dev/null; else curl -f http://localhost:5090/health || exit 1; fi

# Run API or worker role with the same image
CMD ["/app/docker-entrypoint.sh"]
