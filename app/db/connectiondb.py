from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Iterator

import psycopg2
from psycopg2.extras import Json, RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

_pool: ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()
_schema_ready = False
_schema_lock = threading.Lock()


def _get_database_url() -> str | None:
    return os.getenv("DATABASE_URL") or os.getenv("DOC_EXTRACTION_DATABASE_URL")


def _build_pool() -> ThreadedConnectionPool | None:
    database_url = _get_database_url()
    if not database_url:
        print("No se encontró la variable de entorno 'DATABASE_URL'. Configúrala primero.")
        return None

    minconn = max(1, int(os.getenv("DATABASE_POOL_MIN", "1")))
    maxconn = max(minconn, int(os.getenv("DATABASE_POOL_MAX", "8")))
    return ThreadedConnectionPool(minconn=minconn, maxconn=maxconn, dsn=database_url)


def get_connection_pool() -> ThreadedConnectionPool | None:
    global _pool
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is None:
            _pool = _build_pool()
    return _pool


@contextmanager
def get_connection() -> Iterator[psycopg2.extensions.connection | None]:
    pool = get_connection_pool()
    if pool is None:
        yield None
        return

    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


def ensure_billing_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return

    with _schema_lock:
        if _schema_ready:
            return

        with get_connection() as conn:
            if conn is None:
                return

            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS billing_metadata (
                        id SERIAL PRIMARY KEY,
                        document_id VARCHAR(255),
                        filename VARCHAR(255),
                        extracted_pages INT,
                        azure_model_id VARCHAR(255),
                        tokens_input INT,
                        tokens_output INT,
                        openai_tokens INT,
                        processed_authorizations INT,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE billing_metadata
                    ADD COLUMN IF NOT EXISTS processed_authorizations INT DEFAULT 0;
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE billing_metadata
                    ADD COLUMN IF NOT EXISTS filename VARCHAR(255);
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE billing_metadata
                    ADD COLUMN IF NOT EXISTS tokens_input INT;
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE billing_metadata
                    ADD COLUMN IF NOT EXISTS tokens_output INT;
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE billing_metadata DROP COLUMN IF EXISTS azure_credits_left;
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS processing_jobs (
                        job_id VARCHAR(255) PRIMARY KEY,
                        job_type VARCHAR(64) NOT NULL,
                        document_id VARCHAR(255) NOT NULL,
                        file_hash VARCHAR(64),
                        filename VARCHAR(255) NOT NULL,
                        content_type VARCHAR(255),
                        size_bytes BIGINT NOT NULL,
                        stored_path TEXT NOT NULL,
                        status VARCHAR(32) NOT NULL DEFAULT 'queued',
                        result_payload JSONB,
                        error_type VARCHAR(64),
                        error_message TEXT,
                        metadata JSONB,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        started_at TIMESTAMPTZ,
                        finished_at TIMESTAMPTZ
                    );
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_processing_jobs_status_created
                    ON processing_jobs (status, created_at);
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_processing_jobs_job_type_hash
                    ON processing_jobs (job_type, file_hash);
                    """
                )
                conn.commit()
                _schema_ready = True
            except Exception as exc:
                conn.rollback()
                print("Error al preparar esquemas de facturación/jobs:", exc)
            finally:
                cursor.close()


def verificar_conexion() -> None:
    with get_connection() as conn:
        if conn is None:
            return

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            resultado = cursor.fetchone()
            if resultado:
                print("La conexión con PostgreSQL funciona correctamente.")
            cursor.close()
        except Exception as exc:
            print("Error al ejecutar query de verificación:", exc)
            conn.rollback()


def is_file_processed(filename: str) -> bool:
    ensure_billing_schema()
    with get_connection() as conn:
        if conn is None:
            return False

        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM billing_metadata WHERE filename = %s LIMIT 1;",
                (str(filename),),
            )
            result = cursor.fetchone()
            cursor.close()
            conn.commit()
            return bool(result)
        except Exception as exc:
            print("Error al verificar si el archivo fue procesado:", exc)
            conn.rollback()
            return False


def save_billing_metadata(
    document_id: str,
    filename: str,
    extracted_pages: int,
    azure_model_id: str,
    tokens_input: int,
    tokens_output: int,
    processed_authorizations: int = 0,
) -> None:
    ensure_billing_schema()
    with get_connection() as conn:
        if conn is None:
            print("No DATABASE_URL configured, skipping billing log.")
            return

        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO billing_metadata
                (document_id, filename, extracted_pages, azure_model_id,
                tokens_input, tokens_output, processed_authorizations)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    str(document_id),
                    str(filename),
                    extracted_pages,
                    azure_model_id,
                    tokens_input,
                    tokens_output,
                    processed_authorizations,
                ),
            )
            conn.commit()
            cursor.close()
            print(f"Facturación guardada para documento: {document_id}")
        except Exception as exc:
            print("Error al guardar metadata de facturación:", exc)
            conn.rollback()


def create_processing_job(
    *,
    job_id: str,
    job_type: str,
    document_id: str,
    file_hash: str,
    filename: str,
    content_type: str | None,
    size_bytes: int,
    stored_path: str,
    created_at: datetime,
    metadata: dict[str, Any] | None = None,
) -> None:
    ensure_billing_schema()
    with get_connection() as conn:
        if conn is None:
            raise RuntimeError("No DATABASE_URL configured, cannot create processing job.")

        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO processing_jobs (
                    job_id, job_type, document_id, file_hash, filename, content_type,
                    size_bytes, stored_path, status, metadata, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'queued', %s, %s)
                ON CONFLICT (job_id) DO NOTHING;
                """,
                (
                    job_id,
                    job_type,
                    document_id,
                    file_hash,
                    filename,
                    content_type,
                    size_bytes,
                    stored_path,
                    Json(metadata or {}),
                    created_at,
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()


def get_processing_job(job_id: str) -> dict[str, Any] | None:
    ensure_billing_schema()
    with get_connection() as conn:
        if conn is None:
            return None

        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(
                "SELECT * FROM processing_jobs WHERE job_id = %s;",
                (job_id,),
            )
            row = cursor.fetchone()
            conn.commit()
            return dict(row) if row else None
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()


def mark_processing_job_running(job_id: str) -> None:
    ensure_billing_schema()
    with get_connection() as conn:
        if conn is None:
            raise RuntimeError("No DATABASE_URL configured, cannot mark processing job.")

        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                UPDATE processing_jobs
                SET status = 'running',
                    started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
                    error_type = NULL,
                    error_message = NULL
                WHERE job_id = %s;
                """,
                (job_id,),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()


def complete_processing_job(job_id: str, payload: dict[str, Any]) -> None:
    ensure_billing_schema()
    with get_connection() as conn:
        if conn is None:
            raise RuntimeError("No DATABASE_URL configured, cannot complete processing job.")

        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                UPDATE processing_jobs
                SET status = 'succeeded',
                    result_payload = %s,
                    error_type = NULL,
                    error_message = NULL,
                    finished_at = CURRENT_TIMESTAMP
                WHERE job_id = %s;
                """,
                (Json(payload), job_id),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()


def fail_processing_job(job_id: str, *, error_type: str, error_message: str) -> None:
    ensure_billing_schema()
    with get_connection() as conn:
        if conn is None:
            raise RuntimeError("No DATABASE_URL configured, cannot fail processing job.")

        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                UPDATE processing_jobs
                SET status = 'failed',
                    error_type = %s,
                    error_message = %s,
                    finished_at = CURRENT_TIMESTAMP
                WHERE job_id = %s;
                """,
                (error_type, error_message, job_id),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()


def get_latest_successful_job_by_hash(job_type: str, file_hash: str) -> dict[str, Any] | None:
    ensure_billing_schema()
    with get_connection() as conn:
        if conn is None:
            return None

        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(
                """
                SELECT *
                FROM processing_jobs
                WHERE job_type = %s
                  AND file_hash = %s
                  AND status = 'succeeded'
                ORDER BY finished_at DESC NULLS LAST
                LIMIT 1;
                """,
                (job_type, file_hash),
            )
            row = cursor.fetchone()
            conn.commit()
            return dict(row) if row else None
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()


def now_utc() -> datetime:
    return datetime.now(UTC)


if __name__ == "__main__":
    verificar_conexion()
