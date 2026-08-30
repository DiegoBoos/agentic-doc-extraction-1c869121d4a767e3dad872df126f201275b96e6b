from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator

import psycopg2
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
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                conn.commit()
                _schema_ready = True
            except Exception as exc:
                conn.rollback()
                print("Error al preparar billing_metadata:", exc)
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


if __name__ == "__main__":
    verificar_conexion()
