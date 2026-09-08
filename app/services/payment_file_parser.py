from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pypdf import PdfReader

from app.schemas.payment_file_report import PaymentFileBeneficiary, PaymentFileReport

if TYPE_CHECKING:
    from app.services.document_parser import DocumentParserRouter

_HEADER_LABELS: list[tuple[str, str]] = [
    ("clave_empresa", "Clave Empresa:"),
    ("cuenta_cargo", "Cuenta de Cargo:"),
    ("fecha_proceso", "F.de proceso:"),
    ("nombre_fichero", "Nombre Fichero:"),
    ("referencia", "Referencia:"),
    ("ordenes", "Órdenes:"),
    ("producto", "Producto:"),
    ("importe_total", "Importe Total:"),
]

_BENEFICIARY_LABELS: list[tuple[str, str]] = [
    ("tipo_identificacion", "Tipo de identificación:"),
    ("numero_identificacion", "Nº identificación:"),
    ("nombre", "Nombre:"),
    ("email", "E-mail:"),
    ("direccion_1", "Dirección 1:"),
    ("direccion_2", "Dirección 2:"),
    ("forma_pago", "Forma de Pago:"),
    ("tipo_cuenta", "Tipo de cuenta:"),
    ("banco", "Banco:"),
    ("cuenta_tarjeta", "Cuenta-Tarjeta:"),
    ("codigo_oficina_pagadora", "Código Oficina Pagadora:"),
    ("fecha_limite_vencimiento", "Fecha Límite Vencimiento:"),
    ("importe", "Importe:"),
    ("motivo_devolucion", "Motivo Devolución:"),
    ("concepto_1", "Concepto 1:"),
    ("concepto_2", "Concepto 2:"),
]

_BENEFICIARY_BLOCK_MARKER = "Tipo de identificación:"
_REPORT_TITLE_MARKER = "Informe Detallado del Fichero"
_REPORT_TIMESTAMP_PATTERN = re.compile(r"\d{2}/[A-Za-z]{3}/\d{2}\s+\d{2}:\d{2}:\d{2}")


class PaymentFileParseError(ValueError):
    """Raised when the uploaded PDF does not match the expected report layout."""


def _extract_labeled_fields(text: str, labels: list[tuple[str, str]]) -> dict[str, str]:
    key_by_label = {label: key for key, label in labels}
    pattern = "|".join(re.escape(label) for _, label in labels)
    matches = list(re.finditer(pattern, text))

    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = key_by_label[match.group(0)]
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = " ".join(text[start:end].split())
        fields[key] = value
    return fields


def _extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _split_report_texts(text: str) -> list[str]:
    """Split concatenated report text into one chunk per report.

    A single PDF (or OCR output) may contain several "Informe Detallado del
    Fichero" reports back to back (e.g. one per page). Each occurrence of the
    title marker starts a new report.
    """
    starts = [m.start() for m in re.finditer(re.escape(_REPORT_TITLE_MARKER), text)]
    if not starts:
        return []
    chunks = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        chunks.append(text[start:end])
    return chunks


def _table_to_lines(table: list[list[str]]) -> list[str]:
    """Reconstruct row-major text lines from a table's cell grid.

    Joining a row's cells with a space reproduces the same "Label: value
    Label: value" shape that pypdf's native text extraction yields for this
    report's two-column layout, so the resulting lines can be fed through the
    same label-matching parser regardless of source.
    """
    return [" ".join(cell for cell in row if cell) for row in table]


def _find_report_headers(content: str) -> list[str]:
    """Return the timestamp immediately following each report title occurrence."""
    headers = []
    for match in re.finditer(re.escape(_REPORT_TITLE_MARKER), content):
        remainder = content[match.end() : match.end() + 60]
        timestamp_match = _REPORT_TIMESTAMP_PATTERN.search(remainder)
        headers.append(timestamp_match.group(0) if timestamp_match else "")
    return headers


def _reconstruct_report_texts_from_tables(content: str, tables: list[list[list[str]]]) -> list[str]:
    """Rebuild one parseable text blob per report from OCR'd table structure.

    Each report contributes two tables (the header key/value grid and the
    beneficiary detail grid); a scanned PDF with several reports back to back
    (e.g. one per page) yields that many table pairs, in document order.
    """
    timestamps = _find_report_headers(content)
    if not timestamps:
        return []

    table_lines = [_table_to_lines(table) for table in tables]

    if len(timestamps) == 1:
        groups = [table_lines]
    elif len(table_lines) == 2 * len(timestamps):
        groups = [table_lines[i * 2 : i * 2 + 2] for i in range(len(timestamps))]
    else:
        raise PaymentFileParseError(
            "No fue posible asociar las tablas detectadas por el OCR con cada informe "
            "del PDF escaneado (estructura de tablas inesperada)."
        )

    report_texts = []
    for timestamp, group in zip(timestamps, groups, strict=True):
        lines = [_REPORT_TITLE_MARKER, timestamp]
        for table in group:
            lines.extend(table)
        report_texts.append("\n".join(lines))
    return report_texts


def _parse_report_text(text: str) -> PaymentFileReport:
    first_block_idx = text.index(_BENEFICIARY_BLOCK_MARKER)
    table_marker_idx = text.find("BENEFICIARIO")
    header_end = table_marker_idx if 0 <= table_marker_idx < first_block_idx else first_block_idx
    header_fields = _extract_labeled_fields(text[:header_end], _HEADER_LABELS)
    timestamp_match = _REPORT_TIMESTAMP_PATTERN.search(text[:header_end])
    if timestamp_match:
        header_fields["fecha_generacion_informe"] = timestamp_match.group(0)

    detail_text = text[first_block_idx:]
    block_starts = [
        m.start() for m in re.finditer(re.escape(_BENEFICIARY_BLOCK_MARKER), detail_text)
    ]
    beneficiaries = []
    for index, start in enumerate(block_starts):
        end = block_starts[index + 1] if index + 1 < len(block_starts) else len(detail_text)
        block_fields = _extract_labeled_fields(detail_text[start:end], _BENEFICIARY_LABELS)
        beneficiaries.append(PaymentFileBeneficiary(**block_fields))

    return PaymentFileReport(**header_fields, beneficiarios=beneficiaries)


async def parse_payment_file_reports(
    pdf_path: Path,
    *,
    parser: "DocumentParserRouter | None" = None,
    document_id: str = "payment-file",
) -> list[PaymentFileReport]:
    """Extract one or more BBVA Net Cash "Informe Detallado del Fichero" reports from a PDF.

    The report layout is fixed (bank-generated, not user-authored), so fields are
    extracted via label matching rather than an LLM. Text is read directly from the
    PDF's text layer first; if the PDF has no extractable text (a scanned/flattened
    image), it falls back to Azure Document Intelligence OCR. A single PDF may bundle
    several reports back to back, each parsed independently.
    """
    text = _extract_pdf_text(pdf_path)

    if text.strip():
        report_texts = _split_report_texts(text)
    else:
        if parser is None:
            raise PaymentFileParseError(
                "El PDF no contiene texto extraíble y no hay un proveedor de OCR configurado."
            )
        try:
            tables_result = await asyncio.to_thread(
                parser.parse_tables,
                document_path=pdf_path,
                document_id=document_id,
            )
        except ValueError as exc:
            raise PaymentFileParseError(
                f"El PDF no contiene texto extraíble y el OCR no está disponible: {exc}"
            ) from exc
        report_texts = _reconstruct_report_texts_from_tables(
            tables_result.content, tables_result.tables
        )

    reports = [chunk for chunk in report_texts if _BENEFICIARY_BLOCK_MARKER in chunk]

    if not reports:
        raise PaymentFileParseError(
            "El documento no corresponde al formato esperado de 'Informe Detallado del Fichero'."
        )

    return [_parse_report_text(chunk) for chunk in reports]
