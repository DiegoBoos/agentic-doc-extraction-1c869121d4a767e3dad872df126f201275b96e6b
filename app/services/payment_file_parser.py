from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from app.schemas.payment_file_report import PaymentFileBeneficiary, PaymentFileReport

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


def _extract_page_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_payment_file_report(pdf_path: Path) -> PaymentFileReport:
    """Deterministically parse a BBVA Net Cash "Informe Detallado del Fichero" PDF.

    The report layout is fixed (bank-generated, not user-authored), so field
    positions are extracted via label matching rather than an LLM.
    """
    text = _extract_page_text(pdf_path)
    if _REPORT_TITLE_MARKER not in text or _BENEFICIARY_BLOCK_MARKER not in text:
        raise PaymentFileParseError(
            "El documento no corresponde al formato esperado de 'Informe Detallado del Fichero'."
        )

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
