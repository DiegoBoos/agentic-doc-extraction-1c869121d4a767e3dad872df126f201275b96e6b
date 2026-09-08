from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, Field, field_validator

_REPORT_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def parse_cop_amount(value: str | None) -> Decimal | None:
    """Parse a Colombian-formatted amount ("1.275.110,30") into a Decimal."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    normalized = text.replace(".", "").replace(",", ".")
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


class PaymentFileBeneficiary(BaseModel):
    tipo_identificacion: str | None = None
    numero_identificacion: str | None = None
    nombre: str | None = None
    email: str | None = None
    direccion_1: str | None = None
    direccion_2: str | None = None
    forma_pago: str | None = None
    tipo_cuenta: str | None = None
    banco: str | None = None
    cuenta_tarjeta: str | None = None
    codigo_oficina_pagadora: str | None = None
    fecha_limite_vencimiento: str | None = None
    importe: Decimal | None = None
    motivo_devolucion: str | None = None
    concepto_1: str | None = None
    concepto_2: str | None = None

    @field_validator(
        "tipo_identificacion",
        "numero_identificacion",
        "nombre",
        "email",
        "direccion_1",
        "direccion_2",
        "forma_pago",
        "tipo_cuenta",
        "banco",
        "cuenta_tarjeta",
        "codigo_oficina_pagadora",
        "fecha_limite_vencimiento",
        "motivo_devolucion",
        "concepto_1",
        "concepto_2",
        mode="before",
    )
    @classmethod
    def blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None

    @field_validator("importe", mode="before")
    @classmethod
    def normalize_importe(cls, value: str | Decimal | None) -> Decimal | None:
        if value is None or isinstance(value, Decimal):
            return value
        return parse_cop_amount(str(value))


class PaymentFileReport(BaseModel):
    fecha_generacion_informe: datetime | None = None
    clave_empresa: str | None = None
    cuenta_cargo: str | None = None
    fecha_proceso: date | None = None
    nombre_fichero: str | None = None
    referencia: str | None = None
    ordenes: int | None = None
    producto: str | None = None
    importe_total: Decimal | None = None
    beneficiarios: list[PaymentFileBeneficiary] = Field(default_factory=list)

    @field_validator(
        "clave_empresa", "cuenta_cargo", "nombre_fichero", "referencia", "producto", mode="before"
    )
    @classmethod
    def blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None

    @field_validator("fecha_generacion_informe", mode="before")
    @classmethod
    def normalize_report_timestamp(cls, value: str | datetime | None) -> datetime | None:
        if value is None or isinstance(value, datetime):
            return value
        text = value.strip()
        # Format: "18/Mar/26 08:47:34" — English month abbreviation, 2-digit year.
        parts = text.replace(",", " ").split()
        if len(parts) != 2:
            return None
        date_part, time_part = parts
        day_str, month_str, year_str = date_part.split("/")
        month = _REPORT_MONTHS.get(month_str.strip().lower())
        if month is None:
            return None
        try:
            day = int(day_str)
            year = 2000 + int(year_str)
            hour, minute, second = (int(piece) for piece in time_part.split(":"))
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            return None

    @field_validator("fecha_proceso", mode="before")
    @classmethod
    def normalize_fecha_proceso(cls, value: str | date | None) -> date | None:
        if value is None or isinstance(value, date):
            return value
        text = value.strip()
        try:
            return datetime.strptime(text, "%d-%m-%Y").date()
        except ValueError:
            return None

    @field_validator("ordenes", mode="before")
    @classmethod
    def normalize_ordenes(cls, value: str | int | None) -> int | None:
        if value is None or isinstance(value, int):
            return value
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else None

    @field_validator("importe_total", mode="before")
    @classmethod
    def normalize_importe_total(cls, value: str | Decimal | None) -> Decimal | None:
        if value is None or isinstance(value, Decimal):
            return value
        return parse_cop_amount(str(value))
