from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator


class PrestadorAutorizado(BaseModel):
    nombre: str | None = None
    tipo_identificacion_o_nit: str | None = None
    numero_identificacion_o_nit: str | None = None


class DatosPaciente(BaseModel):
    apellido1: str | None = None
    apellido2: str | None = None
    nombre1: str | None = None
    nombre2: str | None = None
    tipo_identificacion: str | None = None
    numero_identificacion: str | None = None


class ServicioAutorizadoItem(BaseModel):
    codigo_cups: str = Field(description="Código CUPS del servicio autorizado")
    cantidad: str | None = None
    dias_tratamiento: str | None = None
    descripcion: str | None = None


class ServiciosAutorizados(BaseModel):
    ubicacion_paciente: str | None = None
    grupo_servicio: str | None = None
    items: list[ServicioAutorizadoItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_location_vs_group(self) -> "ServiciosAutorizados":
        location_values = {"ambulatorio", "hospitalario", "urgencias", "domiciliario"}
        group_values = {
            "consulta externa",
            "hospitalización",
            "hospitalizacion",
            "cirugía",
            "cirugia",
            "apoyo diagnóstico",
            "apoyo diagnostico",
        }

        ubicacion = (self.ubicacion_paciente or "").strip()
        grupo = (self.grupo_servicio or "").strip()
        ubicacion_l = ubicacion.lower()
        grupo_l = grupo.lower()

        if not ubicacion and grupo_l in location_values:
            self.ubicacion_paciente = grupo
            self.grupo_servicio = None
        elif not grupo and ubicacion_l in group_values:
            self.grupo_servicio = ubicacion
            self.ubicacion_paciente = None

        return self


class AuthorizationExtraction(BaseModel):
    numero_autorizacion: str | None = None
    fecha_autorizacion: str | None = None
    prestador_autorizado: PrestadorAutorizado = Field(default_factory=PrestadorAutorizado)
    datos_paciente: DatosPaciente = Field(default_factory=DatosPaciente)
    servicios_autorizados: ServiciosAutorizados = Field(default_factory=ServiciosAutorizados)
    vigencia: str | None = None

    @field_validator("fecha_autorizacion", mode="before")
    @classmethod
    def normalize_fecha_autorizacion(cls, value: str | None) -> str | None:
        if value is None or not isinstance(value, str):
            return value
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(text, fmt).strftime("%Y/%m/%d")
            except ValueError:
                continue
        return text

    @field_validator("vigencia", mode="before")
    @classmethod
    def normalize_vigencia(cls, value: str | None) -> str | None:
        if value is None or not isinstance(value, str):
            return value
        text = " ".join(value.strip().split())
        upper = text.upper()
        if upper.endswith(" DIAS"):
            digits = "".join(ch for ch in upper if ch.isdigit())
            if digits:
                return f"{digits} dias"
        return text

    @field_validator("numero_autorizacion", mode="before")
    @classmethod
    def normalize_numero_autorizacion(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, int):
            value = str(value)
        if not isinstance(value, str):
            return None
        digits = "".join(ch for ch in value if ch.isdigit())
        if len(digits) != 14:
            return None
        return digits


class AuthorizationResponse(BaseModel):
    authorizations: list[AuthorizationExtraction] = Field(default_factory=list)

    @model_validator(mode="after")
    def filter_invalid_authorizations(self) -> "AuthorizationResponse":
        self.authorizations = [
            auth for auth in self.authorizations if auth.numero_autorizacion not in (None, "")
        ]
        return self
