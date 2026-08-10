from __future__ import annotations

import logging
from typing import TypeVar

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel

from app.core.config import Settings
from app.schemas.authorization import AuthorizationResponse
from app.schemas.patient import PatientExtractionResponse

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Eres un extractor experto de autorizaciones médicas en Colombia.
Recibirás markdown crudo de OCR/Azure Document Intelligence.

Instrucciones:
1. Identifica y extrae TODAS las autorizaciones médicas presentes en el documento.
2. Usa el campo 'numero_autorizacion' como ancla principal para separar y distinguir
   cada autorización.
   Puede aparecer con variantes como: "AUTORIZACION No.:", "AUTORIZACION No .:",
   "AUTORIZACIÓN No:", "AUTORIZACION N°"
   o diferencias de espacios/acentos; trata todas esas variantes como equivalentes.
3. El numero_autorizacion debe tener exactamente 14 dígitos. Si no cumple, pasa a la
   siguiente, no la incluyas.
4. Devuelve una lista de autorizaciones en el campo 'authorizations'.
5. En items, incluye cada servicio autorizado con su codigo_cups, cantidad,
   dias_tratamiento y descripcion.
""".strip()

PATIENT_SYSTEM_PROMPT = """
Eres un extractor experto de datos demográficos de pacientes en documentos clínicos colombianos.
Recibirás markdown crudo generado por OCR/Azure Document Intelligence de la primera
página del documento.

Instrucciones:
1. Extrae solo datos del paciente, no datos del prestador, médico, acompañante,
   acudiente o entidad salvo que el campo pida EPS/entidad.
2. Prioriza dirección y teléfonos. Busca variantes como direccion, dir., residencia,
   barrio, telefono, tel., celular, movil, contacto, casa, trabajo y laboral.
3. Conserva el texto tal como aparece cuando no haya una normalización evidente.
4. Si hay varios teléfonos, clasifícalos en casa, movil, trabajo u otros según la etiqueta cercana.
5. Incluye en otros_datos_demograficos cualquier dato demográfico relevante que aparezca
   y no tenga campo propio.
6. No inventes información. Usa null para campos no encontrados y listas vacías cuando no
   haya datos adicionales.
""".strip()

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
_MAX_RETRIES = 3
StructuredResponseT = TypeVar("StructuredResponseT", bound=BaseModel)


class LLMConnectionError(Exception):
    """Raised when the LLM service is unreachable or times out."""


def _build_openai_client(*, api_key: str, base_url: str | None = None) -> AsyncOpenAI:
    kwargs: dict = dict(
        api_key=api_key,
        timeout=_TIMEOUT,
        max_retries=_MAX_RETRIES,
    )
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


class OpenAIExtractorService:
    def __init__(self, settings: Settings) -> None:
        provider = settings.model_provider.lower()
        if provider != "openai":
            raise RuntimeError(f"Unsupported model_provider: '{provider}'. Only 'openai' is supported.")

        if not settings.openai_api_key:
            raise RuntimeError("OpenAI API key is not configured. Set OPENAI_API_KEY.")

        self.client = _build_openai_client(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.openai_model

        logger.info(
            "LLM extractor ready | provider=openai model=%s",
            self.model,
        )

    async def extract_authorization(self, markdown: str) -> tuple[AuthorizationResponse, int, int]:
        user_content = (
            "Extrae todas las autorizaciones médicas del siguiente markdown y responde "
            "en el esquema estructurado.\n\n"
            f"{markdown}"
        )
        return await self._extract_structured(
            system_prompt=SYSTEM_PROMPT,
            user_content=user_content,
            response_model=AuthorizationResponse,
        )

    async def extract_patient_data(
        self, markdown: str
    ) -> tuple[PatientExtractionResponse, int, int]:
        user_content = (
            "Extrae los datos demográficos del paciente desde el siguiente markdown de la "
            "primera página y responde en el esquema estructurado.\n\n"
            f"{markdown}"
        )
        return await self._extract_structured(
            system_prompt=PATIENT_SYSTEM_PROMPT,
            user_content=user_content,
            response_model=PatientExtractionResponse,
        )

    async def _extract_structured(
        self,
        *,
        system_prompt: str,
        user_content: str,
        response_model: type[StructuredResponseT],
    ) -> tuple[StructuredResponseT, int, int]:
        try:
            return await self._call_llm(
                self.client,
                self.model,
                system_prompt,
                user_content,
                response_model,
            )
        except (APIConnectionError, APITimeoutError) as exc:
            raise LLMConnectionError(
                f"LLM service unavailable ({type(exc).__name__}): {exc}"
            ) from exc

    async def _call_llm(
        self,
        client: AsyncOpenAI,
        model: str,
        system_prompt: str,
        user_content: str,
        response_model: type[StructuredResponseT],
    ) -> tuple[StructuredResponseT, int, int]:
        response = await client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            text_format=response_model,
        )

        if response.output_parsed is None:
            raise RuntimeError("LLM returned no structured output.")

        tokens_input = 0
        tokens_output = 0
        if hasattr(response, "usage"):
            usage = response.usage
            tokens_input = getattr(usage, "input_tokens", 0) or 0
            tokens_output = getattr(usage, "output_tokens", 0) or 0
            if not tokens_input and not tokens_output:
                total_tokens = getattr(usage, "total_tokens", 0) or 0
                tokens_input = total_tokens
                tokens_output = 0

        return response.output_parsed, tokens_input, tokens_output
