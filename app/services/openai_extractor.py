from __future__ import annotations

import logging
import time
from typing import Any, TypeVar

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, BadRequestError
from pydantic import BaseModel

from app.core.config import Settings
from app.schemas.authorization import AuthorizationResponse
from app.schemas.patient import PatientExtractionResponse
from app.services.authorization_chunking import TextChunk, build_authorization_chunks

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
   Si el documento es una SOLICITUD y no trae numero_autorizacion final de 14 dígitos,
   usa 'numero_solicitud' como ancla secundaria para separar cada registro.
3. El numero_autorizacion debe tener exactamente 14 dígitos. Si no cumple, deja
   numero_autorizacion en null.
4. Si no existe numero_autorizacion válido pero sí aparece "No. Solicitud" o variantes
   equivalentes, llena 'numero_solicitud' con ese consecutivo y conserva el registro.
   No inventes un numero_autorizacion de 14 dígitos.
5. Devuelve una lista de autorizaciones en el campo 'authorizations'.
6. En items, incluye cada servicio autorizado con su codigo_cups, cantidad,
   dias_tratamiento y descripcion.
7. No inventes información. Si un campo no es claro o no aparece, usa null.
8. fecha_autorizacion:
   - extrae la fecha de expedición/emisión/autorización del documento,
   - normalízala como YYYY/MM/DD si es posible,
   - no uses fechas de vigencia, de cita o de vencimiento como fecha_autorizacion.
9. vigencia:
   - prioriza el TEXTO LITERAL de duración o vigencia, por ejemplo "120 dias",
   - si el documento dice una duración y también una fecha futura, conserva la duración,
   - no reemplaces vigencia por una fecha de vencimiento salvo que no exista duración explícita.
10. servicios_autorizados.ubicacion_paciente:
   - representa la ubicación/modalidad del paciente, por ejemplo: Ambulatorio,
     Hospitalario, Urgencias, Domiciliario,
   - no pongas aquí el grupo del servicio.
11. servicios_autorizados.grupo_servicio:
   - representa la categoría del servicio, por ejemplo: Consulta externa,
     Hospitalización, Cirugía, Apoyo diagnóstico,
   - no pongas aquí valores de ubicación del paciente como Ambulatorio.
12. Si ves ambos valores, usa este mapeo:
   - Ambulatorio/Hospitalario/Urgencias/Domiciliario -> ubicacion_paciente
   - Consulta externa/Hospitalización/Cirugía/etc. -> grupo_servicio
13. Conserva el texto clínico/administrativo fiel al documento; no resumas ni parafrasees
   los nombres de prestador o descripciones CUPS.
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


class ContextWindowExceededError(ValueError):
    """Raised when the markdown still exceeds the model context window after chunking."""


def _build_openai_client(*, api_key: str, base_url: str | None = None) -> AsyncOpenAI:
    kwargs: dict[str, Any] = dict(
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
            raise RuntimeError(
                f"Unsupported model_provider: '{provider}'. Only 'openai' is supported."
            )

        if not settings.openai_api_key:
            raise RuntimeError("OpenAI API key is not configured. Set OPENAI_API_KEY.")

        self.client = _build_openai_client(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.openai_model
        self.max_input_tokens = settings.openai_max_input_tokens
        self.chunk_target_tokens = settings.openai_chunk_target_tokens
        self.chunk_max_pages = settings.openai_chunk_max_pages
        self.chunk_overlap_pages = settings.openai_chunk_overlap_pages

        logger.info(
            "LLM extractor ready | provider=openai model=%s max_input_tokens=%s "
            "chunk_target_tokens=%s chunk_max_pages=%s",
            self.model,
            self.max_input_tokens,
            self.chunk_target_tokens,
            self.chunk_max_pages,
        )

    async def extract_authorization(
        self,
        markdown: str,
        document_chunks: list[dict[str, Any]] | None = None,
    ) -> tuple[AuthorizationResponse, int, int]:
        started_at = time.perf_counter()
        text_chunks = build_authorization_chunks(
            markdown=markdown,
            chunks=document_chunks,
            max_input_tokens=self.max_input_tokens,
            target_chunk_tokens=self.chunk_target_tokens,
            max_pages_per_chunk=self.chunk_max_pages,
            overlap_pages=self.chunk_overlap_pages,
        )

        if len(text_chunks) == 1:
            logger.info(
                "Starting single-pass authorization extraction | model=%s markdown_chars=%s estimated_tokens=%s",
                self.model,
                len(markdown),
                text_chunks[0].estimated_tokens,
            )
            try:
                result, tokens_input, tokens_output = await self._extract_authorization_from_text(
                    text_chunks[0],
                    is_fragment=False,
                )
                logger.info(
                    "Completed single-pass authorization extraction | model=%s authorizations=%s tokens_in=%s tokens_out=%s elapsed_ms=%.2f",
                    self.model,
                    len(result.authorizations),
                    tokens_input,
                    tokens_output,
                    (time.perf_counter() - started_at) * 1000,
                )
                return result, tokens_input, tokens_output
            except ContextWindowExceededError:
                logger.warning(
                    "Context window exceeded on single-pass extraction; "
                    "retrying in forced chunk mode"
                )
                text_chunks = build_authorization_chunks(
                    markdown=markdown,
                    chunks=None,
                    max_input_tokens=1,
                    target_chunk_tokens=max(4000, self.chunk_target_tokens // 2),
                    max_pages_per_chunk=1,
                    overlap_pages=0,
                )

        merged: list = []
        total_input_tokens = 0
        total_output_tokens = 0

        logger.info(
            "Starting chunked authorization extraction | chunks=%s model=%s markdown_chars=%s",
            len(text_chunks),
            self.model,
            len(markdown),
        )

        for index, chunk in enumerate(text_chunks, start=1):
            chunk_started_at = time.perf_counter()
            logger.info(
                "Processing authorization chunk %s/%s | scope=%s estimated_tokens=%s",
                index,
                len(text_chunks),
                chunk.label,
                chunk.estimated_tokens,
            )
            result, chunk_input_tokens, chunk_output_tokens = (
                await self._extract_authorization_from_text(
                    chunk,
                    is_fragment=True,
                )
            )
            logger.info(
                "Completed authorization chunk %s/%s | scope=%s authorizations=%s tokens_in=%s tokens_out=%s elapsed_ms=%.2f",
                index,
                len(text_chunks),
                chunk.label,
                len(result.authorizations),
                chunk_input_tokens,
                chunk_output_tokens,
                (time.perf_counter() - chunk_started_at) * 1000,
            )
            merged.extend(result.authorizations)
            total_input_tokens += chunk_input_tokens
            total_output_tokens += chunk_output_tokens

        merged_response = AuthorizationResponse(authorizations=merged)
        logger.info(
            "Completed chunked authorization extraction | chunks=%s authorizations=%s tokens_in=%s tokens_out=%s elapsed_ms=%.2f",
            len(text_chunks),
            len(merged_response.authorizations),
            total_input_tokens,
            total_output_tokens,
            (time.perf_counter() - started_at) * 1000,
        )
        return (
            merged_response,
            total_input_tokens,
            total_output_tokens,
        )

    async def extract_patient_data(
        self, markdown: str
    ) -> tuple[PatientExtractionResponse, int, int]:
        started_at = time.perf_counter()
        logger.info(
            "Starting patient extraction | model=%s markdown_chars=%s",
            self.model,
            len(markdown),
        )
        user_content = (
            "Extrae los datos demográficos del paciente desde el siguiente markdown de la "
            "primera página y responde en el esquema estructurado.\n\n"
            f"{markdown}"
        )
        result, tokens_input, tokens_output = await self._extract_structured(
            system_prompt=PATIENT_SYSTEM_PROMPT,
            user_content=user_content,
            response_model=PatientExtractionResponse,
        )
        logger.info(
            "Completed patient extraction | model=%s tokens_in=%s tokens_out=%s elapsed_ms=%.2f",
            self.model,
            tokens_input,
            tokens_output,
            (time.perf_counter() - started_at) * 1000,
        )
        return result, tokens_input, tokens_output

    async def _extract_authorization_from_text(
        self,
        chunk: TextChunk,
        *,
        is_fragment: bool,
    ) -> tuple[AuthorizationResponse, int, int]:
        if is_fragment:
            user_content = (
                "Extrae todas las autorizaciones médicas del siguiente fragmento del documento "
                "y responde en el esquema estructurado.\n"
                "Recuerda especialmente: vigencia no es fecha de vencimiento si existe una "
                "duración explícita; ubicacion_paciente y grupo_servicio no deben mezclarse.\n\n"
                f"{chunk.text}"
            )
        else:
            user_content = (
                "Extrae todas las autorizaciones médicas del siguiente markdown y responde "
                "en el esquema estructurado.\n"
                "Recuerda especialmente: vigencia no es fecha de vencimiento si existe una "
                "duración explícita; ubicacion_paciente y grupo_servicio no deben mezclarse.\n\n"
                f"{chunk.text}"
            )

        return await self._extract_structured(
            system_prompt=SYSTEM_PROMPT,
            user_content=user_content,
            response_model=AuthorizationResponse,
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
        started_at = time.perf_counter()
        logger.info(
            "Calling LLM | model=%s response_model=%s input_chars=%s",
            model,
            response_model.__name__,
            len(user_content),
        )
        try:
            response = await client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                text_format=response_model,
            )
        except BadRequestError as exc:
            if self._is_context_window_error(exc):
                raise ContextWindowExceededError(
                    "El documento excede la ventana de contexto del modelo y "
                    "requiere fragmentación adicional."
                ) from exc
            raise

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

        logger.info(
            "LLM response parsed | model=%s response_model=%s tokens_in=%s tokens_out=%s elapsed_ms=%.2f",
            model,
            response_model.__name__,
            tokens_input,
            tokens_output,
            (time.perf_counter() - started_at) * 1000,
        )
        return response.output_parsed, tokens_input, tokens_output

    @staticmethod
    def _is_context_window_error(exc: BadRequestError) -> bool:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error") or {}
            if error.get("code") == "context_length_exceeded":
                return True
            message = str(error.get("message") or "")
            return "context window" in message.lower()
        return "context window" in str(exc).lower()
