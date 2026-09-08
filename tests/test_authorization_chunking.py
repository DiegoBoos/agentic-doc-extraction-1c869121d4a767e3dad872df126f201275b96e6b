from __future__ import annotations

from types import MethodType

import pytest

from app.schemas.authorization import AuthorizationExtraction, AuthorizationResponse
from app.services.authorization_chunking import build_authorization_chunks, build_page_texts
from app.services.openai_extractor import OpenAIExtractorService


def test_build_page_texts_groups_chunks_by_page() -> None:
    page_texts = build_page_texts(
        [
            {"page_number": 2, "content": "Servicio B"},
            {"page_number": 1, "content": "Autorización A"},
            {"page_number": 1, "content": "Paciente X"},
        ]
    )

    assert page_texts == ["Autorización A\nPaciente X", "Servicio B"]


def test_build_authorization_chunks_splits_large_document_by_pages() -> None:
    chunks = build_authorization_chunks(
        markdown="resumen corto",
        chunks=[
            {"page_number": 1, "content": "A" * 500},
            {"page_number": 2, "content": "B" * 500},
            {"page_number": 3, "content": "C" * 500},
        ],
        max_input_tokens=100,
        target_chunk_tokens=150,
        max_pages_per_chunk=1,
        overlap_pages=0,
    )

    assert len(chunks) == 3
    assert chunks[0].page_start == 1
    assert chunks[1].page_start == 2
    assert chunks[2].page_start == 3


@pytest.mark.asyncio
async def test_extract_authorization_aggregates_chunked_results() -> None:
    service = OpenAIExtractorService.__new__(OpenAIExtractorService)
    service.model = "fake-model"
    service.max_input_tokens = 100
    service.chunk_target_tokens = 150
    service.chunk_max_pages = 1
    service.chunk_overlap_pages = 0
    service.large_document_chunk_target_tokens = 80
    service.large_document_chunk_max_pages = 1
    service.large_document_chunk_overlap_pages = 0
    service.client = None

    calls: list[str] = []

    async def fake_extract_structured(self, *, system_prompt, user_content, response_model):
        calls.append(user_content)
        index = len(calls)
        return (
            AuthorizationResponse(
                authorizations=[
                    AuthorizationExtraction(numero_autorizacion=f"{index:014d}"),
                ]
            ),
            index * 10,
            index * 2,
        )

    service._extract_structured = MethodType(fake_extract_structured, service)

    result, tokens_input, tokens_output = await service.extract_authorization(
        markdown="documento corto",
        document_chunks=[
            {"page_number": 1, "content": "A" * 500},
            {"page_number": 2, "content": "B" * 500},
        ],
    )

    assert len(calls) == 2
    assert [auth.numero_autorizacion for auth in result.authorizations] == [
        "00000000000001",
        "00000000000002",
    ]
    assert tokens_input == 30
    assert tokens_output == 6


@pytest.mark.asyncio
async def test_extract_authorization_allows_aggressive_large_document_overrides() -> None:
    service = OpenAIExtractorService.__new__(OpenAIExtractorService)
    service.model = "fake-model"
    service.max_input_tokens = 10_000
    service.chunk_target_tokens = 10_000
    service.chunk_max_pages = 12
    service.chunk_overlap_pages = 1
    service.large_document_chunk_target_tokens = 150
    service.large_document_chunk_max_pages = 1
    service.large_document_chunk_overlap_pages = 0
    service.client = None

    calls: list[str] = []

    async def fake_extract_structured(self, *, system_prompt, user_content, response_model):
        calls.append(user_content)
        index = len(calls)
        return (
            AuthorizationResponse(
                authorizations=[
                    AuthorizationExtraction(numero_autorizacion=f"{index:014d}"),
                ]
            ),
            index,
            index,
        )

    service._extract_structured = MethodType(fake_extract_structured, service)

    result, tokens_input, tokens_output = await service.extract_authorization(
        markdown="documento grande",
        document_chunks=[
            {"page_number": 1, "content": "A" * 500},
            {"page_number": 2, "content": "B" * 500},
            {"page_number": 3, "content": "C" * 500},
        ],
        chunk_target_tokens=service.large_document_chunk_target_tokens,
        chunk_max_pages=service.large_document_chunk_max_pages,
        chunk_overlap_pages=service.large_document_chunk_overlap_pages,
        strategy_label="large-document",
    )

    assert len(calls) == 3
    assert [auth.numero_autorizacion for auth in result.authorizations] == [
        "00000000000001",
        "00000000000002",
        "00000000000003",
    ]
    assert tokens_input == 6
    assert tokens_output == 6
