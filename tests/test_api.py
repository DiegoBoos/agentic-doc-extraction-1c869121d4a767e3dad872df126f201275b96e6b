from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter

from app.main import app
from app.schemas.authorization import AuthorizationExtraction, AuthorizationResponse
from app.schemas.patient import PatientData, PatientExtractionResponse
from app.services import document_parser as document_parser_module
from app.services.document_parser import AzureDocumentIntelligenceParser, ParsedDocumentResult


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_parse_endpoint_with_mocked_openai_structured_output() -> None:
    markdown = "AUTORIZACION No.: 1234567890"

    with TestClient(app) as client:
        client.app.state.parser.parse_document = lambda **_: ParsedDocumentResult(
            markdown=markdown,
            chunks=[{"text": "mock"}],
            json_output_path=str(Path("data/parsed/mock.json")),
            markdown_output_path=str(Path("data/parsed/mock.md")),
            model="prebuilt-read",
            provider="azure",
        )

        class _MockExtractor:
            async def extract_authorization(
                self, _markdown: str, _document_chunks=None
            ) -> tuple[AuthorizationResponse, int, int]:
                return (
                    AuthorizationResponse(
                        authorizations=[
                            AuthorizationExtraction(numero_autorizacion="12345678901234")
                        ]
                    ),
                    100,
                    50,
                )

        client.app.state.extractor = _MockExtractor()

        response = client.post(
            "/api/v1/parse",
            files={"file": ("invoice.pdf", BytesIO(b"fake pdf bytes"), "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json()["authorizations"][0]["numero_autorizacion"] == "12345678901234"


def test_parse_endpoint_filters_null_authorization_number() -> None:
    markdown = "AUTORIZACION No.:"

    with TestClient(app) as client:
        client.app.state.parser.parse_document = lambda **_: ParsedDocumentResult(
            markdown=markdown,
            chunks=[{"text": "mock"}],
            json_output_path=str(Path("data/parsed/mock.json")),
            markdown_output_path=str(Path("data/parsed/mock.md")),
            model="prebuilt-read",
            provider="azure",
        )

        class _MockExtractor:
            async def extract_authorization(
                self, _markdown: str, _document_chunks=None
            ) -> tuple[AuthorizationResponse, int, int]:
                return (
                    AuthorizationResponse(authorizations=[AuthorizationExtraction()]),
                    100,
                    50,
                )

        client.app.state.extractor = _MockExtractor()

        response = client.post(
            "/api/v1/parse",
            files={"file": ("invoice.pdf", BytesIO(b"fake pdf bytes"), "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json() == {"authorizations": []}


def test_extract_patient_endpoint_with_mocked_structured_output(tmp_path: Path) -> None:
    markdown = "Paciente: Ana Gomez\nDireccion: Calle 1\nCelular: 3001234567"
    raw_json_path = tmp_path / "patient.azure_parse_output.json"
    raw_json_path.write_text('{"pages": [{"pageNumber": 1}]}', encoding="utf-8")
    calls = {}

    with TestClient(app) as client:

        def _parse_first_page(**kwargs) -> ParsedDocumentResult:
            calls.update(kwargs)
            return ParsedDocumentResult(
                markdown=markdown,
                chunks=[{"text": "mock"}],
                json_output_path=str(raw_json_path),
                markdown_output_path=str(tmp_path / "patient.md"),
                model="prebuilt-read",
                provider="azure",
            )

        client.app.state.parser.parse_first_page = _parse_first_page

        class _MockExtractor:
            async def extract_patient_data(
                self, _markdown: str
            ) -> tuple[PatientExtractionResponse, int, int]:
                return (
                    PatientExtractionResponse(
                        patient=PatientData(
                            nombre_completo="Ana Gomez",
                            direccion="Calle 1",
                            telefonos={"movil": "3001234567"},
                        )
                    ),
                    42,
                    12,
                )

        client.app.state.extractor = _MockExtractor()

        response = client.post(
            "/api/v1/extract-patient",
            files={"file": ("historia.pdf", BytesIO(b"fake pdf bytes"), "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "historia.pdf"
    assert data["provider"] == "azure"
    assert data["extracted_pages"] == 1
    assert data["tokens_input"] == 42
    assert data["patient"]["nombre_completo"] == "Ana Gomez"
    assert data["patient"]["direccion"] == "Calle 1"
    assert data["patient"]["telefonos"]["movil"] == "3001234567"
    assert "fhir" in data
    assert data["fhir"]["resourceType"] == "Patient"
    assert calls["document_id"] == data["document_id"]


def test_azure_parser_first_page_limits_analyze_request_to_page_one(
    monkeypatch, tmp_path: Path
) -> None:
    captured = {}

    class _AnalyzeDocumentRequest:
        def __init__(self, *, bytes_source: bytes) -> None:
            self.bytes_source = bytes_source

    class _Result:
        content = "Paciente: Ana Gomez"
        pages = [SimpleNamespace(page_number=1, lines=[])]

        def as_dict(self) -> dict:
            return {"content": self.content, "pages": [{"pageNumber": 1}]}

    class _Poller:
        def result(self) -> _Result:
            return _Result()

    class _Client:
        def begin_analyze_document(self, model: str, request, **kwargs) -> _Poller:
            captured["model"] = model
            captured["request"] = request
            captured["kwargs"] = kwargs
            return _Poller()

    monkeypatch.setattr(
        document_parser_module,
        "AnalyzeDocumentRequest",
        _AnalyzeDocumentRequest,
    )

    parser = AzureDocumentIntelligenceParser.__new__(AzureDocumentIntelligenceParser)
    parser.output_root = tmp_path
    parser.model = "prebuilt-read"
    parser.client = _Client()

    document_path = tmp_path / "historia.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    with document_path.open("wb") as handle:
        writer.write(handle)

    result = parser.parse_first_page(document_path=document_path, document_id="doc-1")

    assert captured["model"] == "prebuilt-read"
    assert len(PdfReader(str(document_path)).pages) == 2
    assert len(PdfReader(BytesIO(captured["request"].bytes_source)).pages) == 1
    assert captured["kwargs"]["pages"] == "1"
    assert captured["kwargs"]["output_content_format"] == "markdown"
    assert result.provider == "azure"
    assert result.markdown == "Paciente: Ana Gomez"
