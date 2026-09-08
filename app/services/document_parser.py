from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    from azure.core.credentials import AzureKeyCredential
except ImportError:  # pragma: no cover - dependency optional at import time
    DocumentIntelligenceClient = None  # type: ignore[assignment]
    AnalyzeDocumentRequest = None  # type: ignore[assignment]
    AzureKeyCredential = None  # type: ignore[assignment]

try:
    from google.cloud import vision as google_vision
    from google.protobuf.json_format import MessageToDict
except ImportError:  # pragma: no cover - dependency optional at import time
    google_vision = None  # type: ignore[assignment]
    MessageToDict = None  # type: ignore[assignment]

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:  # pragma: no cover - dependency optional at import time
    PdfReader = None  # type: ignore[assignment]
    PdfWriter = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

ParserProvider = Literal["azure", "google_vision"]


@dataclass(slots=True)
class ParsedDocumentResult:
    markdown: str
    chunks: list[dict[str, Any]]
    json_output_path: str
    markdown_output_path: str
    model: str
    provider: ParserProvider = "google_vision"


@dataclass(slots=True)
class ParsedTablesResult:
    """Row-major table cells plus the surrounding non-table text (titles,
    timestamps, etc). Table structure is extracted via a layout model rather
    than a read model, since read-model reading order can interleave columns
    of a multi-column key-value grid unpredictably.
    """

    content: str
    tables: list[list[list[str]]]


class BaseDocumentParser:
    provider: ParserProvider

    def parse_document(self, *, document_path: Path, document_id: str) -> ParsedDocumentResult:
        raise NotImplementedError

    def parse_first_page(self, *, document_path: Path, document_id: str) -> ParsedDocumentResult:
        raise NotImplementedError

    def parse_tables(self, *, document_path: Path, document_id: str) -> ParsedTablesResult:
        raise NotImplementedError


class AzureDocumentIntelligenceParser(BaseDocumentParser):
    provider: ParserProvider = "azure"

    def __init__(
        self,
        *,
        output_root: Path,
        endpoint: str,
        key: str,
        model: str = "prebuilt-read",
    ) -> None:
        if (
            DocumentIntelligenceClient is None
            or AzureKeyCredential is None
            or AnalyzeDocumentRequest is None
        ):
            raise RuntimeError(
                "Azure Document Intelligence dependencies are not installed. "
                "Install azure-ai-documentintelligence and azure-core."
            )

        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint
        self.key = key
        self.model = model
        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key),
        )

    def parse_document(self, *, document_path: Path, document_id: str) -> ParsedDocumentResult:
        return self._parse_document(document_path=document_path, document_id=document_id)

    def parse_first_page(self, *, document_path: Path, document_id: str) -> ParsedDocumentResult:
        return self._parse_document(
            document_path=document_path,
            document_id=document_id,
            pages="1",
            first_page_only=True,
            output_name_suffix=".first_page",
        )

    def _parse_document(
        self,
        *,
        document_path: Path,
        document_id: str,
        pages: str | None = None,
        first_page_only: bool = False,
        output_name_suffix: str = "",
    ) -> ParsedDocumentResult:
        started_at = time.perf_counter()
        output_dir = self.output_root / document_id
        output_dir.mkdir(parents=True, exist_ok=True)

        file_bytes = self._read_document_bytes(
            document_path=document_path,
            first_page_only=first_page_only,
        )
        analyze_kwargs: dict[str, Any] = {"output_content_format": "markdown"}
        if pages:
            analyze_kwargs["pages"] = pages

        logger.info(
            "Starting Azure OCR | document_id=%s file=%s bytes=%s pages=%s first_page_only=%s model=%s",
            document_id,
            document_path.name,
            len(file_bytes),
            pages or "all",
            first_page_only,
            self.model,
        )
        poller = self.client.begin_analyze_document(
            self.model,
            AnalyzeDocumentRequest(bytes_source=file_bytes),
            **analyze_kwargs,
        )
        result = poller.result()

        markdown = getattr(result, "content", "") or ""
        output_stem = f"{document_path.stem}{output_name_suffix}"
        markdown_output_path = output_dir / f"{output_stem}.md"
        markdown_output_path.write_text(markdown, encoding="utf-8")

        raw_json_path = output_dir / f"{output_stem}.azure_parse_output.json"
        raw_payload = self._to_jsonable(result)
        raw_json_path.write_text(
            json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        chunks = self._extract_chunks(result)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "Completed Azure OCR | document_id=%s file=%s markdown_chars=%s chunks=%s elapsed_ms=%.2f",
            document_id,
            document_path.name,
            len(markdown),
            len(chunks),
            elapsed_ms,
        )

        return ParsedDocumentResult(
            markdown=markdown,
            chunks=chunks,
            json_output_path=str(raw_json_path),
            markdown_output_path=str(markdown_output_path),
            model=self.model,
            provider=self.provider,
        )

    def parse_tables(self, *, document_path: Path, document_id: str) -> ParsedTablesResult:
        started_at = time.perf_counter()
        file_bytes = document_path.read_bytes()

        logger.info(
            "Starting Azure layout table extraction | document_id=%s file=%s bytes=%s",
            document_id,
            document_path.name,
            len(file_bytes),
        )
        poller = self.client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=file_bytes),
        )
        result = poller.result()

        tables: list[list[list[str]]] = []
        for table in result.tables or []:
            grid = [["" for _ in range(table.column_count)] for _ in range(table.row_count)]
            for cell in table.cells:
                grid[cell.row_index][cell.column_index] = cell.content
            tables.append(grid)

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "Completed Azure layout table extraction | document_id=%s file=%s "
            "tables=%s elapsed_ms=%.2f",
            document_id,
            document_path.name,
            len(tables),
            elapsed_ms,
        )

        return ParsedTablesResult(content=getattr(result, "content", "") or "", tables=tables)

    def _extract_chunks(self, result: Any) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for page in getattr(result, "pages", []) or []:
            page_number = getattr(page, "page_number", None)
            for line in getattr(page, "lines", []) or []:
                polygon = getattr(getattr(line, "polygon", None), "__iter__", None)
                polygon_points = list(line.polygon) if polygon else None
                chunks.append(
                    {
                        "type": "line",
                        "page_number": page_number,
                        "content": getattr(line, "content", None),
                        "polygon": polygon_points,
                    }
                )

        if not chunks and getattr(result, "content", None):
            chunks.append({"type": "document", "content": result.content})

        return chunks

    def _read_document_bytes(self, *, document_path: Path, first_page_only: bool) -> bytes:
        if not first_page_only or document_path.suffix.lower() != ".pdf":
            return document_path.read_bytes()

        if PdfReader is None or PdfWriter is None:
            raise RuntimeError("pypdf is required to extract the first PDF page.")

        reader = PdfReader(str(document_path))
        if not reader.pages:
            raise ValueError("PDF has no pages to parse.")

        writer = PdfWriter()
        writer.add_page(reader.pages[0])

        buffer = BytesIO()
        writer.write(buffer)
        return buffer.getvalue()

    def _to_jsonable(self, result: Any) -> dict[str, Any]:
        if hasattr(result, "as_dict"):
            return result.as_dict()
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return {"content": getattr(result, "content", "")}


_PDF_MIME = "application/pdf"


class GoogleCloudVisionParser(BaseDocumentParser):
    provider: ParserProvider = "google_vision"

    def __init__(
        self,
        *,
        output_root: Path,
        credentials_file: str | None = None,
    ) -> None:
        if google_vision is None:
            raise RuntimeError(
                "Google Cloud Vision dependencies are not installed. Install google-cloud-vision."
            )
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

        if credentials_file:
            self.client = google_vision.ImageAnnotatorClient.from_service_account_file(
                credentials_file
            )
        else:
            # Falls back to GOOGLE_APPLICATION_CREDENTIALS env var or ADC
            self.client = google_vision.ImageAnnotatorClient()

    def parse_document(self, *, document_path: Path, document_id: str) -> ParsedDocumentResult:
        started_at = time.perf_counter()
        output_dir = self.output_root / document_id
        output_dir.mkdir(parents=True, exist_ok=True)

        file_bytes = document_path.read_bytes()
        suffix = document_path.suffix.lower()

        logger.info(
            "Starting Google Vision OCR | document_id=%s file=%s bytes=%s kind=%s",
            document_id,
            document_path.name,
            len(file_bytes),
            suffix or "unknown",
        )

        if suffix == ".pdf":
            full_text, chunks, raw_payload = self._process_pdf(file_bytes)
        else:
            full_text, chunks, raw_payload = self._process_image(file_bytes)

        markdown_output_path = output_dir / f"{document_path.stem}.md"
        markdown_output_path.write_text(full_text, encoding="utf-8")

        raw_json_path = output_dir / f"{document_path.stem}.google_vision_output.json"
        raw_json_path.write_text(
            json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "Completed Google Vision OCR | document_id=%s file=%s markdown_chars=%s chunks=%s elapsed_ms=%.2f",
            document_id,
            document_path.name,
            len(full_text),
            len(chunks),
            elapsed_ms,
        )

        return ParsedDocumentResult(
            markdown=full_text,
            chunks=chunks,
            json_output_path=str(raw_json_path),
            markdown_output_path=str(markdown_output_path),
            model="document-text-detection",
            provider=self.provider,
        )

    def _process_image(self, file_bytes: bytes) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        image = google_vision.Image(content=file_bytes)
        response = self.client.document_text_detection(image=image)
        full_text = getattr(response.full_text_annotation, "text", "") or ""
        chunks = self._extract_chunks(response.full_text_annotation, page_number=1)
        raw_payload = MessageToDict(response._pb)  # type: ignore[union-attr]
        return full_text, chunks, raw_payload

    def _process_pdf(self, file_bytes: bytes) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        input_config = google_vision.InputConfig(content=file_bytes, mime_type=_PDF_MIME)
        feature = google_vision.Feature(type_=google_vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
        request = google_vision.AnnotateFileRequest(
            input_config=input_config,
            features=[feature],
        )
        batch_response = self.client.batch_annotate_files(requests=[request])

        full_text_parts: list[str] = []
        chunks: list[dict[str, Any]] = []
        raw_responses: list[dict[str, Any]] = []

        for file_response in batch_response.responses:
            for page_num, page_response in enumerate(file_response.responses, start=1):
                page_text = getattr(page_response.full_text_annotation, "text", "") or ""
                full_text_parts.append(page_text)
                chunks.extend(
                    self._extract_chunks(page_response.full_text_annotation, page_number=page_num)
                )
                raw_responses.append(MessageToDict(page_response._pb))  # type: ignore[union-attr]

        return "\n\n".join(full_text_parts), chunks, {"responses": raw_responses}

    def _extract_chunks(self, text_annotation: Any, *, page_number: int) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for page in getattr(text_annotation, "pages", []) or []:
            for block in getattr(page, "blocks", []) or []:
                block_text_parts: list[str] = []
                for paragraph in getattr(block, "paragraphs", []) or []:
                    words = getattr(paragraph, "words", []) or []
                    para_text = " ".join(
                        "".join(
                            getattr(symbol, "text", "") for symbol in getattr(word, "symbols", [])
                        )
                        for word in words
                    )
                    block_text_parts.append(para_text)

                bounding_box = getattr(block, "bounding_box", None)
                vertices = (
                    [{"x": v.x, "y": v.y} for v in getattr(bounding_box, "vertices", [])]
                    if bounding_box
                    else None
                )
                chunks.append(
                    {
                        "type": "block",
                        "page_number": page_number,
                        "content": "\n".join(block_text_parts),
                        "bounding_box": vertices,
                    }
                )

        if not chunks and getattr(text_annotation, "text", None):
            chunks.append({"type": "document", "content": text_annotation.text})

        return chunks


class DocumentParserRouter:
    def __init__(
        self, *, default_provider: ParserProvider, parsers: dict[ParserProvider, BaseDocumentParser]
    ) -> None:
        self.default_provider = default_provider
        self.parsers = parsers

    def parse_document(
        self,
        *,
        document_path: Path,
        document_id: str,
        provider: ParserProvider | None = None,
    ) -> ParsedDocumentResult:
        selected_provider = provider or self.default_provider
        logger.info(
            "Routing document OCR | document_id=%s provider=%s file=%s",
            document_id,
            selected_provider,
            document_path.name,
        )
        parser = self._select_parser(selected_provider)
        return parser.parse_document(document_path=document_path, document_id=document_id)

    def parse_first_page(
        self,
        *,
        document_path: Path,
        document_id: str,
    ) -> ParsedDocumentResult:
        logger.info(
            "Routing first-page OCR | document_id=%s provider=azure file=%s",
            document_id,
            document_path.name,
        )
        parser = self._select_parser("azure")
        return parser.parse_first_page(document_path=document_path, document_id=document_id)

    def parse_tables(
        self,
        *,
        document_path: Path,
        document_id: str,
    ) -> ParsedTablesResult:
        logger.info(
            "Routing table extraction | document_id=%s provider=azure file=%s",
            document_id,
            document_path.name,
        )
        parser = self._select_parser("azure")
        return parser.parse_tables(document_path=document_path, document_id=document_id)

    def _select_parser(self, selected: ParserProvider) -> BaseDocumentParser:
        parser = self.parsers.get(selected)
        if parser is None:
            available = ", ".join(sorted(self.parsers.keys()))
            msg = f"Parser provider '{selected}' is not available or not configured."
            if available:
                msg += f" Available providers: {available}"
            else:
                msg += " No providers are currently configured."
            raise ValueError(msg)
        return parser
