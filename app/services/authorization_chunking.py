from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_DEFAULT_TOKENS_PER_CHAR = 0.25


@dataclass(slots=True)
class TextChunk:
    text: str
    estimated_tokens: int
    page_start: int | None = None
    page_end: int | None = None

    @property
    def label(self) -> str:
        if self.page_start is None:
            return "documento"
        if self.page_start == self.page_end:
            return f"página {self.page_start}"
        return f"páginas {self.page_start}-{self.page_end}"


def estimate_tokens(text: str) -> int:
    normalized = (text or "").strip()
    if not normalized:
        return 0
    return max(1, int(len(normalized) * _DEFAULT_TOKENS_PER_CHAR))


def build_page_texts(chunks: list[dict[str, Any]] | None) -> list[str]:
    if not chunks:
        return []

    pages: dict[int, list[str]] = {}
    for chunk in chunks:
        page_number = chunk.get("page_number")
        content = (chunk.get("content") or "").strip()
        if page_number in (None, "") or not content:
            continue
        try:
            page_index = int(page_number)
        except (TypeError, ValueError):
            continue
        pages.setdefault(page_index, []).append(content)

    if not pages:
        return []

    return ["\n".join(filter(None, pages[page])).strip() for page in sorted(pages)]


def build_authorization_chunks(
    *,
    markdown: str,
    chunks: list[dict[str, Any]] | None,
    max_input_tokens: int,
    target_chunk_tokens: int,
    max_pages_per_chunk: int,
    overlap_pages: int,
) -> list[TextChunk]:
    page_texts = build_page_texts(chunks)
    if not page_texts:
        if estimate_tokens(markdown) <= max_input_tokens:
            return [TextChunk(text=markdown, estimated_tokens=estimate_tokens(markdown))]
        return _split_text_chunk(markdown, target_chunk_tokens)

    total_estimated_tokens = sum(estimate_tokens(page) for page in page_texts)
    if len(page_texts) <= max_pages_per_chunk and total_estimated_tokens <= max_input_tokens:
        return [
            TextChunk(
                text=markdown,
                estimated_tokens=max(total_estimated_tokens, estimate_tokens(markdown)),
            )
        ]

    return _chunk_by_pages(
        page_texts=page_texts,
        target_chunk_tokens=target_chunk_tokens,
        max_pages_per_chunk=max_pages_per_chunk,
        overlap_pages=overlap_pages,
    )


def _chunk_by_pages(
    *,
    page_texts: list[str],
    target_chunk_tokens: int,
    max_pages_per_chunk: int,
    overlap_pages: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    index = 0
    safe_overlap = max(0, overlap_pages)
    safe_max_pages = max(1, max_pages_per_chunk)
    safe_target_tokens = max(1000, target_chunk_tokens)

    while index < len(page_texts):
        start_index = index
        page_parts: list[str] = []
        estimated_tokens = 0

        while index < len(page_texts) and (index - start_index) < safe_max_pages:
            page_number = index + 1
            candidate = f"## Página {page_number}\n{page_texts[index].strip()}"
            candidate_tokens = estimate_tokens(candidate)

            if page_parts and estimated_tokens + candidate_tokens > safe_target_tokens:
                break

            if candidate_tokens > safe_target_tokens and not page_parts:
                for sub_chunk in _split_text_chunk(
                    candidate,
                    safe_target_tokens,
                    page_start=page_number,
                    page_end=page_number,
                ):
                    chunks.append(sub_chunk)
                index += 1
                break

            page_parts.append(candidate)
            estimated_tokens += candidate_tokens
            index += 1

            if estimated_tokens >= safe_target_tokens:
                break
        else:
            # Completed naturally without forced split.
            pass

        if page_parts:
            chunks.append(
                TextChunk(
                    text="\n\n".join(page_parts),
                    estimated_tokens=estimated_tokens,
                    page_start=start_index + 1,
                    page_end=start_index + len(page_parts),
                )
            )

        if index >= len(page_texts):
            break

        if page_parts:
            index = max(start_index + 1, index - safe_overlap)

    return chunks


def _split_text_chunk(
    text: str,
    target_chunk_tokens: int,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[TextChunk]:
    safe_target_tokens = max(1000, target_chunk_tokens)
    max_chars = safe_target_tokens * 4
    normalized = (text or "").strip()
    if not normalized:
        return []

    if len(normalized) <= max_chars:
        return [
            TextChunk(
                text=normalized,
                estimated_tokens=estimate_tokens(normalized),
                page_start=page_start,
                page_end=page_end,
            )
        ]

    paragraphs = [segment.strip() for segment in normalized.split("\n\n") if segment.strip()]
    if not paragraphs:
        paragraphs = [normalized]

    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_chars = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        if current_parts and current_chars + paragraph_len > max_chars:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    estimated_tokens=estimate_tokens(chunk_text),
                    page_start=page_start,
                    page_end=page_end,
                )
            )
            current_parts = []
            current_chars = 0

        if paragraph_len > max_chars:
            for start in range(0, paragraph_len, max_chars):
                partial = paragraph[start : start + max_chars].strip()
                if partial:
                    chunks.append(
                        TextChunk(
                            text=partial,
                            estimated_tokens=estimate_tokens(partial),
                            page_start=page_start,
                            page_end=page_end,
                        )
                    )
            continue

        current_parts.append(paragraph)
        current_chars += paragraph_len

    if current_parts:
        chunk_text = "\n\n".join(current_parts)
        chunks.append(
            TextChunk(
                text=chunk_text,
                estimated_tokens=estimate_tokens(chunk_text),
                page_start=page_start,
                page_end=page_end,
            )
        )

    return chunks
