from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    upload_dir: Path = Path("data/uploads")
    parse_output_dir: Path = Path("data/parsed")
    max_upload_size_mb: int = 50
    processing_max_concurrent_documents: int = Field(
        default=2,
        validation_alias=AliasChoices(
            "PROCESSING_MAX_CONCURRENT_DOCUMENTS",
            "DOC_EXTRACTION_PROCESSING_MAX_CONCURRENT_DOCUMENTS",
        ),
    )
    processing_large_document_page_threshold: int = Field(
        default=50,
        validation_alias=AliasChoices(
            "PROCESSING_LARGE_DOCUMENT_PAGE_THRESHOLD",
            "DOC_EXTRACTION_PROCESSING_LARGE_DOCUMENT_PAGE_THRESHOLD",
        ),
    )
    openai_max_input_tokens: int = Field(
        default=45000,
        validation_alias=AliasChoices(
            "OPENAI_MAX_INPUT_TOKENS",
            "DOC_EXTRACTION_OPENAI_MAX_INPUT_TOKENS",
        ),
    )
    openai_chunk_target_tokens: int = Field(
        default=24000,
        validation_alias=AliasChoices(
            "OPENAI_CHUNK_TARGET_TOKENS",
            "DOC_EXTRACTION_OPENAI_CHUNK_TARGET_TOKENS",
        ),
    )
    openai_chunk_max_pages: int = Field(
        default=12,
        validation_alias=AliasChoices(
            "OPENAI_CHUNK_MAX_PAGES",
            "DOC_EXTRACTION_OPENAI_CHUNK_MAX_PAGES",
        ),
    )
    openai_chunk_overlap_pages: int = Field(
        default=1,
        validation_alias=AliasChoices(
            "OPENAI_CHUNK_OVERLAP_PAGES",
            "DOC_EXTRACTION_OPENAI_CHUNK_OVERLAP_PAGES",
        ),
    )
    allowed_upload_extensions: list[str] = [".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    allowed_upload_content_types: list[str] = [
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/tiff",
    ]

    model_provider: str = Field(
        default="openai",
        validation_alias=AliasChoices(
            "MODEL_PROVIDER",
            "DOC_EXTRACTION_MODEL_PROVIDER",
        ),
    )

    azure_document_intelligence_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
            "DOC_EXTRACTION_AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        ),
    )
    azure_document_intelligence_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_DOCUMENT_INTELLIGENCE_KEY",
            "DOC_EXTRACTION_AZURE_DOCUMENT_INTELLIGENCE_KEY",
        ),
    )
    azure_document_intelligence_model: str = "prebuilt-read"

    google_vision_credentials_file: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GOOGLE_VISION_CREDENTIALS_FILE",
            "DOC_EXTRACTION_GOOGLE_VISION_CREDENTIALS_FILE",
        ),
    )

    ocr_provider: str = Field(
        default="google_vision",
        validation_alias=AliasChoices(
            "OCR_PROVIDER",
            "DOC_EXTRACTION_OCR_PROVIDER",
        ),
    )

    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_API_KEY",
            "DOC_EXTRACTION_OPENAI_API_KEY",
        ),
    )
    openai_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_BASE_URL",
            "DOC_EXTRACTION_OPENAI_BASE_URL",
        ),
    )

    api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "API_KEY",
            "DOC_EXTRACTION_API_KEY",
        ),
    )

    openai_model: str = Field(
        default="gpt-5-nano",
        validation_alias=AliasChoices(
            "OPENAI_MODEL",
            "DOC_EXTRACTION_OPENAI_MODEL",
        ),
    )

    database_url: str | None = Field(
        default="postgres://postgres.your-tenant-id:odhtklmeursthgffjr5eocuctgakioub@192.168.5.133:6543/postgres",
        validation_alias=AliasChoices(
            "DATABASE_URL",
            "DOC_EXTRACTION_DATABASE_URL",
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DOC_EXTRACTION_",
        extra="ignore",
    )
