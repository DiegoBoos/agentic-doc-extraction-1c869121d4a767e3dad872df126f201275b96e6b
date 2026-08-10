from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    upload_dir: Path = Path("data/uploads")
    parse_output_dir: Path = Path("data/parsed")
    max_upload_size_mb: int = 50
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
