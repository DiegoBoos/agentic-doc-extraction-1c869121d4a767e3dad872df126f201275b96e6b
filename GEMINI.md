# GEMINI.md

This document provides a comprehensive overview of the `agentic-doc-extraction` project, intended to be used as a quick reference for developers and as context for AI-assisted development.

## Project Overview

`agentic-doc-extraction` is a FastAPI-based backend service designed for parsing and extracting structured data from documents, with a specific focus on medical authorizations.

The application receives a document (e.g., a PDF or image file) through a REST API, processes it through an OCR provider, and then uses a large language model (LLM) to extract predefined fields.

### Architecture

The application is structured in a modular way, with clear separation of concerns:

*   **API Layer (`app/api`):** Defines the API endpoints. The main endpoint is `POST /api/v1/parse`.
*   **Configuration (`app/core/config.py`):** Uses `pydantic-settings` to manage configuration from environment variables and `.env` files.
*   **Services (`app/services`):** Contains the core business logic:
    *   `document_parser`: A router for different OCR/parsing providers (Azure Document Intelligence and Google Cloud Vision are supported).
    *   `openai_extractor`: Uses an LLM (like OpenAI's models) to extract structured data from the parsed text.
    *   `file_ingest`: Handles file uploads, validation, and storage.
*   **Dependencies (`app/dependencies`):** Manages dependencies for the FastAPI application, such as getting services and handling authentication.
*   **Database (`app/db`):** Contains database-related logic, such as saving billing metadata.

### Technologies

*   **Backend Framework:** FastAPI
*   **Dependency Management:** uv
*   **OCR/Parsing:** Azure Document Intelligence, Google Cloud Vision
*   **Data Extraction:** OpenAI
*   **Linting and Formatting:** Ruff
*   **Testing:** Pytest

## Building and Running

### Prerequisites

*   Python 3.12+
*   `uv` installed

### Installation

To install the project dependencies, run:

```bash
uv sync
```

### Configuration

The application is configured using environment variables. You can create a `.env` file in the project root to store these variables.

Here are some of the key configuration variables:

*   `API_KEY`: The API key to protect the `/api/v1/parse` endpoint.
*   `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`: The endpoint for your Azure Document Intelligence resource.
*   `AZURE_DOCUMENT_INTELLIGENCE_KEY`: The key for your Azure Document Intelligence resource.
*   `OPENAI_API_KEY`: The API key for the OpenAI service.
*   `DATABASE_URL`: The connection string for the PostgreSQL database.

For a full list of configuration options, please refer to `app/core/config.py`.

### Running the Application

To run the development server, use the following command:

```bash
uv run uvicorn app.main:app --reload
```

## Development Conventions

### Testing

Tests are located in the `tests/` directory. To run the test suite, use:

```bash
uv run python -m pytest
```

### Linting and Formatting

This project uses `Ruff` for linting and formatting.

To check for linting errors, run:
```bash
uv run python -m ruff check .
```

To check for formatting issues, run:
```bash
uv run python -m ruff format --check .
```

To automatically fix formatting issues, run:
```bash
uv run python -m ruff format .
```
