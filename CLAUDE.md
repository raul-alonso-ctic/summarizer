# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Summarizer is a multimodal document processing microservice that extracts titles and descriptions from documents using vision-language models. It supports Google Drive and local filesystem sources.

**Supported file types:**
- **Multimodal (vision):** PDF, DOCX, DOC, ODT, JPG, JPEG, PNG, GIF, WEBP, BMP, TIFF
- **Archives:** ZIP, RAR, CBR, 7Z, TAR (including .tar.gz, .tgz, .tar.bz2, .tbz2, .tar.xz)
- **Text-only:** XML, EML (emails)
- **Excluded:** `.xsig` (digital signature files)

## Build & Run Commands

### Docker (Recommended)
```bash
# Build and run
docker compose up --build

# Run CLI inside container
docker exec -it summarizer python3 -m app.cli gdrive FOLDER_ID
docker exec -it summarizer python3 -m app.cli local /path/to/files
docker exec -it summarizer python3 -m app.cli retry-failed FOLDER_ID
```

### Local Development
```bash
python3 -m venv venv && source venv/activate
pip install -r requirements.txt

# Run API server (port 8567)
python3 -m uvicorn app.main:app --reload

# Run CLI with options
python3 -m app.cli local /path/to/files [--language es] [--output results.json] \
  [--initial-pages 2] [--final-pages 2] [--max-tokens 1024] \
  [--temperature-vllm 0.1] [--temperature-llm 0.3] [--top-p 0.9]

python3 -m app.cli gdrive FOLDER_ID [--name folder_name] [--file filename] [--file-id FILE_ID]
python3 -m app.cli retry-failed FOLDER_ID

# Utility commands
python3 -m app.cli checkpoint-to-results /path/to/checkpoint.json [--output results.json]
python3 -m app.cli add-missing-files /path/to/results.json [--output updated.json]
```

### Health Checks
```bash
curl http://localhost:8567/health
curl http://localhost:8567/health/gdrive
curl http://localhost:8567/health/llm
curl http://localhost:8567/health/vllm
```

## Architecture

### Processing Pipeline by File Type

**PDF & DOCX/DOC/ODT** (Multimodal):
1. Extract first N + last M pages (configurable, default 2 each)
2. Convert pages to JPEG images via pdf2image (PDF) or LibreOffice→pdf2image (DOCX/DOC/ODT)
3. Send images + prompt to VLLM (vision-language model)
4. Parse JSON response `{title, description}`
5. Corrupt/truncated PDFs are detected and skipped

**Archives (ZIP/RAR/7Z/TAR)** (Macro-summarization):
1. Decompress and recursively process all contained documents (parallel with `ARCHIVE_WORKERS`)
2. ZIP encoding auto-detection: UTF-8 → CP1252 (Windows) → CP437 (DOS) with graceful fallback
3. macOS metadata automatically filtered (`__MACOSX/` directories and `._` prefixed files)
4. Aggregate all descriptions from children
5. Two-stage LLM processing: description generation → title extraction
6. Return hierarchical result with children array

**XML & EML** (Text-only):
1. Extract text content (XML: recursive extraction filtering signature elements; EML: headers + body)
2. Content truncated to `XML_EML_CONTENT_LIMIT` (default 5000 chars)
3. Send to LLM for description, then title generation
4. Return plain text result

**Images (JPG/PNG/GIF/WEBP/BMP/TIFF)** (Multimodal):
1. Send image directly to VLLM (no page extraction needed)
2. Parse JSON response `{title, description}`
3. Supports images as standalone files or inside archives

### Key Services

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI server with endpoints |
| `app/cli.py` | CLI for batch processing (local/gdrive/retry-failed/checkpoint-to-results/add-missing-files) |
| `app/services/processor.py` | Main orchestrator - routes documents to handlers |
| `app/services/pdf.py` | PDF to images conversion with corruption detection |
| `app/services/docx.py` | DOCX/DOC/ODT to PDF via LibreOffice, then to images |
| `app/services/xml_eml.py` | XML and EML text extraction |
| `app/services/vllm.py` | Multimodal LLM service (documents with images) |
| `app/services/llm.py` | Text-only LLM service (archives/XML/EML) |
| `app/services/gdrive.py` | Google Drive API (Service Account + OAuth2 support) |
| `app/services/checkpoint.py` | Checkpoint/resume system for unattended batch processing |

### Data Flow
```
Request → processor.py → [pdf.py|docx.py|xml_eml.py] → [vllm.py|llm.py] → Response
```

## Environment Configuration

Key variables in `.env` (see `.env.example`):

### Model Configuration
| Variable | Purpose | Default |
|----------|---------|---------|
| `MODEL_API_URL` | LLM API endpoint | `http://localhost:11434/v1/chat/completions` |
| `MODEL_API_TOKEN` | Bearer token for API auth | (optional) |
| `VLLM_MODEL` | Multimodal model for PDF/DOCX | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` |
| `LLM_MODEL` | Text-only model for archives/XML/EML | `Qwen/Qwen3-32B` |
| `USE_VLLM_FOR_ALL` | Use VLLM_MODEL for all file types | `false` |
| `LLM_ENABLE_THINKING` | Enable extended thinking in LLM | `false` |

### Google Drive Configuration
| Variable | Purpose | Default |
|----------|---------|---------|
| `GOOGLE_DRIVE_ENABLED` | Enable Google Drive service | `true` |
| `GOOGLE_DRIVE_CREDENTIALS` | Path to credentials JSON | `/secrets/google-credentials.json` |
| `GOOGLE_DRIVE_TOKEN_PATH` | OAuth token cache path | `/data/gdrive_token.pickle` |
| `GOOGLE_DRIVE_FOLDER_ID` | Optional root folder ID | (optional) |
| `GDRIVE_DOWNLOAD_RETRIES` | Download retry attempts | `3` |

### Batch Processing & Checkpoints
| Variable | Purpose | Default |
|----------|---------|---------|
| `UNATTENDED_MODE` | Enable checkpoint system | `false` |
| `CHECKPOINT_DIR` | Checkpoint files directory | `/data/checkpoints` |
| `CHECKPOINT_INTERVAL` | Auto-save interval (seconds) | `60` |
| `BATCH_SIZE` | Files per batch | `1` |
| `MAX_WORKERS` | Parallel worker threads | `1` |
| `ARCHIVE_WORKERS` | Parallel workers for archive contents | `4` |
| `ARCHIVE_MAX_FILES` | Max files to process inside archives (0=unlimited) | `0` |
| `MAX_CONCURRENT_INFERENCE` | Global semaphore for inference calls | `16` |

### Content Processing
| Variable | Purpose | Default |
|----------|---------|---------|
| `XML_EML_CONTENT_LIMIT` | Max chars for XML/EML content | `5000` |
| `API_PORT` | Server port | `8000` |
| `NORMALIZE_NAMES` | Optional VIP list for name normalization | (optional) |

**Name Normalization:** When `NORMALIZE_NAMES` is set to a comma-separated list of names (e.g., `'Carlos Charro, Pablo Coca, Luisa Paz'`), the LLM will strictly normalize person names in document descriptions to match only these VIP names. Names not in the list are not included. Leave empty or unset to disable.

## Checkpoint System

For unattended batch processing with resume capability:

1. Enable with `UNATTENDED_MODE=true`
2. Checkpoints saved to `CHECKPOINT_DIR` every `CHECKPOINT_INTERVAL` seconds
3. Tracks: processed files, failed files, pending files
4. Resume interrupted processing automatically
5. Use `retry-failed` CLI command to reprocess failed files
6. Monitor progress via `GET /checkpoint/{folder_id}`

**Re-running on previously processed folders:**
When running on a folder with an existing checkpoint, the system automatically:
- Skips files already in `processed_files`
- Retries files in `failed_files`
- Processes new files not seen before

This is useful when new file formats are added (e.g., RAR support) - just re-run the same command and only the new format files will be processed.

## Google Drive File Discovery

Files are discovered by **both MIME type and file extension** to handle cases where Google Drive reports files with generic MIME types (e.g., `application/octet-stream`). The `get_all_files_recursive()` function in `gdrive.py` checks:

- MIME types: `application/pdf`, `application/zip`, `application/x-rar-compressed`, etc.
- Extensions: `.pdf`, `.docx`, `.rar`, `.7z`, `.tar.gz`, etc.

This ensures archives like RAR and 7Z are detected even when Google Drive doesn't recognize their specific MIME type.

## System Dependencies

- **poppler-utils**: PDF rendering (for pdf2image)
- **LibreOffice**: DOCX/DOC/ODT to PDF conversion
- **unrar**: RAR/CBR archive extraction (proprietary, from non-free repos — required for RAR5 support)
- **default-jre-headless**: Java Runtime Environment (LibreOffice dependency)

All are installed in the Docker image.

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Web UI with drag-and-drop upload |
| `/upload` | POST | File upload with advanced controls (temperature, pages, etc.) |
| `/summarize` | POST | Process documents from any source |
| `/process-folder` | POST | Batch process Google Drive folder |
| `/checkpoint/{folder_id}` | GET | Check processing progress |
| `/health` | GET | Basic health check |
| `/health/gdrive` | GET | Google Drive connectivity test |
| `/health/llm` | GET | LLM text service test |
| `/health/vllm` | GET | VLLM multimodal service test |
| `/docs` | GET | OpenAPI/Swagger documentation |

## Web UI Features

- Multi-file drag-and-drop upload
- Configurable initial/final pages or "Process All" option
- Temperature controls for VLLM and LLM separately
- Top-P sampling configuration
- JSON export of results
- Error handling with specific messages for corrupt/empty files
