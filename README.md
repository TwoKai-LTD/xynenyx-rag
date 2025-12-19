# Xynenyx RAG Service

LlamaIndex RAG service for RSS feed ingestion, content parsing, embedding, and hybrid retrieval.

## Overview

The RAG service:

- Ingests RSS feeds from startup/VC news sources
- Parses and chunks content with metadata extraction
- Generates embeddings via LLM service
- Stores in Supabase pgvector
- Provides hybrid search (vector + BM25) with reranking

## Quick Start

### Local Development

```bash
# Install dependencies
poetry install

# Run locally
poetry run uvicorn app.main:app --port 8002 --reload
```

### Docker

```bash
docker build -t xynenyx-rag .
docker run -p 8002:8002 --env-file .env xynenyx-rag
```

## API Endpoints

- `GET /health` - Health check
- `GET /ready` - Readiness check
- `POST /feeds` - Register RSS feed
- `GET /feeds` - List feeds
- `POST /feeds/{id}/ingest` - Trigger ingestion
- `POST /query` - Vector search query
- `GET /documents` - List documents

## Configuration

See `.env.example` for all configuration options.

## Testing

```bash
poetry run pytest -v
poetry run pytest --cov=app --cov-report=html
```

## License

MIT License - see [LICENSE](LICENSE) file

