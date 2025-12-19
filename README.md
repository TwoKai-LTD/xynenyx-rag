# Xynenyx RAG Service

LlamaIndex RAG service for RSS feed ingestion, content parsing, embedding, and vector retrieval.

## Overview

The RAG service:

- Ingests RSS feeds from startup/VC news sources
- Parses and chunks content with metadata extraction
- Generates embeddings via LLM service
- Stores in Supabase pgvector
- Provides vector similarity search

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

### Health & Readiness

- `GET /health` - Health check
- `GET /ready` - Readiness check

### RSS Feed Management

- `POST /feeds` - Register new RSS feed
  ```json
  {
    "name": "TechCrunch Startups",
    "url": "https://techcrunch.com/feed/",
    "update_frequency": "hourly"
  }
  ```

- `GET /feeds` - List all feeds
- `GET /feeds/{id}` - Get feed details
- `POST /feeds/{id}/ingest` - Trigger manual ingestion
- `DELETE /feeds/{id}` - Remove feed

### Vector Search

- `POST /query` - Vector similarity search
  ```json
  {
    "query": "AI startup funding",
    "top_k": 5,
    "filter_document_ids": ["optional-doc-id"]
  }
  ```

### Document Management

- `GET /documents` - List documents (with optional filters: `?status=ready&limit=100&offset=0`)
- `GET /documents/{id}` - Get document details

## Configuration

See `.env.example` for all configuration options. Key variables:

- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Supabase service role key
- `LLM_SERVICE_URL` - LLM service URL (default: http://localhost:8003)
- `CHUNK_SIZE` - Chunk size in tokens (default: 512)
- `CHUNK_OVERLAP` - Chunk overlap in tokens (default: 50)

## Testing

```bash
# Run all tests
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test file
poetry run pytest tests/test_ingestion.py -v
```

## Architecture

### Ingestion Pipeline

1. **RSS Parser** - Parses RSS/Atom feeds using `feedparser`
2. **HTML Parser** - Extracts main content from article URLs using BeautifulSoup
3. **Metadata Extractor** - Extracts companies, funding, dates, investors using regex patterns
4. **Chunker** - Splits content into chunks using LlamaIndex SentenceSplitter
5. **Embedding Generation** - Calls LLM service `/embeddings` endpoint
6. **Vector Storage** - Stores chunks with embeddings in Supabase pgvector

### Retrieval

- Vector similarity search using cosine similarity
- Query embedding generation via LLM service
- Results filtered by similarity threshold
- Metadata preserved for source attribution

## Usage Examples

### Register and Ingest a Feed

```bash
# Register feed
curl -X POST http://localhost:8002/feeds \
  -H "Content-Type: application/json" \
  -H "X-User-ID: user-123" \
  -d '{
    "name": "TechCrunch",
    "url": "https://techcrunch.com/feed/",
    "update_frequency": "hourly"
  }'

# Trigger ingestion
curl -X POST http://localhost:8002/feeds/{feed_id}/ingest \
  -H "X-User-ID: user-123"
```

### Query Vector Search

```bash
curl -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -H "X-User-ID: user-123" \
  -d '{
    "query": "AI startup funding rounds",
    "top_k": 10
  }'
```

### List Documents

```bash
curl http://localhost:8002/documents?status=ready \
  -H "X-User-ID: user-123"
```

## Development

### Project Structure

```
xynenyx-rag/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── clients/              # External service clients
│   │   ├── supabase.py      # Supabase client
│   │   └── llm.py           # LLM service client
│   ├── ingestion/           # RSS ingestion pipeline
│   │   ├── rss_parser.py
│   │   ├── html_parser.py
│   │   ├── metadata_extractor.py
│   │   ├── chunkers.py
│   │   ├── pipeline.py
│   │   └── scheduler.py
│   ├── retrieval/           # Vector retrieval
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── routers/              # API routes
│   │   ├── feeds.py
│   │   ├── query.py
│   │   └── documents.py
│   └── schemas/              # Pydantic models
│       ├── feeds.py
│       ├── queries.py
│       └── documents.py
└── tests/                     # Test suite
```

## License

MIT License - see [LICENSE](LICENSE) file
