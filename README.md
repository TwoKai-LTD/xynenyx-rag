# Xynenyx RAG Service

LlamaIndex RAG service for RSS feed ingestion, content parsing, embedding, and vector retrieval.

## Overview

The RAG service:

- Ingests RSS feeds from startup/VC news sources (scheduled and manual)
- Parses and chunks content with enhanced metadata extraction
- Generates embeddings via LLM service
- Stores in Supabase pgvector
- Provides hybrid search (BM25 + vector) with RRF fusion
- Cross-encoder reranking for improved relevance
- Temporal and entity filtering
- Database-backed feed management

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

### Vector Search (Advanced)

- `POST /query` - Hybrid search with reranking and filtering
  ```json
  {
    "query": "AI startup funding",
    "top_k": 10,
    "use_hybrid_search": true,
    "use_reranking": true,
    "rerank_top_n": 20,
    "date_filter": "last_week",
    "company_filter": ["Anthropic", "OpenAI"],
    "investor_filter": ["Andreessen Horowitz"],
    "sector_filter": ["AI", "FinTech"]
  }
  ```

  **Query Parameters:**
  - `query` (required) - Search query text
  - `top_k` (default: 10) - Number of results to return
  - `use_hybrid_search` (default: true) - Enable hybrid search (BM25 + vector)
  - `use_reranking` (default: true) - Enable cross-encoder reranking
  - `rerank_top_n` (default: 20) - Top N results to rerank
  - `date_filter` (optional) - Temporal filter: `"last_week"`, `"this_month"`, or `{"start_date": "2024-01-01", "end_date": "2024-01-31"}`
  - `company_filter` (optional) - Filter by company names
  - `investor_filter` (optional) - Filter by investor names
  - `sector_filter` (optional) - Filter by sectors/industries
  - `filter_document_ids` (optional) - Filter by specific document IDs

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
3. **Metadata Extractor** - Enhanced extraction of companies, funding (with rounds), dates (dateparser), investors (with roles), sectors (with confidence)
4. **Chunker** - Splits content into chunks using LlamaIndex SentenceSplitter
5. **Embedding Generation** - Calls LLM service `/embeddings` endpoint
6. **Vector Storage** - Stores chunks with embeddings in Supabase pgvector

### Retrieval

- **Hybrid Search:** Combines BM25 (keyword) and vector (semantic) search using Reciprocal Rank Fusion (RRF)
- **Reranking:** Cross-encoder reranking improves precision on top results
- **Filtering:** Temporal (date ranges) and entity (company, investor, sector) filtering
- **Metadata:** Enhanced extraction of companies, funding, dates, investors, sectors with confidence scores
- **Backward Compatible:** Vector-only search still supported

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

### Query Hybrid Search

```bash
# Basic hybrid search
curl -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -H "X-User-ID: user-123" \
  -d '{
    "query": "AI startup funding rounds",
    "top_k": 10,
    "use_hybrid_search": true,
    "use_reranking": true
  }'

# With filters
curl -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -H "X-User-ID: user-123" \
  -d '{
    "query": "funding rounds",
    "top_k": 5,
    "date_filter": "last_week",
    "company_filter": ["Anthropic"],
    "sector_filter": ["AI"]
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
│   ├── retrieval/           # Vector retrieval and hybrid search
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   ├── bm25_retriever.py
│   │   ├── hybrid_retriever.py
│   │   ├── reranker.py
│   │   └── filters.py
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
