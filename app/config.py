"""Configuration settings for Xynenyx RAG Service."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from typing import Dict, Any


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Service settings
    app_name: str = "Xynenyx RAG Service"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8002

    # Supabase settings
    supabase_url: str
    supabase_service_role_key: str | None = None
    # Support alternative name from .env (SUPABASE_SERVICE_KEY)
    supabase_service_key: str | None = None

    # LLM Service settings
    llm_service_url: str = "http://localhost:8003"
    llm_service_timeout: int = 60

    # AWS S3 settings (for future document storage)
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "us-east-1"
    s3_bucket_name: str | None = None

    # RSS feed settings
    default_update_frequency: str = "hourly"  # hourly, daily
    rss_request_timeout: int = 30
    rss_max_retries: int = 3

    # HTML extraction settings
    html_request_timeout: int = 30
    html_max_retries: int = 3
    html_user_agent: str = "Mozilla/5.0 (compatible; XynenyxBot/1.0)"

    # Chunking settings
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 50  # tokens

    # Embedding settings
    embedding_batch_size: int = 20  # chunks per batch
    embedding_max_retries: int = 3
    embedding_retry_delay: float = 1.0  # seconds

    # Vector search settings
    default_top_k: int = 10
    min_similarity_score: float = 0.0  # minimum similarity threshold

    # Hybrid search settings
    use_hybrid_search: bool = True
    bm25_top_k: int = 10  # Top-k for BM25 retrieval
    vector_top_k: int = 10  # Top-k for vector retrieval
    rrf_k: int = 60  # RRF k parameter (higher = more weight to top results)

    # Reranking settings
    use_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 20  # Top N results to rerank
    reranker_cache_dir: str | None = None  # Custom cache directory (defaults to user cache)

    # Filter settings
    temporal_filter_presets: Dict[str, str] = {
        "last_week": "7 days",
        "this_month": "30 days",
        "this_quarter": "90 days",
        "this_year": "365 days",
        "last_month": "30 days",
        "last_quarter": "90 days",
        "last_year": "365 days",
    }

    # Metadata extraction settings
    metadata_confidence_threshold: float = 0.5
    enable_ner: bool = False  # Enable NER for company extraction (requires spaCy)

    # BM25 index settings
    bm25_index_refresh_interval: int = 3600  # Refresh index every hour (seconds)

    # CORS settings
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    @model_validator(mode="after")
    def validate_config(self):
        """Validate all required configuration."""
        errors = []
        
        # Resolve service role key
        if not self.supabase_service_role_key and self.supabase_service_key:
            self.supabase_service_role_key = self.supabase_service_key
        
        # Validate Supabase
        if not self.supabase_url:
            errors.append("SUPABASE_URL is required")
        elif not self.supabase_url.startswith("http"):
            errors.append("SUPABASE_URL must be a valid HTTP/HTTPS URL")
        
        if not self.supabase_service_role_key:
            errors.append("Either SUPABASE_SERVICE_ROLE_KEY or SUPABASE_SERVICE_KEY must be set")
        
        # Validate LLM service URL
        if not self.llm_service_url:
            errors.append("LLM_SERVICE_URL is required")
        elif not self.llm_service_url.startswith("http"):
            errors.append("LLM_SERVICE_URL must be a valid HTTP/HTTPS URL")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from .env
    )


settings = Settings()

