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

    # CORS settings
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    @model_validator(mode="after")
    def resolve_service_role_key(self):
        """Resolve service role key from either variable name."""
        if not self.supabase_service_role_key and self.supabase_service_key:
            self.supabase_service_role_key = self.supabase_service_key
        if not self.supabase_service_role_key:
            raise ValueError("Either supabase_service_role_key or supabase_service_key must be set")
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from .env
    )


settings = Settings()

