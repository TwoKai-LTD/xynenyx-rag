"""FastAPI application for Xynenyx RAG Service."""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.routers import feeds, query, documents
from app.ingestion.scheduler import FeedScheduler
from app.schemas.errors import create_error_response
from app.middleware.logging import LoggingMiddleware

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Global scheduler instance
feed_scheduler: FeedScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager for application startup and shutdown events."""
    global feed_scheduler

    logger.info("RAG Service starting up...")
    # Initialize connections, warm up models, etc.

    # Start feed scheduler
    try:
        feed_scheduler = FeedScheduler()
        feed_scheduler.start()
        logger.info("Feed scheduler started")
    except Exception as e:
        logger.error(f"Failed to start feed scheduler: {e}")

    yield

    # Stop feed scheduler
    if feed_scheduler:
        try:
            feed_scheduler.stop()
            logger.info("Feed scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping feed scheduler: {e}")

    logger.info("RAG Service shutting down...")
    # Close connections, cleanup resources, etc.


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG service for Xynenyx with RSS ingestion and vector search",
    lifespan=lifespan,
)

# Add logging middleware (before CORS to capture all requests)
app.add_middleware(LoggingMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Include routers
app.include_router(feeds.router)
app.include_router(query.router)
app.include_router(documents.router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness check endpoint with dependency verification."""
    checks = {}
    all_ready = True

    # Check Supabase connection
    try:
        from app.clients.supabase import SupabaseClient
        client = SupabaseClient()
        # Simple query to verify connection
        result = client.client.table("documents").select("id").limit(1).execute()
        checks["supabase"] = "ready"
    except Exception as e:
        logger.error(f"Supabase connection check failed: {e}")
        checks["supabase"] = f"error: {str(e)}"
        all_ready = False

    # Check vector store (pgvector)
    try:
        from app.clients.supabase import SupabaseClient
        client = SupabaseClient()
        # Check if document_chunks table exists and is accessible
        result = client.client.table("document_chunks").select("id").limit(1).execute()
        checks["vector_store"] = "ready"
    except Exception as e:
        logger.error(f"Vector store check failed: {e}")
        checks["vector_store"] = f"error: {str(e)}"
        all_ready = False

    # Check LLM service
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as http_client:
            response = await http_client.get(f"{settings.llm_service_url}/health")
            if response.status_code == 200:
                checks["llm_service"] = "ready"
            else:
                checks["llm_service"] = f"unhealthy: {response.status_code}"
                all_ready = False
    except Exception as e:
        logger.error(f"LLM service check failed: {e}")
        checks["llm_service"] = f"error: {str(e)}"
        all_ready = False

    status_code = 200 if all_ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_ready else "not ready",
            "checks": checks,
        },
    )


# Error handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    errors = exc.errors()
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=create_error_response(
            detail="Validation error",
            status_code=422,
            code="VALIDATION_ERROR",
            errors=errors,
        ),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            detail=exc.detail,
            status_code=exc.status_code,
            code=f"HTTP_{exc.status_code}",
        ),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=create_error_response(
            detail="Internal server error",
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
        ),
    )
