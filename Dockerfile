# Build stage
FROM python:3.11-slim AS builder

WORKDIR /build

# Copy requirements file
COPY requirements.txt ./

# Install dependencies with retry logic for network timeouts
RUN pip install --no-cache-dir --upgrade pip && \
    (pip install --no-cache-dir -r requirements.txt || \
     (echo "First attempt failed, retrying..." && sleep 30 && pip install --no-cache-dir -r requirements.txt) || \
     (echo "Second attempt failed, retrying..." && sleep 60 && pip install --no-cache-dir -r requirements.txt))

# Pre-download NLTK data to avoid permission issues at runtime
RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True)" || true

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create writable directory for NLTK data
RUN mkdir -p /home/appuser/nltk_data && chown -R appuser:appuser /home/appuser/nltk_data

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app ./app

# Set ownership
RUN chown -R appuser:appuser /app

# Set NLTK data directory environment variable
ENV NLTK_DATA=/home/appuser/nltk_data

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8002/health')" || exit 1

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]

