FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-slim.txt .
RUN pip install --no-cache-dir -r requirements-slim.txt

COPY backend /app/backend
COPY frontend /app/frontend
COPY docker/dev-entrypoint.sh /app/dev-entrypoint.sh
RUN chmod +x /app/dev-entrypoint.sh

ENV PYTHONPATH=/app \
    POSTGRES_DSN=sqlite:////data/app.db \
    QDRANT_URL=file:///data/qdrant \
    DATA_DIR=/data \
    RUN_INGESTION_INLINE=true \
    API_URL=http://localhost:8000

EXPOSE 8000 8501

ENTRYPOINT ["/app/dev-entrypoint.sh"]
