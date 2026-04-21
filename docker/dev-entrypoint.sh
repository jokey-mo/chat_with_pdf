#!/usr/bin/env bash
set -e

mkdir -p /data/pdfs /data/figures /data/paperqa /data/qdrant

python - <<'PY'
from backend.db.session import engine
from backend.db.models import Base
Base.metadata.create_all(engine)
print("[init] SQLite schema ready at /data/app.db")
PY

uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

trap "kill $API_PID 2>/dev/null || true" EXIT INT TERM

until curl -sf http://localhost:8000/health >/dev/null 2>&1; do
    sleep 1
done
echo "[init] API up on :8000"

exec streamlit run frontend/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false
