FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit==1.40.2 \
    requests==2.32.3 \
    httpx==0.27.2 \
    pydantic==2.9.2 \
    python-dotenv==1.0.1

COPY frontend /app/frontend

ENV PYTHONPATH=/app
EXPOSE 8501
