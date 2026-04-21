# 🌱 Agro-RAG — Scientific Paper Chat for Agronomy

NotebookLM-like сервис для научных статей по **растениеводству, почвоведению, гидропонике и промышленным теплицам**.

В отличие от NotebookLM, здесь есть **cron-поиск** новых статей из проверенных источников и автоматическая ингестия: поиск → скачивание PDF → разбор (включая фигуры) → эмбеддинги → чат с цитированиями.

## Архитектура

- **Источники**: OpenAlex (+ Unpaywall для резолва PDF), agriRxiv (preprints).
- **PDF парсинг**: [Docling](https://github.com/docling-project/docling) — структура + фигуры; fallback PyMuPDF4LLM.
- **Vision captions**: OpenRouter (Gemini 2.5 Flash / Claude 3.7 Sonnet).
- **Эмбеддинги**: `openai/text-embedding-3-large` через OpenRouter (3072d).
- **Vector DB**: Qdrant.
- **Reranker** (опц.): BAAI/bge-reranker-v2-m3.
- **Chat LLM**: OpenRouter (`anthropic/claude-3.7-sonnet` по умолчанию).
- **Scheduler**: APScheduler + RQ (Redis).
- **UI**: Streamlit.
- **Бэкенд**: FastAPI + SQLAlchemy + Postgres.

```
Streamlit ─► FastAPI ─► Postgres (metadata, APS jobs)
                     └► Qdrant   (vectors)
                     └► RQ/Redis ─► Worker ─► sources → PDF → Docling → embed
Scheduler ──────────► RQ/Redis (по cron)
```

## Быстрый старт

### Вариант 1 — dev-контейнер (одна команда, для теста на ноутбуке)

Один образ со всем внутри: FastAPI + Streamlit + SQLite + embedded Qdrant. Без Postgres/Redis/Docling, парсинг через PyMuPDF4LLM. Идеально для проверки на MacBook / Windows / Linux.

```bash
cp .env.example .env
# заполните: OPENROUTER_API_KEY, UNPAYWALL_EMAIL, OPENALEX_EMAIL

docker compose -f docker-compose.dev.yml up --build
```

UI: [http://localhost:8501](http://localhost:8501) · API: [http://localhost:8000/docs](http://localhost:8000/docs)

Данные сохраняются в `./data/` (SQLite, Qdrant, PDF). Ограничения dev-режима: нет cron-scheduler'а и фонового worker'а — ингестия запускается вручную из UI и выполняется inline.

### Вариант 2 — полный стек (production-like)

```bash
cp .env.example .env
# + поменяйте POSTGRES_DSN / QDRANT_URL / REDIS_URL на сервисные hostnames

docker compose up -d postgres qdrant redis
docker compose up -d api worker scheduler frontend
```

## Использование

1. Создайте ноутбук в сайдбаре: `name`, `topic_query`, `sources`, `schedule_cron` (например, `0 3 * * *`).
2. Нажмите **▶ Запустить ингестию** для разового запуска (или дождитесь cron).
3. Дождитесь статуса `embedded` у статей в дашборде.
4. Задавайте вопросы в чате — в ответе появляются маркеры `[P1 p.3]`, кликабельные на PDF/DOI.

## Переменные окружения

См. `.env.example`. Ключевые:

| var | назначение |
|---|---|
| `OPENROUTER_API_KEY` | ключ OpenRouter (LLM + vision + embeddings) |
| `OPENROUTER_CHAT_MODEL` | модель для Q&A (default: `anthropic/claude-3.7-sonnet`) |
| `OPENROUTER_VISION_MODEL` | модель для описания фигур (default: `google/gemini-2.5-flash`) |
| `OPENROUTER_EMBED_MODEL` | embed-модель (default: `openai/text-embedding-3-large`) |
| `OPENALEX_EMAIL`, `OPENALEX_API_KEY` | доступ к OpenAlex |
| `UNPAYWALL_EMAIL` | обязателен для Unpaywall |
| `ENABLE_RERANKER` | `true` — включает BGE-reranker локально (CPU) |

## Модули

```
backend/
  api/          FastAPI + роуты
  db/           SQLAlchemy + Alembic
  ingestion/    источники (OpenAlex, Unpaywall, agriRxiv) + pipeline
  parsing/      Docling-парсер, section-aware chunker, figure captioner
  rag/          embeddings, Qdrant store, retriever, rerank, chat (streaming)
  scheduler/    APScheduler runner + RQ job helpers
  workers/      RQ worker entrypoint
  core/         config, logging, OpenRouter клиент
frontend/       Streamlit UI
```

## Тесты

```bash
pytest tests/
```

## Этапы

- **MVP (текущее)**: OpenAlex + Unpaywall + agriRxiv, Docling, Qdrant, streaming chat с цитированиями, cron через APScheduler+RQ.
- **v1 (след)**: vision captions для фигур в чате (thumbnails), BGE reranker включён, PaperQA2 "deep" режим.
- **v2**: + CORE, EarthArXiv; reranker Cohere; сравнение N статей.

## Лицензия

Apache-2.0 (проект) — зависимости под своими лицензиями (Docling MIT, Qdrant Apache-2.0 и т.д.).
