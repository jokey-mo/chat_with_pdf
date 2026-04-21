# Agro-RAG — контекст проекта

NotebookLM-подобный сервис для научных статей по агрономии: **растениеводство, почвоведение, гидропоника, промышленные теплицы**. В отличие от NotebookLM — cron-поиск новых статей из научных источников и автоматическая ингестия.

## Архитектура

```
Streamlit ─► FastAPI ─► Postgres (metadata, APS jobs)
                     └► Qdrant   (vectors)
                     └► RQ/Redis ─► Worker ─► sources → PDF → Docling → embed
Scheduler ──────────► RQ/Redis (по cron)
```

6 production-контейнеров: `frontend`, `api`, `worker`, `scheduler`, `qdrant`, `postgres`, `redis`.

## Стек

| Слой | Выбор |
|---|---|
| RAG-ядро | Гибрид: свой retriever+chat (режим A) + PaperQA2 (режим B, не полностью разведён в MVP) |
| Источники | **OpenAlex** + **Unpaywall** (резолв DOI→PDF), **agriRxiv** через OSF API. Опц.: CORE, EarthArXiv |
| PDF парсинг | **Docling** (IBM, MIT) primary; **PyMuPDF4LLM** fallback |
| Эмбеддинги | `openai/text-embedding-3-large` через OpenRouter (3072d) |
| Vector DB | Qdrant (hybrid BM25+dense, metadata filters). Поддерживает embedded mode через `file://` |
| Metadata DB | Postgres (prod) / SQLite (dev) |
| Reranker | BGE-Reranker-v2-m3 (опц., CPU) |
| Scheduler | APScheduler (SQLAlchemy jobstore) + RQ на Redis |
| LLM | OpenRouter: `anthropic/claude-3.7-sonnet` (chat), `google/gemini-2.5-flash` (vision captions) |
| UI / API | Streamlit / FastAPI + SSE streaming |

## Текущая ветка и статус

- Ветка разработки: **`claude/scientific-paper-rag-system-s94dj`**
- Последний коммит: `780d46f Add single-container dev mode for laptop testing`
- Предыдущие:
  - `62d6842` — docker-less local dev (embedded Qdrant, SQLite, tiktoken fallback, inline ingestion)
  - `2781305` — первоначальный MVP: 6 сервисов, 62 файла, 2571 insertions

Все изменения запушены в origin.

## Как запускать

### 1. Dev-контейнер (для MacBook / быстрого теста)

Один docker-образ: FastAPI + Streamlit + SQLite + embedded Qdrant. Slim-deps (без docling/FlagEmbedding/paperqa).

```bash
cp .env.example .env
# впиши OPENROUTER_API_KEY, UNPAYWALL_EMAIL, OPENALEX_EMAIL
docker compose -f docker-compose.dev.yml up --build
```

→ UI: `localhost:8501`, API: `localhost:8000/docs`

Ограничения: нет cron/worker (ингестия inline в daemon-треде), только PyMuPDF4LLM (без фигур), без reranker'а.

Файлы: `docker-compose.dev.yml`, `docker/dev.Dockerfile`, `docker/dev-entrypoint.sh`, `requirements-slim.txt`.

### 2. Без Docker (локальный Python)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # + ключи
python -c "from backend.db.session import engine; from backend.db.models import Base; Base.metadata.create_all(engine)"
# в двух терминалах:
uvicorn backend.api.main:app --port 8000 --reload
streamlit run frontend/streamlit_app.py
```

### 3. Полный docker-compose (production-like)

```bash
cp .env.example .env
# замени DSN/URL на сервисные hostnames (postgres/qdrant/redis)
docker compose up -d postgres qdrant redis
docker compose run --rm api alembic upgrade head
docker compose up -d api worker scheduler frontend
```

## Переменные окружения (.env)

| Переменная | Назначение | Обязательна |
|---|---|---|
| `OPENROUTER_API_KEY` | LLM + embeddings | **да** |
| `UNPAYWALL_EMAIL` | DOI→PDF резолвер | **да** |
| `OPENALEX_EMAIL` | polite-pool OpenAlex | **да** |
| `OPENALEX_API_KEY` | ключ OpenAlex (с фев.2026) | нет |
| `CORE_API_KEY` | fallback-агрегатор | нет |
| `OPENROUTER_CHAT_MODEL` | модель чата | `anthropic/claude-3.7-sonnet` |
| `OPENROUTER_VISION_MODEL` | модель для подписей фигур | `google/gemini-2.5-flash` |
| `OPENROUTER_EMBED_MODEL` | embed-модель | `openai/text-embedding-3-large` |
| `POSTGRES_DSN` | БД | `sqlite:///./data/app.db` (dev) |
| `QDRANT_URL` | Qdrant | `file://./data/qdrant` (dev) или `http://qdrant:6333` |
| `REDIS_URL` | RQ/APS | `redis://localhost:6379/0` |
| `DATA_DIR` | путь для PDF/фигур | `./data` (dev) / `/data` (prod) |
| `EMBED_DIM` | размерность | `3072` |
| `RETRIEVE_K` | top-K из Qdrant | `40` |
| `RERANK_TOP` | top-N после reranker | `8` |
| `ENABLE_RERANKER` | включить BGE | `false` |
| `RUN_INGESTION_INLINE` | ингестия в daemon-треде | `true` (dev) / `false` (prod) |

## Требования к VM (production)

| Профиль | vCPU | RAM | SSD | Реальный пример |
|---|---|---|---|---|
| MVP / тест | 2 | 4 GB | 40 GB | Hetzner CPX21 €8/мес |
| Комфорт + reranker | 4 | 8-16 GB | 100-200 GB | Hetzner CPX31 €15/мес |
| Scale (10k+ статей) | 4 | 16 GB | 200+ GB | — |

GPU не требуется. Главный едок RAM — Docling при парсинге (1-2GB пиково). Reranker добавляет ~1.5GB.

Наружу открывать **только** `8501` (UI) и опц. `8000` (API). Postgres/Redis/Qdrant держать внутри docker-сети.

## Структура репо

```
backend/
  api/
    main.py                     # FastAPI app + /health
    routes/{notebooks,chat,ingest,papers}.py
  core/
    config.py                   # pydantic-settings (с RUN_INGESTION_INLINE)
    openrouter.py               # OpenAI SDK с OpenRouter base_url + headers
    logging.py
  db/
    models.py                   # Notebook, Paper, Chunk, Figure, IngestionRun, PaperStatus
    session.py                  # SQLAlchemy engine
    migrations/                 # alembic (пишет под Postgres)
  ingestion/
    pipeline.py                 # run_ingestion(notebook_id)
    dedup.py
    sources/
      base.py                   # PaperRef, Source Protocol
      openalex.py               # OpenAlexSource (pyalex)
      unpaywall.py              # resolve_pdf(doi)
      agrirxiv.py               # OSF API
  parsing/
    docling_parser.py           # parse() с PyMuPDF fallback (docling auto-skip если нет)
    chunker.py                  # section_aware_chunks с tiktoken+word-fallback
    figure_captioner.py         # vision LLM через OpenRouter
  rag/
    embeddings.py               # embed_batch / aembed_batch
    qdrant_store.py             # поддержка file:// и :memory:
    retriever.py                # hybrid search + join с Postgres
    rerank.py                   # BGE-reranker (опц.)
    chat.py                     # SSE streaming с [P#] маркерами
    paperqa_chat.py             # режим B (заготовка)
    paperqa_sync.py
  scheduler/
    runner.py                   # APScheduler + _sync_jobs каждые 60с
    jobs.py                     # enqueue_ingestion в RQ
  workers/
    worker.py                   # RQ Worker
frontend/
  streamlit_app.py              # sidebar notebooks + chat + ingestion dashboard
docker/
  api.Dockerfile
  worker.Dockerfile
  frontend.Dockerfile
  dev.Dockerfile                # single-container dev-режим
  dev-entrypoint.sh
docker-compose.yml              # prod (6 сервисов)
docker-compose.dev.yml          # dev (1 контейнер)
requirements.txt                # полный
requirements-slim.txt           # для dev-контейнера
.env / .env.example
```

## Модель данных

- `notebooks(id, name, topic_query, sources jsonb, schedule_cron, openrouter_model, created_at)`
- `papers(id, external_id, source, doi, title, authors, abstract, year, pdf_path, parsed_json_path, status[queued|parsing|embedded|failed], notebook_id FK, ingested_at, checksum, error_msg)`, unique `(source, external_id)`
- `chunks(id, paper_id FK, section, page_start, page_end, text, token_count, qdrant_point_id)`
- `figures(id, paper_id FK, page, bbox, image_path, caption_original, caption_vlm, qdrant_point_id)`
- `ingestion_runs(id, notebook_id FK, started_at, finished_at, n_found, n_new, n_embedded, n_failed, error_summary)`

Qdrant collection `paper_chunks`: dense 3072d + sparse BM25, payload `{paper_id, notebook_id, source, section, page, kind}`.

## Сделанные архитектурные решения

1. **Гибрид собственного RAG + PaperQA2** — не полный PaperQA2, т.к. нужна кастомная ingestion + cron + per-notebook изоляция.
2. **OpenAlex+Unpaywall+agriRxiv** вместо arxiv — потому что arxiv слабо покрывает agronomy; OpenAlex даёт >250M работ с фильтрами по concepts (crop/soil/horticulture).
3. **Docling primary, PyMuPDF4LLM fallback** — docling извлекает структуру и фигуры, но тяжёлый и требует pytorch. Graceful fallback.
4. **Embeddings через OpenRouter, не локально** — 3072d модели требуют GPU, дешевле через API.
5. **Qdrant вместо pgvector** — нужен hybrid BM25+dense из коробки.
6. **SSE streaming** с inline `[P#]` маркерами, которые фронт превращает в ссылки на PDF/DOI.
7. **Per-notebook cron** через APScheduler SQLAlchemy jobstore — персистентность задач переживает рестарт.
8. **Режим dev без Docker** — для sandbox/CI/MacBook: SQLite + embedded Qdrant + tiktoken word-fallback + inline ingestion.

## Известные ограничения / TODO

- Alembic миграции написаны под Postgres (enum). Для SQLite используется `Base.metadata.create_all(engine)`.
- PaperQA2 "deep" режим (B) — заготовка, нужна разводка `paperqa.Docs.aquery` + персистентность `Docs` per-notebook.
- Figure captions работают только с docling — PyMuPDF4LLM не извлекает bbox фигур.
- `tests/eval/questions.yaml` для recall@K ещё не заполнен тестовыми QA.
- Нет auth/multi-user — любой с доступом к UI видит все ноутбуки.
- Нет reverse proxy / HTTPS в compose — добавляется отдельно (nginx/Caddy).
- Нет rate-limiting и бэкапов из коробки.

## Этапы

- **MVP** (сделано): OpenAlex + Unpaywall + agriRxiv, Docling, Qdrant, SSE chat с цитированиями.
- **v1**: vision figure captions в UI (thumbnails), BGE reranker on, PaperQA2 режим B.
- **v2**: CORE + EarthArXiv, мульти-ноутбук cron, фильтры по дате/источнику, summary/compare N папер.

## Критические файлы

- `backend/ingestion/pipeline.py` — оркестратор ингестии
- `backend/ingestion/sources/openalex.py`, `unpaywall.py` — ядро поиска
- `backend/parsing/docling_parser.py` — парсинг с fallback
- `backend/rag/retriever.py`, `rag/chat.py` — режим A
- `backend/scheduler/runner.py` — cron
- `docker-compose.yml` / `docker-compose.dev.yml` — оркестрация
- `.env.example` — шаблон конфига

## Git

- Origin: `jokey-mo/chat_with_pdf`
- Рабочая ветка: `claude/scientific-paper-rag-system-s94dj`
- Main (protected): не пушить напрямую
- PR не создан (пользователь не просил)
