from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_chat_model: str = "anthropic/claude-3.7-sonnet"
    openrouter_vision_model: str = "google/gemini-2.5-flash"
    openrouter_embed_model: str = "openai/text-embedding-3-large"
    openrouter_app_name: str = "chat_with_pdf"
    openrouter_app_url: str = "http://localhost:8501"

    # Optional direct embeddings provider (falls back to OpenRouter if unset).
    # Set to "https://api.openai.com/v1" and provide EMBED_API_KEY to bypass OpenRouter for embeddings.
    embed_base_url: str = ""
    embed_api_key: str = ""

    openalex_api_key: str = ""
    openalex_email: str = ""
    unpaywall_email: str = ""
    core_api_key: str = ""

    postgres_dsn: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "paper_chunks"
    redis_url: str = "redis://localhost:6379/0"

    data_dir: Path = Path("/data")
    embed_dim: int = 3072

    retrieve_k: int = 40
    rerank_top: int = 8
    enable_reranker: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_url: str = "http://api:8000"


settings = Settings()

settings.data_dir.mkdir(parents=True, exist_ok=True)
(settings.data_dir / "pdfs").mkdir(exist_ok=True)
(settings.data_dir / "figures").mkdir(exist_ok=True)
(settings.data_dir / "paperqa").mkdir(exist_ok=True)
