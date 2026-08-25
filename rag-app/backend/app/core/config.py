from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration, populated from environment variables / .env.

    Field names map to env vars case-insensitively (e.g. `gemini_api_key`
    reads `GEMINI_API_KEY`).
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Gemini -----------------------------------------------------
    gemini_api_key: str = ""
    gemini_generator_model: str = "gemini-3.6-flash"
    gemini_judge_model: str = "gemini-3.5-flash"

    # --- Groq (second LLM provider, selectable per-request in the UI) -----
    groq_api_key: str = ""
    groq_generator_model: str = "llama-3.1-70b-versatile"
    groq_judge_model: str = "llama-3.1-8b-instant"

    # --- Qdrant -------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_documents"

    # --- Models --------------------------------------------------------
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    sparse_model_name: str = "Qdrant/bm25"

    # --- Chunking --------------------------------------------------------
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 75

    # --- Retrieval --------------------------------------------------------
    first_stage_limit: int = 40
    rerank_top_k: int = 6
    query_expansion_variant_count: int = 3
    query_expansion_similarity_threshold: float = 0.80
    off_topic_score_threshold: float = 0.15

    # --- Storage --------------------------------------------------------
    data_dir: str = "data"
    uploads_dir: str = "data/uploads"
    registry_path: str = "data/registry/documents.json"

    # --- Server --------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:8501"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
