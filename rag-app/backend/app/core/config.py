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
    # Free-tier quota is enforced PER MODEL, separately (verified live on
    # 2026-09-04: gemini-3.6-flash hit its 20-requests/day free-tier cap
    # mid-session while gemini-3.5-flash/3.7-flash/3.8-flash all still had
    # quota) -- generator and judge are deliberately kept on two DIFFERENT
    # models, not just for the self-preference-bias reason below, but so a
    # quota hit on one role doesn't block the other. gemini-2.5-flash/-pro
    # still 404 "no longer available to new users" for this key even though
    # they now appear in the live model list -- listed != callable, checked
    # live, don't assume from the catalog alone. Avoid "-latest"/"-flash"
    # bare aliases for the configured model: they can silently start
    # pointing at a different underlying model over time, which breaks the
    # guarantee that a trace's recorded model name reflects what actually
    # generated it.
    # Switched to the "-lite" tier on 2026-09-05 after gemini-3.5-flash (the
    # judge) hit a transient 503 "high demand" mid-session -- the -lite
    # models have their own separate, previously-untouched quota pool, and
    # were verified live (both streaming for the generator role and strict
    # structured-JSON output for the judge role) before switching.
    gemini_api_key: str = ""
    gemini_generator_model: str = "gemini-3.5-flash-lite"
    gemini_judge_model: str = "gemini-3.1-flash-lite"

    # --- Groq (second LLM provider, selectable per-request in the UI) -----
    # Verified live against this account's actual model catalog (GET
    # https://api.groq.com/openai/v1/models) on 2026-09-02, not assumed from
    # memory or docs -- the entire Llama 3.1 lineup (previously
    # llama-3.1-70b-versatile / llama-3.1-8b-instant) has been decommissioned.
    # gpt-oss models stream a separate `reasoning` delta channel alongside
    # `content` -- groq_client.py already only reads `delta.content`, so the
    # reasoning trace is naturally skipped, no code change needed for that.
    # Generator switched to qwen/qwen3.8-27b on 2026-09-05 -- verified live
    # against the actual generation prompt/context shape: clean output,
    # correctly-formatted [1]-style citations, no reasoning leakage, and a
    # separate quota pool from gpt-oss-120b. Its sibling qwen/qwen3.6-27b was
    # tested and rejected for this role: it inlines its full chain-of-thought
    # directly into `delta.content` (wrapped in <think>...</think>, ~20x the
    # length of the real answer) rather than on a separate channel the way
    # gpt-oss does, so it would leak raw reasoning into the streamed answer.
    groq_api_key: str = ""
    groq_generator_model: str = "qwen/qwen3.8-27b"
    groq_judge_model: str = "openai/gpt-oss-20b"

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
    mmr_lambda: float = 0.5
    mmr_pool_size: int = 15

    # --- Storage --------------------------------------------------------
    data_dir: str = "data"
    uploads_dir: str = "data/uploads"
    registry_path: str = "data/registry/documents.json"

    # Traces are split into two append-only files by outcome (error is None
    # -> success, error is set -> failure) rather than one file with a mixed
    # `error` field -- makes "show me what's actually broken" a file-read
    # instead of a scan/filter, and mirrors the same split on the app-level
    # log files below.
    trace_success_log_path: str = "data/traces/successful_traces.jsonl"
    trace_failure_log_path: str = "data/traces/failed_traces.jsonl"

    # Process-level (Python `logging`) output, split the same way: anything
    # below WARNING is normal pipeline progress, WARNING and above is
    # something that actually went wrong. Console output (stderr) is
    # unaffected -- these are additional file handlers, not a replacement.
    success_log_path: str = "data/logs/success.log"
    failure_log_path: str = "data/logs/failure.log"

    # --- Observability (Langfuse, self-hosted) ---------------------------
    # Optional -- the app runs identically with this unset (LANGFUSE_ENABLED
    # defaults to False, so no Langfuse client is constructed and
    # chat_service.py's instrumentation is a no-op). Point LANGFUSE_HOST at
    # a self-hosted instance (see docker-compose.yml) or Langfuse Cloud.
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3001"

    # --- Server --------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:8501"]

    # --- Week 7 agent conversation memory (bonus) ------------------------
    # Window/summary live only in-memory (app.state.agent_conversations) and
    # are deliberately lost on restart; session_store_path is the one thing
    # that survives one (see app/registry/session_store.py).
    conversation_window_turns: int = 6
    conversation_fold_batch_size: int = 3
    session_store_path: str = "data/registry/sessions.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
