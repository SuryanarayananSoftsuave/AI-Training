from __future__ import annotations

import json
from pathlib import Path

from app.core.config import Settings
from app.retrieval.mmr import mmr_select
from app.retrieval.pipeline import rerank_candidates, retrieve_candidates
from app.retrieval.qdrant_store import QdrantStore

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hit_rate_questions.json"


def load_questions() -> list[dict]:
    data = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    return data["questions"]


async def _top_k_filenames(question: str, k: int, use_mmr: bool, settings: Settings, store: QdrantStore) -> list[str]:
    """Retrieval + rerank (+ optional MMR) only -- no generation, no judge.
    Mirrors `chat_service.answer_query`'s retrieval path (hybrid search,
    since that's the already-built baseline this eval measures improvements
    on top of) but skips both LLM calls entirely, since hit-rate@k is purely
    a retrieval-quality metric and doesn't need an actual generated answer.
    """
    candidates, _total_raw = await retrieve_candidates(
        [question], use_keyword_search=True, limit=settings.first_stage_limit,
        doc_ids=None, settings=settings, store=store, with_vectors=use_mmr,
    )
    if not candidates:
        return []

    pool_size = settings.mmr_pool_size if use_mmr else k
    reranked_pool = await rerank_candidates(settings.reranker_model_name, question, candidates, pool_size)

    if use_mmr and len(reranked_pool) > k:
        mmr_indices = mmr_select(
            relevance_scores=[score for _point, score in reranked_pool],
            vectors=[point.vector["dense"] for point, _score in reranked_pool],
            k=k,
            lambda_mult=settings.mmr_lambda,
        )
        top = [reranked_pool[i] for i in mmr_indices]
    else:
        top = reranked_pool[:k]

    return [point.payload["filename"] for point, _score in top]


async def compute_hit_rate_at_k(
    questions: list[dict], k: int, use_mmr: bool, settings: Settings, store: QdrantStore
) -> tuple[float, list[dict]]:
    """Returns (hit_rate, per_question_detail). `per_question_detail` records
    a hit/miss plus the actual top-k filenames returned for each question --
    the evidence Week 4's mentor review wants for "which failures did the
    change not fix."
    """
    detail = []
    for q in questions:
        filenames = await _top_k_filenames(q["question"], k, use_mmr, settings, store)
        hit = any(f in filenames for f in q["expected_filenames"])
        detail.append({"id": q["id"], "question": q["question"], "expected": q["expected_filenames"], "actual_top_k": filenames, "hit": hit})

    hit_rate = sum(d["hit"] for d in detail) / len(detail) if detail else 0.0
    return hit_rate, detail
