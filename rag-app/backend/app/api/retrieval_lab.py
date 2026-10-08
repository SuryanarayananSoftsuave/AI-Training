"""Week 3/4 showcase: retrieval-only comparisons with no LLM cost.

Reuses the exact same retrieve_candidates/rerank_candidates/mmr_select
functions chat_service.answer_query calls for real chat traffic, and the
exact same chunk_page function ingestion_service uses at upload time --
nothing here is a reimplementation, only new orchestration/response shaping
around existing pipeline code.
"""
from __future__ import annotations

import json
from pathlib import Path

import tiktoken
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, model_validator

from app.core.config import Settings, get_settings
from app.core.dependencies import get_store
from app.ingestion.chunker import Chunk, chunk_page
from app.retrieval.mmr import mmr_select
from app.retrieval.pipeline import rerank_candidates, retrieve_candidates
from app.retrieval.qdrant_store import QdrantStore

router = APIRouter(prefix="/retrieval-lab", tags=["retrieval-lab"])

_ENCODING = tiktoken.get_encoding("cl100k_base")  # same encoder chunker.py itself uses
# Written by scripts/run_hit_rate_eval.py -- read-only here, never regenerated
# or fabricated by this endpoint.
_HIT_RATE_RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "evals" / "hit_rate_results.json"


@router.get("/hit-rate-results")
async def hit_rate_results() -> dict | None:
    """The real Week 4 hit-rate@3 numbers, straight off disk, exactly like
    /agents/results already does for Week 7's deliverables. Returns None
    (not an error) when the eval hasn't been run yet -- the UI shows setup
    instructions in that case instead of a blank error.
    """
    if not _HIT_RATE_RESULTS_PATH.exists():
        return None
    return json.loads(_HIT_RATE_RESULTS_PATH.read_text(encoding="utf-8"))


class RetrievedChunkOut(BaseModel):
    filename: str
    page_number: int | None
    section_heading: str | None
    rerank_score: float
    snippet: str


class RetrievalVariant(BaseModel):
    label: str
    chunks: list[RetrievedChunkOut]


class CompareRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=15)
    doc_ids: list[str] | None = None


class CompareResponse(BaseModel):
    variants: list[RetrievalVariant]  # order: semantic (no MMR), semantic+MMR, hybrid (no MMR), hybrid+MMR


def _to_chunk_out(point, score: float) -> RetrievedChunkOut:
    text = point.payload.get("text", "")
    return RetrievedChunkOut(
        filename=point.payload["filename"],
        page_number=point.payload.get("page_number"),
        section_heading=point.payload.get("section_heading"),
        rerank_score=round(score, 4),
        snippet=text[:280],
    )


async def _one_mode(
    label_prefix: str, use_keyword_search: bool, request: CompareRequest, settings: Settings, store: QdrantStore,
) -> list[RetrievalVariant]:
    """One retrieve+rerank cycle for this search mode, reused for BOTH the
    plain-top-k and MMR variant -- mirrors how answer_query itself computes
    one reranked pool per request and only optionally re-selects from it,
    rather than re-retrieving per toggle.
    """
    candidates, _total_raw = await retrieve_candidates(
        [request.query], use_keyword_search, settings.first_stage_limit, request.doc_ids, settings, store, with_vectors=True,
    )
    if not candidates:
        return [
            RetrievalVariant(label=f"{label_prefix} (no MMR)", chunks=[]),
            RetrievalVariant(label=f"{label_prefix} + MMR", chunks=[]),
        ]

    pool_size = max(settings.mmr_pool_size, request.top_k)
    reranked_pool = await rerank_candidates(settings.reranker_model_name, request.query, candidates, pool_size)

    plain = reranked_pool[: request.top_k]
    if len(reranked_pool) > request.top_k:
        mmr_indices = mmr_select(
            relevance_scores=[score for _point, score in reranked_pool],
            vectors=[point.vector["dense"] for point, _score in reranked_pool],
            k=request.top_k,
            lambda_mult=settings.mmr_lambda,
        )
        mmr_chosen = [reranked_pool[i] for i in mmr_indices]
    else:
        mmr_chosen = plain

    return [
        RetrievalVariant(label=f"{label_prefix} (no MMR)", chunks=[_to_chunk_out(p, s) for p, s in plain]),
        RetrievalVariant(label=f"{label_prefix} + MMR", chunks=[_to_chunk_out(p, s) for p, s in mmr_chosen]),
    ]


@router.post("/compare", response_model=CompareResponse)
async def compare(
    request: CompareRequest, settings: Settings = Depends(get_settings), store: QdrantStore = Depends(get_store),
) -> CompareResponse:
    """4-way retrieval comparison on one question: semantic vs hybrid search,
    each with and without MMR re-selection. No generation or judging --
    fast and free of LLM cost, so it's safe to click repeatedly.
    """
    semantic_variants = await _one_mode("Semantic", False, request, settings, store)
    hybrid_variants = await _one_mode("Hybrid", True, request, settings, store)
    return CompareResponse(variants=semantic_variants + hybrid_variants)


class ChunkOut(BaseModel):
    text: str
    section_heading: str | None
    content_type: str
    token_count: int


class ChunkPreviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    chunk_size_a: int = Field(default=500, ge=20, le=4000)
    overlap_a: int = Field(default=75, ge=0, le=1000)
    chunk_size_b: int = Field(default=200, ge=20, le=4000)
    overlap_b: int = Field(default=30, ge=0, le=1000)

    @model_validator(mode="after")
    def _overlap_must_be_smaller_than_chunk_size(self) -> "ChunkPreviewRequest":
        # RecursiveCharacterTextSplitter raises a bare ValueError (-> an
        # unhandled 500) when overlap >= chunk size; the UI's sliders allow
        # exactly this (size down to 20, overlap up to 500/1000
        # independently), so this must be caught as a clean 422 instead.
        if self.overlap_a >= self.chunk_size_a:
            raise ValueError(f"overlap_a ({self.overlap_a}) must be smaller than chunk_size_a ({self.chunk_size_a})")
        if self.overlap_b >= self.chunk_size_b:
            raise ValueError(f"overlap_b ({self.overlap_b}) must be smaller than chunk_size_b ({self.chunk_size_b})")
        return self


class ChunkPreviewResponse(BaseModel):
    chunks_a: list[ChunkOut]
    chunks_b: list[ChunkOut]


def _to_chunk_preview_out(c: Chunk) -> ChunkOut:
    return ChunkOut(
        text=c.text, section_heading=c.section_heading, content_type=c.content_type,
        token_count=len(_ENCODING.encode(c.text)),
    )


@router.post("/chunk-preview", response_model=ChunkPreviewResponse)
async def chunk_preview(request: ChunkPreviewRequest) -> ChunkPreviewResponse:
    """Week 3 showcase: chunk the SAME text at two different (size, overlap)
    settings side by side -- a live, repeatable answer to the still-open
    mentor-check item "did they try more than one chunk size and notice the
    difference?". Pure function call on `chunk_page` (no Qdrant, no LLM, no
    re-ingestion of the live index), so it works even with Docker down.
    """
    chunks_a = chunk_page(request.text, page_number=1, chunk_size_tokens=request.chunk_size_a, chunk_overlap_tokens=request.overlap_a)
    chunks_b = chunk_page(request.text, page_number=1, chunk_size_tokens=request.chunk_size_b, chunk_overlap_tokens=request.overlap_b)
    return ChunkPreviewResponse(
        chunks_a=[_to_chunk_preview_out(c) for c in chunks_a], chunks_b=[_to_chunk_preview_out(c) for c in chunks_b],
    )
