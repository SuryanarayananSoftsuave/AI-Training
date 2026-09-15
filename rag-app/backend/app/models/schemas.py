from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

LLMProvider = Literal["gemini", "groq"]


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    doc_id: str
    original_filename: str
    stored_path: str
    file_hash: str
    file_size_bytes: int
    upload_timestamp: datetime
    status: DocumentStatus
    num_pages: int | None = None
    num_chunks: int | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    doc_id: str
    status: DocumentStatus
    duplicate_of: str | None = None
    message: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    use_keyword_search: bool = False
    use_query_expansion: bool = False
    use_mmr: bool = False
    generator_provider: LLMProvider = "gemini"
    judge_provider: LLMProvider = "gemini"
    generator_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int = Field(default=6, ge=1, le=20)
    doc_ids: list[str] | None = None


class Citation(BaseModel):
    marker: int
    doc_id: str
    filename: str
    page_number: int | None
    chunk_id: str
    snippet: str
    rerank_score: float


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    evidence_quote: str


class JudgeVerdict(str, Enum):
    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    HALLUCINATED = "hallucinated"
    NO_ANSWER = "no_answer"
    OUT_OF_SCOPE = "out_of_scope"
    JUDGE_UNAVAILABLE = "judge_unavailable"


class Judgment(BaseModel):
    verdict: JudgeVerdict
    confidence: int = Field(ge=0, le=100)
    claims: list[ClaimVerdict]
    notes: str


class RetrievalDebug(BaseModel):
    mode: str  # "hybrid" | "semantic"
    candidates_after_fusion: int
    reranked_returned: int
    query_variants: list[str] | None = None  # the extra phrasings generated, when query expansion is on
    generator_provider: LLMProvider = "gemini"
    judge_provider: LLMProvider = "gemini"
    generator_temperature: float = 0.7
    judge_temperature: float = 0.0
    use_mmr: bool = False


class ChatStreamFinal(BaseModel):
    """The one non-streamed event in a `/chat` SSE response -- sent once,
    after every `delta` chunk of the streamed answer text, carrying
    everything that's only knowable once generation AND judging have both
    finished (or that short-circuits generation/judging entirely, for the
    no-candidates and off-topic gates).
    """

    answer: str
    citations: list[Citation]
    judgment: Judgment
    retrieval_debug: RetrievalDebug


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    collection_exists: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
