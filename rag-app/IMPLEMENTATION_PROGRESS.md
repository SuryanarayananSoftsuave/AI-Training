# Implementation Progress

Tracking build-out of the plan in `../RAG_IMPLEMENTATION_PLAN.md`. Check items off as they land; note blockers inline.

## Backend — foundation
- [x] `app/core/config.py` — Settings (env-driven)
- [x] `app/core/dependencies.py` — FastAPI dependency accessors
- [x] `app/models/schemas.py` — Pydantic request/response models

## Backend — registry
- [x] `app/registry/json_store.py` — hash-keyed JSON registry, filelock + atomic writes

## Backend — ingestion
- [x] `app/ingestion/parser.py` — PyMuPDF4LLM per-page parsing
- [x] `app/ingestion/chunker.py` — header split + token split + table isolation
- [x] `app/ingestion/embedder.py` — Qwen3-Embedding-0.6B (dense)
- [x] `app/ingestion/sparse.py` — FastEmbed BM25 (sparse)

## Backend — retrieval
- [x] `app/retrieval/qdrant_store.py` — collection mgmt, upsert, hybrid query
- [x] `app/retrieval/reranker.py` — bge-reranker-v2-m3 cross-encoder

## Backend — LLM
- [x] `app/llm/prompts.py`
- [x] `app/llm/gemini_client.py` — generate + judge, structured output

## Backend — services
- [x] `app/services/ingestion_service.py`
- [x] `app/services/chat_service.py`

## Backend — API
- [x] `app/api/documents.py` — upload / list / get / delete
- [x] `app/api/chat.py`
- [x] `app/api/health.py`
- [x] `app/main.py` — lifespan startup, CORS, error envelopes, router mounting

## Backend — packaging
- [x] `requirements.txt`
- [x] `.env.example`

## Frontend
- [x] `streamlit_app.py`
- [x] `utils/api_client.py`
- [x] `requirements.txt`

## Infra / test data
- [x] `docker-compose.yml` (Qdrant)
- [x] `.gitignore`
- [x] `scripts/generate_sample_pdf.py`
- [x] generated sample PDF (`sample_data/acme_robotics_handbook.pdf`)

## Environment & run
- [x] Docker Desktop engine up, Qdrant container running (`docker compose up -d`)
- [x] Backend venv created, dependencies installed (first attempt hit a transient Windows OSError mid-torch-install; retry with `--no-cache-dir` succeeded)
- [x] Frontend venv created, dependencies installed
- [x] `GEMINI_API_KEY` set in `backend/.env` (user-provided)
- [x] Qdrant collection created (`rag_documents`, dense+sparse, 5 payload indexes)
- [x] FastAPI server running (`uvicorn`), `/health` green
- [x] Streamlit server running (`localhost:8501`)

## Gemini model availability (discovered live, not assumed)
This API key/account rejects several models the plan originally assumed:
- `gemini-2.5-flash` and `gemini-2.5-pro` → 404 "no longer available to new users"
- `gemini-1.5-flash` → doesn't exist in this key's model catalog at all anymore
- `gemini-3.1-pro-preview` → 429, free tier grants **0 quota** to any `-pro` model
- **Working config**: generator = `gemini-3.6-flash`, judge = `gemini-3.5-flash` (two distinct flash-tier models — free tier has no pro-tier quota at all, so tier separation is by model generation/version rather than flash-vs-pro)
- Updated in `app/core/config.py` defaults, `.env.example`, and the user's `backend/.env`

## End-to-end verification (all tested live against the running app)
- [x] Sample PDF uploaded, ingested, status `indexed` (3 pages, 7 chunks)
- [x] Re-uploading the same PDF is detected as a duplicate (hash match, no re-ingestion)
- [x] Chat works with keyword search **off** (semantic-only) — verified via API and browser
- [x] Chat works with keyword search **on** (hybrid RRF fusion) — verified via API and browser, `retrieval_debug.mode` flips correctly
- [x] Citations map to real chunks (filename + page + snippet + rerank score) — verified in Sources panel
- [x] LLM-judge confidence score renders (`Grounded — 100%`), with per-claim evidence-quote verification working
- [x] Metadata filter (scope to one document) works — valid `doc_ids` scopes correctly; a bogus id correctly returns zero candidates and a graceful "no answer" response

- [x] `DELETE /documents/{id}` exercised live via the Streamlit UI (204 No Content, confirmed in backend log)

## VS Code debugging
- [x] `.vscode/launch.json` at the true workspace root (`D:\surya\AI Training`, not `rag-app`) — configs: `FastAPI: Debug Backend`, `Streamlit: Debug Frontend`, and compound `Full Stack: Backend + Frontend`
- Note: `--reload` deliberately omitted from the backend config — uvicorn's reloader subprocess doesn't reliably get debugger breakpoints

## Reliability fixes (found via real use, not anticipated in the original plan)
- **No ingestion progress logging** — `run_ingestion` only logged at the very end (success/failure), so a slow upload was indistinguishable from a stuck one. Fixed: `ingestion_service.py` now logs parse/chunk/embed(dense)/embed(sparse)/upsert, each with counts and timing, plus a final total.
- **Concurrent uploads thrash the CPU** — every upload's BackgroundTask ran immediately, so N simultaneous uploads all fought over the same cores for embedding at once and collectively crawled (discovered when 6 KB-article PDFs uploaded together all sat unfinished for 5+ minutes). Fixed: a process-wide lock in `ingestion_service.py` now serializes the heavy part of ingestion — uploads queue and process one at a time.
- **A stuck/failed document could never be retried** — dedup-by-hash returned the same stuck `processing`/`failed` record forever on re-upload, never re-triggering ingestion. Fixed: `documents.py` now only short-circuits as a duplicate when the existing record is truly `indexed` or still legitimately `processing`; a `failed` record is retried in place. `main.py` also resets any `processing` record to `failed` on startup (it cannot have survived a process restart), so a crash never leaves a permanent silent-stuck entry.
- **`/chat` had no pipeline visibility** — only the FastAPI access log line (`POST /chat 200 OK`) existed; nothing showed the actual query, retrieval branch, rerank scores, generated answer, or judge verdict. Fixed: `chat_service.py` now logs each step (embed query → retrieve → rerank → generate → judge) with timing and counts, e.g. `chat query: 'What warranty prefix...' (mode=hybrid, top_k=5)`, `reranked 14 -> top 5: [(filename, page, score), ...]`, `generated answer (1.8s, 142 chars, cited markers=[1, 2]): '...'`, `judged answer (2.1s): verdict=grounded confidence=100 (3/3 claims supported)`. The full prompt/context text sent to Gemini is logged at `DEBUG` level (not shown by default — bump `logging.basicConfig(level=...)` in `main.py` to `DEBUG` to see it).
- **The `md_to_pdf.py` test-data converter corrupted non-ASCII punctuation** — ReportLab's base fonts don't cover Unicode characters like the non-breaking hyphen (‑) or rightwards arrow (→) used in the source KB articles, rendering them as black box glyphs; some WinAnsi-covered punctuation (en dash, curly quotes) rendered fine on screen but came back as U+FFFD when `pymupdf4llm` extracted text from the PDF (no ToUnicode CMap), silently corrupting the text that would get embedded. Fixed: every non-ASCII typographic character is now normalized to a plain-ASCII equivalent before layout.

## Feature: multi-query expansion (added post-plan, user-requested)
- New toggle in the UI, independent of the hybrid/semantic toggle: **"Enable query expansion (multi-query)"**.
- When on: `GeminiClient.expand_query()` generates 3 alternate phrasings of the question (structured output, same generator model). The original question + all 3 variants are retrieved **in parallel** via a `ThreadPoolExecutor` in `chat_service.py` (`_retrieve_for_query`, one thread per query text) — each doing its own embed + Qdrant search respecting the existing hybrid/semantic toggle.
- Results are merged and **deduped by Qdrant point ID**, then reranked as one pool against the **original** question only (never the paraphrases — reranking must measure relevance to what the user actually asked, not to a rewritten version of it).
- `RetrievalDebug.query_variants` surfaces the generated phrasings; the Streamlit UI shows them in a "Query expansion (N variants)" expander under each answer.
- **Same-meaning enforcement (not just a prompt instruction):** the prompt asks the model to preserve the original question's intent, but that alone isn't trusted — `_filter_variants_by_similarity()` in `chat_service.py` embeds the original question and every variant (via the same embedding model, no extra API cost) and drops any variant whose cosine similarity to the original falls below `QUERY_EXPANSION_SIMILARITY_THRESHOLD` (default `0.80`), logging a warning naming the dropped variant and its score. If every variant drifts, retrieval just falls back to the original question alone — never a hard failure.
- Known cost, called out to the user before building: +1 Gemini call per chat message (real pressure on an already-tight free-tier quota — see the model-availability notes above), and up to 4x the retrieval/embedding calls per question (parallelized, so wall-clock cost is closer to the slowest single query than 4x, but not free).
- [ ] Not yet live-tested end-to-end (implemented this session, pending a restart + manual test)

## Feature: off-topic / out-of-scope guardrail (added post-plan, user-requested)
- Vector search always returns its nearest neighbors, even for a totally unrelated question — it never returns "nothing," it just returns the least-bad match. Previously the only defense against an off-topic question was the generation prompt's instruction to say "I don't know" — a soft ask, not an enforced rule.
- Now: after reranking, if the **best** rerank score across all candidates is below `OFF_TOPIC_SCORE_THRESHOLD` (default `0.15`), the query is treated as out-of-scope for this knowledge base — `chat_service.py` returns a fixed response ("...please contact an administrator or your support team...") with `judgment.verdict = "out_of_scope"`, **without calling Gemini at all** (skips both `generate_answer` and `judge_answer` — a real cost saving on clearly off-topic questions, not just a UX improvement).
- Surfaced in the UI as a `🚫 Out of scope` badge (`_VERDICT_STYLE` in `streamlit_app.py`).
- `0.15` is a starting default, not empirically tuned — flagged the same way as the other thresholds in this app (chunk overlap %, query-expansion similarity): validate against real queries and adjust `OFF_TOPIC_SCORE_THRESHOLD` in `.env` if it's too strict (rejecting valid questions) or too lax (letting unrelated ones through).
- **Gap found via real use, fixed same session:** "tell about the movie VIP" scored just above the threshold (some chunk was coincidentally similar enough), so it reached the generator, which correctly declined ("the provided context does not contain information about the movie VIP") -- but the judge then marked that decline **"Grounded"** (technically correct: the claim "context doesn't cover X" is true), which reads to the user as a successful green answer, not a rejection. Fixed with a second, more reliable gate: after generation, if `used_source_indices` is empty (the generator itself cited zero retrieved chunks), that's treated as out-of-scope too -- skips the judge call and returns the same fixed contact-support message instead of showing the model's own decline text under a misleadingly positive badge. Refactored the out-of-scope response into one shared `_out_of_scope_response()` helper used by both gates.
- [ ] Not yet live-tested end-to-end (implemented this session, pending a restart + manual test)

## Feature: Groq as a second LLM provider, selectable per-request (added post-plan, user-requested)
- New `LLMClient` Protocol (`app/llm/base.py`) — `expand_query` / `generate_answer` / `judge_answer` — implemented by both `GeminiClient` and the new `GroqClient` (`app/llm/groq_client.py`). Shared judge-verification math (quote-checking + confidence recompute) factored into `app/llm/verification.py` so both clients use identical logic, not a copy each.
- Both clients are constructed once at startup (`main.py` → `app.state.llm_clients = {"gemini": ..., "groq": ...}`) and looked up per-request in `api/chat.py` based on two new independent `ChatRequest` fields: `generator_provider` and `judge_provider` (`Literal["gemini", "groq"]`, both default `"gemini"` so existing behavior is unchanged unless you actively pick Groq).
- New Streamlit **"Model settings"** expander: two selectboxes, **Generator LLM** and **Judge LLM**, each independently "Gemini" or "Groq" — matching the user's ask to control both, not just the generator. Selected providers show in the debug caption under each answer (`generator: groq · judge: gemini`).
- Groq models (verified live against `console.groq.com/docs/models` on 2026-08-24, not assumed from memory — Groq had already deprecated the model names memory would have suggested): generator = `llama-3.1-70b-versatile`, judge = `llama-3.1-8b-instant`. Structured output uses Groq's `response_format: {"type": "json_schema", "strict": true}` — schemas are hand-written flat JSON (not derived from Pydantic's `$ref`-based auto schema), since strict-mode support for nested refs varies by provider and wasn't worth risking.
- Needs `GROQ_API_KEY` in `backend/.env` (user-provided) — the Groq option simply won't work until that's added, same pattern as the Gemini key.
- `groq` Python package (`groq-1.6.0`) installed into the backend venv — all its dependencies (httpx, pydantic, etc.) were already present, no new heavy dependency chain.
- [ ] Not yet live-tested end-to-end (implemented this session, pending `GROQ_API_KEY`, a restart, and a manual test)

## Feature: per-request temperature control from the UI (added post-plan, user-requested)
- Found via question, not a bug report: neither `GeminiClient` nor `GroqClient` set `temperature` anywhere — every call ran on each provider's unspecified default (~1.0, tuned for varied/creative output, not factual consistency).
- Fixed by making temperature a real parameter threaded through every layer, not a fixed constant: `LLMClient` Protocol methods (`expand_query`/`generate_answer`/`judge_answer`) now take `temperature: float`; both clients pass it to their respective API calls (`GenerateContentConfig(temperature=...)` for Gemini, `temperature=...` on Groq's `chat.completions.create`).
- Two independent sliders in the Streamlit "Model settings" panel: **Generator temperature** (default `0.2`) and **Judge temperature** (default `0.0`) — matches the reasoning given when the user asked about this: the generator should stick close to retrieved context, and the judge should be near-deterministic since a fact-checker that changes its verdict on repeated runs of the same answer is a weak one. Query expansion reuses the generator's temperature (no separate slider — one less control for a task where a small amount of shared variability is fine).
- `ChatRequest.generator_temperature` / `judge_temperature` (bounded 0.0-2.0) carry the UI's choice through `api/chat.py` -> `chat_service.answer_query()` -> both LLM clients; `RetrievalDebug` echoes back what was actually used (`generator: gemini@0.2 · judge: gemini@0.0` in the debug caption) so it's never silently different from what's shown in the sidebar.
- [ ] Not yet live-tested end-to-end (implemented this session, pending a restart + manual test)

## Known gaps / not yet done
- [ ] No automated test suite (manual/live verification only, per the scope of this session)
- [ ] Phase 6 enhancements (parent-document chunking, Contextual Retrieval blurbs, SQLite registry, ARQ task queue) intentionally not built — documented as future work in the plan
- [ ] Ingestion is now serialized (one at a time) rather than concurrent — correct for a single-CPU dev box, but a real throughput ceiling if bulk-uploading many large documents; the documented Phase 6 move to `arq` + Redis workers is the right fix if that's ever needed
