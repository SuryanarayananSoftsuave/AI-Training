# RAG App — Customer Support Knowledge Base

A retrieval-augmented generation app: upload PDF documents, ask questions about them, get
grounded answers with citations and an LLM-judge confidence score.

**Stack**
| Component | Tech | Port |
|---|---|---|
| Backend API | FastAPI (Python) | `8000` |
| Frontend UI | Streamlit (Python) | `8501` |
| Vector DB | Qdrant (Docker) | `6333` (HTTP), `6334` (gRPC) |
| LLM providers | Google Gemini and/or Groq (selectable per request) | — |
| Embedding / rerank | `Qwen/Qwen3-Embedding-0.6B`, `BAAI/bge-reranker-v2-m3`, `Qdrant/bm25` (downloaded from Hugging Face on first backend start) | — |

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10 – 3.12** | Built and tested on 3.11. Get it from [python.org](https://www.python.org/downloads/) — check "Add python.exe to PATH" during install. |
| **Docker Desktop** | Runs Qdrant. Get it from [docker.com](https://www.docker.com/products/docker-desktop/). Must be started/running before setup. |
| **Internet access** | Needed for `pip install` (torch is a large download), first-run model downloads from Hugging Face, and live Gemini/Groq API calls. |
| **A Gemini API key** (required) | Free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). |
| **A Groq API key** (optional) | Only needed if you want the Groq model option in the UI. Free at [console.groq.com/keys](https://console.groq.com/keys). |
| **~3–5 GB free disk** | Python deps (incl. torch) + downloaded ML models. |

---

## 2. Quick start (automated, one click)

1. **Install Docker Desktop** and **Python** if you don't have them (see prerequisites), and make sure Docker Desktop is running.
2. Double-click **`setup.bat`** in the `rag-app` folder.
   - This creates two Python virtual environments (`backend\.venv`, `frontend\.venv`), installs all dependencies, creates `backend\.env` from the template, and starts Qdrant via `docker compose up -d`.
   - First run takes several minutes (torch is a big download). Watch the window for progress.
3. Open **`backend\.env`** in a text editor and set:
   ```
   GEMINI_API_KEY=your_actual_key_here
   ```
   (add `GROQ_API_KEY` too if you want the Groq option).
4. Double-click **`start.bat`**.
   - This opens two new windows (backend + frontend) and makes sure Qdrant is running.
   - On first start, the backend downloads the embedding/reranker models — wait for `startup complete` in the backend window.
5. Open **http://localhost:8501** in your browser.

That's it. Re-running `setup.bat` later is safe — it reuses existing venvs/`.env` and just re-syncs dependencies.

### Script options (PowerShell, if you don't want the `.bat` wrappers)

```powershell
cd rag-app
.\setup.ps1                 # install everything + start Qdrant
.\setup.ps1 -SkipDocker     # skip starting Qdrant (e.g. you run it elsewhere / already running)
.\setup.ps1 -Run            # setup, then immediately launch backend + frontend too
.\start.ps1                 # start Qdrant + backend + frontend (after setup has run once)
```

> If PowerShell blocks the scripts with an execution-policy error, either use the `.bat`
> wrappers (they bypass the policy for that one run) or run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

## 3. Manual setup (step by step)

If you'd rather not use the scripts, or want to understand exactly what they do:

### 3.1 Start Qdrant

```powershell
cd rag-app
docker compose up -d
```

Verify it's up: http://localhost:6333/dashboard

### 3.2 Backend

```powershell
cd rag-app\backend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

copy .env.example .env
notepad .env        # set GEMINI_API_KEY (required), GROQ_API_KEY (optional)

$env:PYTHONPATH = "$PWD"
.\.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait for `startup complete` in the log — the first start downloads the embedding and
reranker models from Hugging Face, which can take a few minutes. Verify at
http://localhost:8000/docs (Swagger UI) or http://localhost:8000/health.

> `PYTHONPATH` must include `backend\` so `app.main:app` resolves — the app is run with `backend\` as its working directory, not from the repo root.

### 3.3 Frontend

In a **new** terminal (leave the backend running):

```powershell
cd rag-app\frontend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run streamlit_app.py
```

Open http://localhost:8501.

By default the frontend talks to the backend at `http://localhost:8000`. Override with an
env var if needed: `$env:BACKEND_URL = "http://localhost:8000"` before launching Streamlit.

### 3.4 Try it

1. Upload a PDF (e.g. one from `sample_data/customer_support_kb/`).
2. Wait for its status to become `indexed`.
3. Ask a question about its contents in the chat box.

---

## 4. Configuration reference (`backend/.env`)

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | *(empty)* | Required to use the Gemini generator/judge. |
| `GEMINI_GENERATOR_MODEL` | `gemini-3.6-flash` | Model used to generate answers. |
| `GEMINI_JUDGE_MODEL` | `gemini-3.5-flash` | Model used to grade groundedness of answers. |
| `GROQ_API_KEY` | *(empty)* | Only needed if you select Groq as generator/judge in the UI. |
| `GROQ_GENERATOR_MODEL` | `llama-3.1-70b-versatile` | Groq generator model. |
| `GROQ_JUDGE_MODEL` | `llama-3.1-8b-instant` | Groq judge model. |
| `QDRANT_URL` | `http://localhost:6333` | Where the backend reaches Qdrant. |
| `QDRANT_COLLECTION` | `rag_documents` | Collection name for vectors. |
| `EMBEDDING_MODEL_NAME` | `Qwen/Qwen3-Embedding-0.6B` | Dense embedding model (downloaded on first run). |
| `EMBEDDING_DIM` | `1024` | Must match the embedding model's output dimension. |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker. |
| `SPARSE_MODEL_NAME` | `Qdrant/bm25` | Sparse/keyword model for hybrid search. |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `500` / `75` | Document chunking parameters. |
| `FIRST_STAGE_LIMIT` | `40` | Candidates pulled before reranking. |
| `RERANK_TOP_K` | `6` | Candidates kept after reranking. |
| `QUERY_EXPANSION_VARIANT_COUNT` | `3` | Alternate phrasings generated when multi-query expansion is enabled. |
| `QUERY_EXPANSION_SIMILARITY_THRESHOLD` | `0.80` | Minimum cosine similarity for a generated variant to be kept. |
| `OFF_TOPIC_SCORE_THRESHOLD` | `0.15` | Below this best-rerank-score, a question is treated as out-of-scope. |
| `DATA_DIR` / `UPLOADS_DIR` / `REGISTRY_PATH` | `data`, `data/uploads`, `data/registry/documents.json` | Local storage paths (relative to `backend/`). |

`backend/.env` is **git-ignored** — it holds live API keys and must never be committed.
Only `backend/.env.example` (no real secrets) is tracked in git.

---

## 5. Troubleshooting

- **"Docker is installed but the engine isn't running"** — open Docker Desktop and wait for it to fully start, then re-run `setup.bat`/`docker compose up -d`.
- **`pip install` fails partway through installing torch** — a known transient issue on Windows; re-run `setup.bat` (the script auto-retries once with `--no-cache-dir`), or manually: `.\.venv\Scripts\pip install --no-cache-dir -r requirements.txt`.
- **Backend fails with `ModuleNotFoundError: No module named 'app'`** — you must run uvicorn from inside `backend\` with `PYTHONPATH` set to that folder (the scripts handle this automatically).
- **Gemini call fails with a 404 "model not found"** — the exact model names available differ per API key/tier and change over time; check https://aistudio.google.com for models available to your account and update `GEMINI_GENERATOR_MODEL`/`GEMINI_JUDGE_MODEL` in `backend/.env`.
- **Gemini call fails with 429 (quota)** — free-tier keys often have zero quota for `-pro` models; stick to flash-tier models, or reduce usage (query expansion adds one extra Gemini call per chat message).
- **First backend startup is slow / looks stuck** — it's downloading the embedding + reranker models (a few hundred MB) from Hugging Face; watch the log for `loading embedding model ...` / `loading reranker model ...` then `startup complete`.
- **Port already in use (8000 / 8501 / 6333)** — stop whatever else is using it, or change the port in the run command (backend: `--port`, frontend: `--server.port`) and update `BACKEND_URL` accordingly.
- **Upload gets stuck at `processing`** — a backend crash mid-ingestion leaves this; just restart the backend (it auto-resets stuck `processing` records to `failed` on startup) and re-upload the same file to retry.

---

## 6. Stopping everything

```powershell
# Stop backend / frontend: close their PowerShell windows, or Ctrl+C in each.

# Stop Qdrant:
cd rag-app
docker compose down       # keeps data in .\qdrant_storage
```

---

## 7. Project layout

```
rag-app/
├── setup.bat / setup.ps1   # one-click install (venvs, pip, .env, Qdrant)
├── start.bat  / start.ps1  # one-click launch (Qdrant + backend + frontend)
├── docker-compose.yml      # Qdrant service definition
├── backend/
│   ├── app/                # FastAPI app (api / core / ingestion / retrieval / llm / services)
│   ├── requirements.txt
│   └── .env.example        # template — copy to .env and fill in real keys
├── frontend/
│   ├── streamlit_app.py
│   └── requirements.txt
└── sample_data/            # sample PDFs to try the app with
```
