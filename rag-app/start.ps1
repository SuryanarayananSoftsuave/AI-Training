<#
.SYNOPSIS
    Starts Qdrant (Docker), the FastAPI backend, and the Streamlit frontend.
    Run .\setup.ps1 first if you haven't already.

.USAGE
    .\start.ps1
#>

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }

if (-not (Test-Path "backend\.venv\Scripts\python.exe")) { throw "backend\.venv not found. Run .\setup.ps1 first." }
if (-not (Test-Path "frontend\.venv\Scripts\python.exe")) { throw "frontend\.venv not found. Run .\setup.ps1 first." }
if (-not (Test-Path "backend\.env")) { throw "backend\.env not found. Run .\setup.ps1 first, then add your API key(s)." }

Write-Step "Qdrant (vector database)"
$dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerCmd) {
    docker compose up -d
    Write-Ok "Qdrant running - http://localhost:6333/dashboard"
} else {
    Write-Warn "Docker not found on PATH - make sure Qdrant is reachable at the QDRANT_URL set in backend\.env"
}

Write-Step "Starting backend (FastAPI, http://localhost:8000)"
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root\backend'; `$env:PYTHONPATH='$root\backend'; & '.\.venv\Scripts\python.exe' -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
)

Write-Ok "waiting a couple seconds for the backend to start loading models..."
Start-Sleep -Seconds 3

Write-Step "Starting frontend (Streamlit, http://localhost:8501)"
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root\frontend'; & '.\.venv\Scripts\python.exe' -m streamlit run streamlit_app.py"
)

Write-Host ""
Write-Host "Two new PowerShell windows were opened (backend + frontend) - watch them for logs / errors." -ForegroundColor Cyan
Write-Host "  Backend API docs:  http://localhost:8000/docs" -ForegroundColor Green
Write-Host "  Frontend UI:       http://localhost:8501" -ForegroundColor Green
Write-Host "  Qdrant dashboard:  http://localhost:6333/dashboard" -ForegroundColor Green
Write-Host ""
Write-Host "Note: on first run the backend has to download the embedding + reranker models from Hugging Face - this can take a few minutes. Watch the backend window for 'startup complete'." -ForegroundColor Yellow
