<#
.SYNOPSIS
    One-click setup for the RAG app: Python venvs, pip installs, .env bootstrap, Qdrant via Docker.

.USAGE
    Right-click setup.bat -> Run (easiest), or from PowerShell:
        .\setup.ps1
        .\setup.ps1 -SkipDocker      # skip starting Qdrant (e.g. you run it elsewhere)
        .\setup.ps1 -Run             # also launch backend + frontend when done
#>

param(
    [switch]$SkipDocker,
    [switch]$Run
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "    $msg" -ForegroundColor Red }

function Install-Requirements($venvPython, $requirementsFile, $label) {
    Write-Step "$label`: installing dependencies (this can take several minutes on first run - torch + ML models)"
    & $venvPython -m pip install --upgrade pip | Out-Null
    & $venvPython -m pip install -r $requirementsFile
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Install failed, retrying once with --no-cache-dir (known transient issue on Windows for large wheels like torch)..."
        & $venvPython -m pip install --no-cache-dir -r $requirementsFile
        if ($LASTEXITCODE -ne 0) {
            throw "$label dependency install failed. See the error above."
        }
    }
    Write-Ok "$label dependencies installed"
}

# --- 1. Python ------------------------------------------------------------
Write-Step "Checking Python"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python not found on PATH. Install Python 3.11 from https://www.python.org/downloads/ (check 'Add python.exe to PATH' during install) and re-run this script."
}
$verOutput = (& python --version) 2>&1
Write-Ok "$verOutput found at $($python.Source)"
if ($verOutput -match "Python (\d+)\.(\d+)") {
    $maj = [int]$Matches[1]; $min = [int]$Matches[2]
    if ($maj -ne 3 -or $min -lt 10 -or $min -gt 12) {
        Write-Warn "This project was built and tested on Python 3.11. Python $maj.$min may hit dependency issues (torch / fastembed wheel availability). 3.10-3.12 is the safe range."
    }
}

# --- 2. Backend venv + deps ------------------------------------------------
Write-Step "Backend: virtual environment"
if (-not (Test-Path "backend\.venv\Scripts\python.exe")) {
    python -m venv backend\.venv
    Write-Ok "created backend\.venv"
} else {
    Write-Ok "backend\.venv already exists, reusing it"
}
Install-Requirements "backend\.venv\Scripts\python.exe" "backend\requirements.txt" "Backend"

# --- 3. Frontend venv + deps -------------------------------------------------
Write-Step "Frontend: virtual environment"
if (-not (Test-Path "frontend\.venv\Scripts\python.exe")) {
    python -m venv frontend\.venv
    Write-Ok "created frontend\.venv"
} else {
    Write-Ok "frontend\.venv already exists, reusing it"
}
Install-Requirements "frontend\.venv\Scripts\python.exe" "frontend\requirements.txt" "Frontend"

# --- 4. .env bootstrap -------------------------------------------------------
Write-Step "Backend: environment file"
$envNeedsKeys = $false
if (-not (Test-Path "backend\.env")) {
    Copy-Item "backend\.env.example" "backend\.env"
    Write-Warn "Created backend\.env from backend\.env.example."
    $envNeedsKeys = $true
} else {
    Write-Ok "backend\.env already exists, leaving it untouched"
    $content = Get-Content "backend\.env" -Raw
    if ($content -match "GEMINI_API_KEY=your_gemini_api_key_here|GEMINI_API_KEY=\s*$") {
        $envNeedsKeys = $true
    }
}

# --- 5. Docker / Qdrant ------------------------------------------------------
$qdrantStarted = $false
if (-not $SkipDocker) {
    Write-Step "Checking Docker"
    $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $dockerCmd) {
        Write-Warn "Docker not found on PATH. Install Docker Desktop from https://www.docker.com/products/docker-desktop, start it, then re-run this script (or run 'docker compose up -d' yourself once it's installed)."
    } else {
        docker info *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Docker CLI found but the Docker Desktop engine isn't running. Start Docker Desktop and re-run this script, or run 'docker compose up -d' manually once it's up."
        } else {
            Write-Step "Starting Qdrant (vector database) via docker compose"
            docker compose up -d
            if ($LASTEXITCODE -eq 0) {
                Write-Ok "Qdrant running - dashboard: http://localhost:6333/dashboard"
                $qdrantStarted = $true
            } else {
                Write-Err "docker compose up -d failed - see output above."
            }
        }
    }
} else {
    Write-Warn "Skipping Docker/Qdrant startup (-SkipDocker passed)."
}

# --- Summary -----------------------------------------------------------------
Write-Step "Setup complete"
Write-Host ""
if ($envNeedsKeys) {
    Write-Warn "ACTION NEEDED: open backend\.env and set GEMINI_API_KEY (required) and GROQ_API_KEY (optional, only if you want the Groq model option)."
    Write-Host "  Get a Gemini key:  https://aistudio.google.com/apikey"
    Write-Host "  Get a Groq key:    https://console.groq.com/keys"
    Write-Host ""
}
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Make sure backend\.env has a valid GEMINI_API_KEY."
Write-Host "  2. Run .\start.ps1 (or double-click start.bat) to launch the backend + frontend."
Write-Host ""
Write-Host "  Backend docs (once started):  http://localhost:8000/docs"
Write-Host "  Frontend (once started):      http://localhost:8501"
Write-Host "  Qdrant dashboard:             http://localhost:6333/dashboard"
Write-Host ""

if ($Run) {
    & "$root\start.ps1"
}
