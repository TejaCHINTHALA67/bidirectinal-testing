# Complete experiment runner for Bidirectional RAG publication
# Executes 36 experiments: 3 systems x 4 datasets x 3 seeds
# Expected runtime: 24-48 hours

Write-Host "============================================================" -ForegroundColor Green
Write-Host "Starting Complete Experimental Run for Publication" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Configuration:"
Write-Host "- Systems: 3 (Standard RAG, Naive Write-back, Bidirectional RAG)"
Write-Host "- Datasets: 4 (NQ, TriviaQA, HotpotQA, Stack Overflow)"
Write-Host "- Seeds: 3 (42, 43, 44)"
Write-Host "- Total experiments: 36"
Write-Host "- Queries per experiment: 500"
Write-Host "- Estimated time: 24-48 hours"
Write-Host ""

# Change to project directory
Set-Location "D:\bidirectional\bidirectional-rag-research"

# Activate environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Set environment variables
$env:PYTHONPATH = "D:\bidirectional\bidirectional-rag-research"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# Backup existing results
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (Test-Path "results") {
    Write-Host "Backing up existing results..." -ForegroundColor Yellow
    Copy-Item -Path "results" -Destination "results_backup_$timestamp" -Recurse
    Write-Host "Backup created: results_backup_$timestamp" -ForegroundColor Green
}

# Clear old results (optional - comment out to resume)
# Write-Host "Clearing old results..." -ForegroundColor Yellow
# Remove-Item -Path "results" -Recurse -Force -ErrorAction SilentlyContinue
# New-Item -Path "results" -ItemType Directory -Force | Out-Null

# Run experiments
Write-Host ""
Write-Host "Starting experiments..." -ForegroundColor Green
Write-Host ""

python main.py `
    --systems all `
    --datasets all `
    --seeds 42 43 44 `
    --num_queries 500 `
    --checkpoint_every 125 `
    --offline `
    --max_workers 1 `
    --resume

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "Experiments completed successfully!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ""
    
    # Run analyses
    Write-Host "Running post-experiment analyses..." -ForegroundColor Green
    
    Write-Host "1. Computing hallucination rates..."
    python -c "from src.evaluation.hallucination_analyzer import HallucinationAnalyzer; HallucinationAnalyzer().analyze_all_experiments('results')"
    
    Write-Host "2. Generating figures..."
    python experiments/generate_publication_figures.py
    
    Write-Host "3. Generating LaTeX tables..."
    python experiments/generate_latex_tables.py
    
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "All analyses complete!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "Results location: results/"
    Write-Host ""
    
} else {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "Experiments failed! Check logs for errors." -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host ""
    exit 1
}

