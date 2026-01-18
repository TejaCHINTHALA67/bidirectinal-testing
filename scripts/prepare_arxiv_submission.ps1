# Prepare arXiv Submission Package
# Creates a .tar.gz ready for upload to arxiv.org

Write-Host "============================================================" -ForegroundColor Green
Write-Host "Preparing arXiv Submission Package" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

# Set working directory
Set-Location "D:\bidirectional\bidirectional-rag-research"

# Create submission directory
$submissionDir = "arxiv_submission"
if (Test-Path $submissionDir) {
    Remove-Item -Path $submissionDir -Recurse -Force
}
New-Item -Path $submissionDir -ItemType Directory -Force | Out-Null

Write-Host "Creating submission directory..." -ForegroundColor Yellow

# Copy main paper file
Copy-Item "paper\bidirectional_rag_ieee.tex" "$submissionDir\main.tex"

# Update figure paths in main.tex (relative to submission folder)
$content = Get-Content "$submissionDir\main.tex" -Raw
$content = $content -replace '\.\./results/figures/', 'figures/'
$content = $content -replace '\.\./results/latex_tables/', ''
Set-Content "$submissionDir\main.tex" $content

# Copy bibliography
Copy-Item "paper\references.bib" "$submissionDir\"

# Copy figures
New-Item -Path "$submissionDir\figures" -ItemType Directory -Force | Out-Null
Copy-Item "results\figures\*.pdf" "$submissionDir\figures\"

# Create README
$readme = @"
Bidirectional RAG: Safe Self-Improving Retrieval-Augmented Generation

Main file: main.tex
Compile: pdflatex + bibtex + pdflatex + pdflatex

Files:
- main.tex (paper)
- references.bib (bibliography)
- figures/ (all figures in PDF)

Author: [Your Name]
Contact: [your-email]@example.com
"@
$readme | Out-File -FilePath "$submissionDir\00README.txt" -Encoding UTF8

Write-Host "Submission files prepared:" -ForegroundColor Green
Get-ChildItem -Path $submissionDir -Recurse | ForEach-Object {
    Write-Host "  $($_.FullName.Replace((Get-Location).Path + '\', ''))" -ForegroundColor Cyan
}

# Create tarball (using tar if available, otherwise create zip)
Write-Host ""
Write-Host "Creating archive..." -ForegroundColor Yellow

$tarFile = "bidirectional_rag_arxiv.tar.gz"
$zipFile = "bidirectional_rag_arxiv.zip"

try {
    # Try using tar (available on Windows 10+)
    tar -czvf $tarFile -C $submissionDir .
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "arXiv package created: $tarFile" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
} catch {
    # Fallback to zip
    Compress-Archive -Path "$submissionDir\*" -DestinationPath $zipFile -Force
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "arXiv package created: $zipFile" -ForegroundColor Green
    Write-Host "(Note: arXiv also accepts .zip files)" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Green
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Go to https://arxiv.org/submit"
Write-Host "2. Upload $tarFile (or $zipFile)"
Write-Host "3. Fill in metadata:"
Write-Host "   - Title: Bidirectional RAG: Safe Self-Improving Retrieval-Augmented Generation Through Multi-Stage Validation"
Write-Host "   - Subject: cs.CL (Computation and Language)"
Write-Host "   - Categories: cs.AI, cs.LG"
Write-Host "4. Submit!"
Write-Host ""

