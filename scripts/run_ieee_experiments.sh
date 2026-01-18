#!/bin/bash
# ==============================================================================
# Bidirectional RAG: Run Full IEEE Access Experiments
# ==============================================================================

set -e

# Activate virtual environment (Not needed on Vast.ai if using global env)
# source venv/bin/activate

echo "=============================================="
echo "IEEE Access Publication Experiments"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  - Corpus: REALISTIC (Stack Overflow + Wikipedia)"
echo "  - Systems: ALL (Standard RAG, Self-RAG, FLARE, CRAG, Naive Writeback, Bidirectional RAG)"
echo "  - Seeds: 20 (42-61)"
echo "  - Queries: 1000 per dataset"
echo "  - Datasets: 4 (NQ, TriviaQA, HotpotQA, Stack Overflow)"
echo "  - Total Experiments: 480"
echo ""
echo "Estimated Runtime: 24-48 hours"
echo "=============================================="
echo ""

# Start time
START_TIME=$(date +%s)

# Run experiments (Ollama Llama 3 8B)
python main.py \
    --llm_model llama3 \
    --corpus_type realistic \
    --systems all \
    --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
    --num_queries 1000 \
    --max_workers 4 \
    --output_dir results/ieee_access_final

# End time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "=============================================="
echo "Experiments Complete!"
echo "Runtime: ${HOURS}h ${MINUTES}m"
echo "=============================================="
echo ""

# Generate analysis
echo "Generating statistical analysis..."
python -c "
from src.evaluation.statistical_analysis import StatisticalAnalyzer
import json, glob

# Load all results
results_by_system = {}
for f in glob.glob('results/ieee_access_final/**/experiment_results.json', recursive=True):
    with open(f) as rf:
        data = json.load(rf)
        system = data.get('system', 'unknown')
        if system not in results_by_system:
            results_by_system[system] = []
        results_by_system[system].append(data)

# Run analysis
analyzer = StatisticalAnalyzer()
report = analyzer.generate_report(results_by_system, 'results/ieee_access_final/statistical_analysis_report.txt')
print('Statistical analysis saved to: results/ieee_access_final/statistical_analysis_report.txt')
"

echo ""
echo "Generating figures..."
python experiments/generate_figures.py --input results/ieee_access_final --output "ieee publishing/figures"

echo ""
echo "Generating LaTeX tables..."
python experiments/generate_latex_tables.py --input results/ieee_access_final --output "ieee publishing/paper"

echo ""
echo "=============================================="
echo "All Post-Processing Complete!"
echo "=============================================="
echo ""
echo "Next Steps:"
echo "1. Review results/ieee_access_final/statistical_analysis_report.txt"
echo "2. Check figures in: ieee publishing/figures/"
echo "3. Update LaTeX manuscript with new tables"
echo "4. Recompile PDF: cd 'ieee publishing/paper' && pdflatex bidirectional_rag_ieee_v2.tex"
echo "=============================================="
