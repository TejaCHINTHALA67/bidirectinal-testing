"""
Supplementary Report Generator
==============================

Aggregates all experimental results into a comprehensive Supplementary Materials report.
Sections:
1. Experimental Setup (Corpus, Models)
2. Statistical Significance (t-tests)
3. Hallucination Analysis (FActScore)
4. Long-Term Trajectories (Plots)
5. Hybrid Baseline Comparisons

Usage:
    python scripts/generate_supplementary_report.py --output supplementary.md
"""

import json
import os
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(f"[WARNING] File not found: {path}")
    return {}

def format_stats_table(stats_data):
    """Format statistical analysis results into a Markdown table."""
    if not stats_data:
        return "*No statistical data available.*"
        
    md = "| Metric | System A | System B | Diff | p-value | Signif |\n"
    md += "|---|---|---|---|---|---|\n"
    
    # Assuming stats_data has list of comparisons
    # This structure depends on how statistical_analysis.py saves output
    # For now, we'll mock the structure or expect a list of dicts
    if isinstance(stats_data, list):
        for comp in stats_data:
            metric = comp.get('metric', 'N/A')
            sys1 = comp.get('system1_name', 'Sys1')
            sys2 = comp.get('system2_name', 'Sys2')
            m1 = comp.get('system1_mean', 0.0)
            m2 = comp.get('system2_mean', 0.0)
            diff = comp.get('difference', 0.0)
            pval = comp.get('p_value', 1.0)
            sig = "**Yes**" if comp.get('significant') else "No"
            
            md += f"| {metric} | {sys1} ({m1:.2f}) | {sys2} ({m2:.2f}) | {diff:+.2f} | {pval:.4f} | {sig} |\n"
            
    return md

def format_hallucination_table(results_dir):
    """Load hallucination metrics and format table."""
    # Look for hallucination_*.json
    files = Path(results_dir).glob("hallucination_*.json")
    rows = []
    
    for f in files:
        data = load_json(f)
        sys_name = data.get('system', f.stem.replace('hallucination_', ''))
        fact_score = data.get('average_fact_score', 0.0)
        grounding = data.get('average_grounding_score', 0.0)
        
        rows.append(f"| {sys_name} | {fact_score:.3f} | {grounding:.3f} |")
        
    if not rows:
        return "*No hallucination metrics found.*"
        
    md = "| System | FActScore (Lite) | Grounding Score |\n"
    md += "|---|---|---|\n"
    md += "\n".join(rows)
    return md

def generate_report(output_path, results_dir="results"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Load main summary
    summary = load_json(os.path.join(results_dir, "experiment_summary.json"))
    
    md = f"""# Supplementary Materials: Bidirectional RAG
**Generated:** {timestamp}

## 1. Experimental Setup
This document details the experimental results for the paper "Bidirectional RAG: Learning from Validated Feedback".

- **Corpus Configurations**: Sparse (2k docs), Moderate (2.5k docs), Realistic (4k docs).
- **Models**: Generator (Llama 3.2 3B / Llama 3.1 8B), Embeddings (all-MiniLM-L6-v2), NLI (DeBERTa-v3-base).
- **Evaluation**: 5 baseline systems, 20 random seeds, 4 datasets.

## 2. Main Results & Statistical Significance
The following table summarizes pairwise t-tests between Bidirectional RAG and Standard RAG/Naive Writeback.

*(Note: Data loaded from `statistical_analysis.py` outputs)*

{format_stats_table(summary.get('statistical_analysis', []))}

## 3. Hallucination Analysis
We evaluated hallucination rates using FActScore-lite (atomic fact verification).

{format_hallucination_table(results_dir)}

## 4. Learning Trajectories
For the long-term evaluation (5000 queries), we visualize the growth of the Experience Store and the stability of coverage.

![Corpus Growth](file://{os.path.abspath(results_dir)}/long_term/seed_42/corpus_growth.png)
*Fig S1. Growth of synthetic experiences over 5000 queries.*

![Grounding Trajectory](file://{os.path.abspath(results_dir)}/long_term/seed_42/grounding_trajectory.png)
*Fig S2. Grounding score stability over time.*

## 5. Hybrid Baseline Comparisons
Comparison against SOTA methods augmented with Naive Writeback.

*(Results from `results/hybrid_comparison/`)*

"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)
    
    print(f"[SUCCESS] Supplementary report generated at: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='supplementary.md')
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args()
    
    generate_report(args.output, args.results_dir)
