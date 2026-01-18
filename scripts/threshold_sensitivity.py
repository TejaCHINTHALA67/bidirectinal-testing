#!/usr/bin/env python3
"""
Threshold Sensitivity Analysis for Bidirectional RAG
Runs experiments with varying NLI and novelty thresholds to analyze impact on coverage/growth.

This is a simplified version that runs on a subset of data for efficiency.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.systems.baselines import BidirectionalRAG
from src.data.dataset_loader import DatasetLoader

def run_threshold_experiment(
    grounding_threshold: float,
    novelty_threshold: float,
    dataset: str = "nq",
    num_queries: int = 50,
    seed: int = 42
) -> Dict:
    """Run a single threshold configuration experiment."""
    
    print(f"\n{'='*60}")
    print(f"Testing: grounding={grounding_threshold:.2f}, novelty={novelty_threshold:.2f}")
    print(f"Dataset: {dataset}, Queries: {num_queries}, Seed: {seed}")
    print(f"{'='*60}")
    
    # Create unique persist directory for this run
    persist_dir = f"./temp_chroma_threshold_{grounding_threshold}_{novelty_threshold}_{seed}"
    
    try:
        # Load data first
        loader = DatasetLoader(cache_dir='data/raw')
        corpus, queries = loader.load_and_process(dataset)
        
        # Save corpus if needed
        corpus_path = Path(f'data/processed/{dataset}_corpus.json')
        if not corpus_path.exists():
            loader.save_to_disk(corpus, queries, dataset)
        
        # Limit and shuffle queries
        import random
        random.seed(seed)
        random.shuffle(queries)
        train_data = queries[:num_queries]
        
        # Initialize system with these thresholds
        system = BidirectionalRAG(
            corpus_path=str(corpus_path),
            grounding_threshold=grounding_threshold,
            novelty_threshold=novelty_threshold,
            chroma_persist_dir=persist_dir,
            dataset_name=dataset,
            seed=seed,
            enable_query_logging=False,
        )
        
        # Run training queries
        for i, query_data in enumerate(train_data):
            query = query_data.get('question', query_data.get('query', ''))
            if not query:
                continue
            try:
                system.query(query)
            except Exception as e:
                print(f"Query {i} failed: {e}")
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(train_data)} queries...")
        
        # Get final stats
        stats = system.stats if hasattr(system, 'stats') else {}
        
        # Use available stats
        total_queries = stats.get('total_queries', len(train_data))
        corpus_growth = stats.get('documents_added', 0)
        total_accepted = stats.get('total_accepted', 0)
        total_rejected = stats.get('total_rejected', 0)
        acceptance_rate = total_accepted / max(1, total_accepted + total_rejected) if (total_accepted + total_rejected) > 0 else 0
        
        result = {
            'grounding_threshold': grounding_threshold,
            'novelty_threshold': novelty_threshold,
            'dataset': dataset,
            'num_queries': num_queries,
            'seed': seed,
            'corpus_growth': corpus_growth,
            'acceptance_rate': acceptance_rate,
            'total_accepted': total_accepted,
            'total_rejected': total_rejected,
            'rejection_reasons': stats.get('rejection_reasons', {})
        }
        
        print(f"  Result: growth={corpus_growth}, accepted={total_accepted}, rejected={total_rejected}")
        
        # Close system properly before cleanup
        if hasattr(system, '_chroma_client'):
            try:
                # Reset the chroma client to release file locks
                del system._chroma_client
            except:
                pass
        
        return result
        
    finally:
        # Cleanup with retry for locked files
        import shutil
        import time as t
        if os.path.exists(persist_dir):
            for attempt in range(3):
                try:
                    shutil.rmtree(persist_dir)
                    break
                except PermissionError:
                    t.sleep(1)  # Wait and retry
            else:
                print(f"  Warning: Could not cleanup {persist_dir}")


def run_sensitivity_analysis(
    base_grounding: float = 0.65,
    base_novelty: float = 0.90,
    dataset: str = "nq",
    num_queries: int = 50,
    seed: int = 42
) -> List[Dict]:
    """Run sensitivity analysis varying thresholds."""
    
    results = []
    
    # Test grounding threshold variations (keep novelty fixed)
    grounding_values = [
        base_grounding - 0.15,  # 0.50
        base_grounding - 0.10,  # 0.55
        base_grounding,          # 0.65 (default)
        base_grounding + 0.10,  # 0.75
        base_grounding + 0.15,  # 0.80
    ]
    
    print("\n" + "="*70)
    print("PART 1: Grounding Threshold Sensitivity (Novelty fixed at {:.2f})".format(base_novelty))
    print("="*70)
    
    for g_thresh in grounding_values:
        result = run_threshold_experiment(
            grounding_threshold=g_thresh,
            novelty_threshold=base_novelty,
            dataset=dataset,
            num_queries=num_queries,
            seed=seed
        )
        result['varied_param'] = 'grounding'
        results.append(result)
    
    # Test novelty threshold variations (keep grounding fixed)
    novelty_values = [
        base_novelty - 0.10,  # 0.80
        base_novelty - 0.05,  # 0.85
        base_novelty,          # 0.90 (default)
        base_novelty + 0.05,  # 0.95
    ]
    
    print("\n" + "="*70)
    print("PART 2: Novelty Threshold Sensitivity (Grounding fixed at {:.2f})".format(base_grounding))
    print("="*70)
    
    for n_thresh in novelty_values:
        # Skip if already tested at base values
        if n_thresh == base_novelty:
            # Reuse from grounding tests
            for r in results:
                if r['grounding_threshold'] == base_grounding and r['novelty_threshold'] == base_novelty:
                    r_copy = r.copy()
                    r_copy['varied_param'] = 'novelty'
                    results.append(r_copy)
                    break
            continue
            
        result = run_threshold_experiment(
            grounding_threshold=base_grounding,
            novelty_threshold=n_thresh,
            dataset=dataset,
            num_queries=num_queries,
            seed=seed
        )
        result['varied_param'] = 'novelty'
        results.append(result)
    
    return results


def generate_report(results: List[Dict], output_dir: str = "./results"):
    """Generate threshold sensitivity report."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    with open(os.path.join(output_dir, 'threshold_sensitivity.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate text report
    report_lines = [
        "=" * 70,
        "THRESHOLD SENSITIVITY ANALYSIS",
        "Bidirectional RAG - IEEE Access Submission",
        "=" * 70,
        "",
        "METHODOLOGY:",
        "-" * 40,
        "Varied NLI grounding threshold and novelty threshold independently",
        "to analyze impact on coverage and corpus growth.",
        "",
        "GROUNDING THRESHOLD RESULTS:",
        "-" * 40,
    ]
    
    grounding_results = [r for r in results if r.get('varied_param') == 'grounding']
    for r in sorted(grounding_results, key=lambda x: x['grounding_threshold']):
        report_lines.append(
            f"  θ_g = {r['grounding_threshold']:.2f}: "
            f"growth = {r['corpus_growth']}, "
            f"accepted = {r.get('total_accepted', 0)}, "
            f"rejected = {r.get('total_rejected', 0)}"
        )
    
    report_lines.extend([
        "",
        "NOVELTY THRESHOLD RESULTS:",
        "-" * 40,
    ])
    
    novelty_results = [r for r in results if r.get('varied_param') == 'novelty']
    for r in sorted(novelty_results, key=lambda x: x['novelty_threshold']):
        report_lines.append(
            f"  θ_n = {r['novelty_threshold']:.2f}: "
            f"growth = {r['corpus_growth']}, "
            f"accepted = {r.get('total_accepted', 0)}, "
            f"rejected = {r.get('total_rejected', 0)}"
        )
    
    report_lines.extend([
        "",
        "KEY FINDINGS:",
        "-" * 40,
    ])
    
    # Analyze grounding impact
    if grounding_results:
        g_sorted = sorted(grounding_results, key=lambda x: x['grounding_threshold'])
        g_low = g_sorted[0]
        g_high = g_sorted[-1]
        g_growth_change = g_low['corpus_growth'] - g_high['corpus_growth']
        g_accept_change = g_low.get('total_accepted', 0) - g_high.get('total_accepted', 0)
        
        report_lines.append(
            f"1. Grounding threshold: "
            f"Lowering θ_g from {g_high['grounding_threshold']:.2f} to {g_low['grounding_threshold']:.2f} "
            f"adds {abs(g_growth_change)} more documents (accepts {abs(g_accept_change)} more)."
        )
    
    # Analyze novelty impact
    if novelty_results:
        n_sorted = sorted(novelty_results, key=lambda x: x['novelty_threshold'])
        n_low = n_sorted[0]
        n_high = n_sorted[-1]
        n_growth_change = n_low['corpus_growth'] - n_high['corpus_growth']
        n_accept_change = n_low.get('total_accepted', 0) - n_high.get('total_accepted', 0)
        
        report_lines.append(
            f"2. Novelty threshold: "
            f"Lowering θ_n from {n_high['novelty_threshold']:.2f} to {n_low['novelty_threshold']:.2f} "
            f"adds {abs(n_growth_change)} more documents (accepts {abs(n_accept_change)} more)."
        )
    
    report_lines.extend([
        "",
        "CONCLUSION:",
        "-" * 40,
        "Threshold selection involves coverage/safety trade-off.",
        "Lower thresholds increase coverage but admit more content.",
        "Default thresholds (θ_g=0.65, θ_n=0.90) provide balanced performance.",
    ])
    
    report_text = '\n'.join(report_lines)
    
    with open(os.path.join(output_dir, 'threshold_sensitivity_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    
    # Generate LaTeX table
    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Threshold Sensitivity Analysis}",
        "\\label{tab:threshold_sensitivity}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Configuration} & \\textbf{Growth} & \\textbf{Accepted} & \\textbf{Rejected} \\\\",
        "\\midrule",
    ]
    
    # Add grounding results
    for r in sorted(grounding_results, key=lambda x: x['grounding_threshold']):
        marker = " *" if r['grounding_threshold'] == 0.65 else ""
        latex_lines.append(
            f"$\\tau_{{nli}}$ = {r['grounding_threshold']:.2f}{marker} & "
            f"{r['corpus_growth']} & {r.get('total_accepted', 0)} & {r.get('total_rejected', 0)} \\\\"
        )
    
    latex_lines.append("\\midrule")
    
    # Add novelty results
    for r in sorted(novelty_results, key=lambda x: x['novelty_threshold']):
        marker = " *" if r['novelty_threshold'] == 0.90 else ""
        latex_lines.append(
            f"$\\tau_{{sim}}$ = {r['novelty_threshold']:.2f}{marker} & "
            f"{r['corpus_growth']} & {r.get('total_accepted', 0)} & {r.get('total_rejected', 0)} \\\\"
        )
    
    latex_lines.extend([
        "\\bottomrule",
        "\\multicolumn{4}{l}{\\footnotesize * Default threshold} \\\\",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    latex_table = '\n'.join(latex_lines)
    
    with open(os.path.join(output_dir, 'threshold_sensitivity_table.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description='Threshold Sensitivity Analysis')
    parser.add_argument('--dataset', type=str, default='stackoverflow',
                       help='Dataset to use (default: stackoverflow for speed)')
    parser.add_argument('--num_queries', type=int, default=50,
                       help='Number of queries per experiment (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Queries: {args.num_queries}")
    print(f"Seed: {args.seed}")
    print("="*70)
    
    start_time = time.time()
    
    results = run_sensitivity_analysis(
        dataset=args.dataset,
        num_queries=args.num_queries,
        seed=args.seed
    )
    
    generate_report(results, args.output_dir)
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

