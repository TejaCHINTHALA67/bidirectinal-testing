"""
Generate Publication Figures
=============================

Creates publication-quality figures for the paper from experimental results.
Updated to work with results/dataset/system/seed structure.
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("[WARNING] matplotlib not available. Install with: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False


def load_results(results_path: str):
    """Load JSON results file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def aggregate_metrics(results_dir='results', datasets=None, systems=None, seeds=None):
    """Aggregate metrics across datasets, systems, and seeds."""
    if datasets is None:
        datasets = ['nq', 'triviaqa', 'hotpotqa', 'stackoverflow']
    if systems is None:
        systems = ['standard_rag', 'naive_writeback', 'bidirectional_rag']
    if seeds is None:
        seeds = ['42', '43', '44']
    
    aggregated = {}
    
    for dataset in datasets:
        for system in systems:
            key = f"{dataset}_{system}"
            metrics_list = []
            
            for seed in seeds:
                metrics_path = os.path.join(results_dir, dataset, system, seed, 'metrics.json')
                if os.path.exists(metrics_path):
                    try:
                        metrics = load_results(metrics_path)
                        metrics_list.append(metrics)
                    except:
                        pass
            
            if metrics_list:
                aggregated[key] = {}
                for metric in ['coverage', 'f1_score', 'grounding_score', 
                             'citation_precision', 'citation_recall', 'citation_f1',
                             'latency_ms', 'corpus_growth', 'acceptance_rate']:
                    values = [m.get(metric, 0) for m in metrics_list if metric in m]
                    if values:
                        aggregated[key][metric] = sum(values) / len(values)
    
    return aggregated


def figure_1_system_comparison(
    results_dir: str = 'results',
    output_dir: str = 'results/figures'
) -> str:
    """
    Figure 1: System Comparison Bar Chart
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[ERROR] matplotlib required for figure generation")
        return ""
    
    aggregated = aggregate_metrics(results_dir)
    
    # Aggregate across datasets for each system
    system_names = {
        'standard_rag': 'Standard RAG',
        'naive_writeback': 'Naive Write-back',
        'bidirectional_rag': 'Bidirectional RAG'
    }
    
    system_metrics = {}
    for system in system_names.keys():
        coverage_values = []
        corpus_growth_values = []
        
        for dataset in ['nq', 'triviaqa', 'hotpotqa', 'stackoverflow']:
            key = f"{dataset}_{system}"
            if key in aggregated:
                if 'coverage' in aggregated[key]:
                    coverage_values.append(aggregated[key]['coverage'] * 100)
                if 'corpus_growth' in aggregated[key]:
                    corpus_growth_values.append(aggregated[key]['corpus_growth'])
        
        if coverage_values:
            system_metrics[system] = {
                'coverage': sum(coverage_values) / len(coverage_values),
                'corpus_growth': sum(corpus_growth_values) / len(corpus_growth_values) if corpus_growth_values else 0
            }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Coverage
    systems = list(system_names.keys())
    coverage_values = [system_metrics[s]['coverage'] for s in systems]
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    bars1 = ax1.bar(range(len(systems)), coverage_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(systems)))
    ax1.set_xticklabels([system_names[s] for s in systems], rotation=15, ha='right')
    ax1.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Average Coverage Across Datasets', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim(0, max(coverage_values) * 1.2)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, coverage_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(coverage_values)*0.02,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Subplot 2: Corpus Growth
    growth_values = [system_metrics[s]['corpus_growth'] for s in systems]
    
    bars2 = ax2.bar(range(len(systems)), growth_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(systems)))
    ax2.set_xticklabels([system_names[s] for s in systems], rotation=15, ha='right')
    ax2.set_ylabel('Corpus Growth (Documents)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Corpus Growth Across Datasets', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylim(0, max(growth_values) * 1.2 if growth_values else 100)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, growth_values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(growth_values)*0.02 if growth_values else 5,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    png_path = os.path.join(output_dir, 'system_comparison.png')
    pdf_path = os.path.join(output_dir, 'system_comparison.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved system_comparison.png and .pdf")
    return png_path


def figure_2_coverage_by_dataset(
    results_dir: str = 'results',
    output_dir: str = 'results/figures'
) -> str:
    """
    Figure 2: Coverage by Dataset
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[ERROR] matplotlib required for figure generation")
        return ""
    
    aggregated = aggregate_metrics(results_dir)
    
    datasets = ['nq', 'triviaqa', 'hotpotqa', 'stackoverflow']
    dataset_labels = ['Natural Questions', 'TriviaQA', 'HotpotQA', 'Stack Overflow']
    systems = ['standard_rag', 'naive_writeback', 'bidirectional_rag']
    system_labels = ['Standard RAG', 'Naive Write-back', 'Bidirectional RAG']
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    x = np.arange(len(datasets))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (system, label) in enumerate(zip(systems, system_labels)):
        coverage_values = []
        for dataset in datasets:
            key = f"{dataset}_{system}"
            if key in aggregated and 'coverage' in aggregated[key]:
                coverage_values.append(aggregated[key]['coverage'] * 100)
            else:
                coverage_values.append(0)
        
        offset = (i - 1) * width
        bars = ax.bar(x + offset, coverage_values, width, label=label, 
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars, coverage_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Coverage by Dataset and System', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    png_path = os.path.join(output_dir, 'coverage_by_dataset.png')
    pdf_path = os.path.join(output_dir, 'coverage_by_dataset.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved coverage_by_dataset.png and .pdf")
    return png_path


def generate_all_figures(results_dir: str = 'results', output_dir: str = 'results/figures'):
    """Generate all publication figures."""
    
    print("="*60)
    print("PUBLICATION FIGURE GENERATION")
    print("="*60)
    
    if not MATPLOTLIB_AVAILABLE:
        print("[ERROR] matplotlib is required. Install with: pip install matplotlib")
        return
    
    print("\n[Figure 1] System Comparison...")
    figure_1_system_comparison(results_dir, output_dir)
    
    print("\n[Figure 2] Coverage by Dataset...")
    figure_2_coverage_by_dataset(results_dir, output_dir)
    
    print("\n" + "="*60)
    print("FIGURE GENERATION COMPLETE")
    print("="*60)
    print(f"\nFigures saved to: {output_dir}/")
    print(f"\nGenerated figures:")
    print(f"  - system_comparison.png/.pdf")
    print(f"  - coverage_by_dataset.png/.pdf")


def main():
    """Main entry point."""
    generate_all_figures()


if __name__ == '__main__':
    main()
