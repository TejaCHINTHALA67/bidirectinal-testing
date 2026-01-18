"""
LaTeX Table Generation (Fixed Version)
=======================================

Generates LaTeX tables for paper from experimental results.
Now includes proper grounding representation and accurate metrics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from pathlib import Path


def load_results(results_path: str):
    """Load JSON results file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class LaTeXTableGenerator:
    """Generates LaTeX tables for publication."""
    
    def __init__(self, results_dir: str = 'results'):
        """Initialize generator."""
        self.results_dir = Path(results_dir)
        
        print("="*60)
        print("LATEX TABLE GENERATION (FIXED VERSION)")
        print("="*60)
    
    def aggregate_all_metrics(self):
        """Aggregate metrics across all datasets, systems, and seeds."""
        datasets = ['nq', 'triviaqa', 'hotpotqa', 'stackoverflow']
        systems = ['standard_rag', 'naive_writeback', 'bidirectional_rag']
        seeds = ['42', '43', '44']
        
        system_aggregates = {}
        
        for system in systems:
            metrics_list = []
            
            for dataset in datasets:
                for seed in seeds:
                    metrics_path = self.results_dir / dataset / system / seed / 'metrics.json'
                    if metrics_path.exists():
                        try:
                            m = load_results(str(metrics_path))
                            metrics_list.append(m)
                        except:
                            pass
            
            if metrics_list:
                system_aggregates[system] = {
                    'coverage': np.mean([m.get('coverage', 0) for m in metrics_list]) * 100,
                    'coverage_std': np.std([m.get('coverage', 0) for m in metrics_list]) * 100,
                    'corpus_growth': np.mean([m.get('corpus_growth', 0) for m in metrics_list]),
                    'corpus_growth_std': np.std([m.get('corpus_growth', 0) for m in metrics_list]),
                    'citation_f1': np.mean([m.get('citation_f1', 0) for m in metrics_list]) * 100,
                    'citation_f1_std': np.std([m.get('citation_f1', 0) for m in metrics_list]) * 100,
                    'citation_precision': np.mean([m.get('citation_precision', 0) for m in metrics_list]) * 100,
                    'citation_recall': np.mean([m.get('citation_recall', 0) for m in metrics_list]) * 100,
                    'latency_ms': np.mean([m.get('latency_ms', 0) for m in metrics_list]),
                    'latency_std': np.std([m.get('latency_ms', 0) for m in metrics_list]),
                    'n_experiments': len(metrics_list)
                }
        
        return system_aggregates
    
    def table_main_results(self):
        """
        Main Results Table with accurate metrics.
        Shows: Coverage, Corpus Growth, Grounding Check (Yes/No), Citation F1, Latency
        """
        print("\n[Table] Main Results (Corrected)...")
        
        aggregates = self.aggregate_all_metrics()
        
        # System display names
        system_names = {
            'standard_rag': 'Standard RAG',
            'naive_writeback': 'Naive Write-back',
            'bidirectional_rag': r'\textbf{Bidirectional RAG (Ours)}'
        }
        
        # Grounding check is a feature, not a metric (only BidirectionalRAG has it)
        grounding_check = {
            'standard_rag': 'No',
            'naive_writeback': 'No',
            'bidirectional_rag': r'\textbf{Yes}'
        }
        
        latex = r"""\begin{table*}[t]
\centering
\caption{Main Results: System Comparison Across All Datasets (Mean $\pm$ Std, 12 experiments per system)}
\label{tab:main_results}
\begin{tabular}{lccccc}
\toprule
\textbf{System} & \textbf{Coverage (\%)} & \textbf{Growth (docs)} & \textbf{Grounding} & \textbf{Citation F1 (\%)} & \textbf{Latency (s)} \\
\midrule
"""
        
        for system in ['standard_rag', 'naive_writeback', 'bidirectional_rag']:
            if system not in aggregates:
                continue
            
            m = aggregates[system]
            name = system_names[system]
            grounding = grounding_check[system]
            
            latex += f"{name} & "
            latex += f"{m['coverage']:.2f} $\\pm$ {m['coverage_std']:.2f} & "
            latex += f"{m['corpus_growth']:.0f} & "
            latex += f"{grounding} & "
            latex += f"{m['citation_f1']:.2f} $\\pm$ {m['citation_f1_std']:.2f} & "
            latex += f"{m['latency_ms']/1000:.1f} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
        
        print(latex)
        return latex
    
    def table_coverage_by_dataset(self):
        """Coverage breakdown by dataset."""
        print("\n[Table] Coverage by Dataset...")
        
        datasets = ['nq', 'triviaqa', 'hotpotqa', 'stackoverflow']
        systems = ['standard_rag', 'naive_writeback', 'bidirectional_rag']
        seeds = ['42', '43', '44']
        
        dataset_names = {
            'nq': 'NQ',
            'triviaqa': 'TriviaQA', 
            'hotpotqa': 'HotpotQA',
            'stackoverflow': 'StackOF'
        }
        
        system_names = {
            'standard_rag': 'Standard RAG',
            'naive_writeback': 'Naive Write-back',
            'bidirectional_rag': r'\textbf{Bidirectional RAG}'
        }
        
        latex = r"""\begin{table}[t]
\centering
\caption{Coverage (\%) by Dataset and System}
\label{tab:coverage_dataset}
\begin{tabular}{lcccc}
\toprule
\textbf{System} & \textbf{NQ} & \textbf{TriviaQA} & \textbf{HotpotQA} & \textbf{StackOF} \\
\midrule
"""
        
        for system in systems:
            name = system_names[system]
            latex += f"{name} & "
            
            for i, dataset in enumerate(datasets):
                coverages = []
                for seed in seeds:
                    metrics_path = self.results_dir / dataset / system / seed / 'metrics.json'
                    if metrics_path.exists():
                        m = load_results(str(metrics_path))
                        coverages.append(m.get('coverage', 0) * 100)
                
                if coverages:
                    mean_cov = np.mean(coverages)
                    latex += f"{mean_cov:.1f}"
                else:
                    latex += "--"
                
                if i < len(datasets) - 1:
                    latex += " & "
            
            latex += " \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        print(latex)
        return latex
    
    def generate_all_tables(self):
        """Generate all tables and save to file."""
        
        tables = {
            'table_main_results': self.table_main_results(),
            'table_coverage_dataset': self.table_coverage_by_dataset()
        }
        
        # Save individual tables
        output_dir = self.results_dir / 'latex_tables'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for table_name, latex_content in tables.items():
            if latex_content:
                output_path = output_dir / f'{table_name}.tex'
                with open(output_path, 'w') as f:
                    f.write(latex_content)
                print(f"    [OK] Saved: {output_path}")
        
        # Save all tables in one file
        all_tables_content = "\n\n".join([latex for latex in tables.values() if latex])
        all_tables_path = output_dir / 'all_tables.tex'
        with open(all_tables_path, 'w') as f:
            f.write(all_tables_content)
        print(f"    [OK] Saved: {all_tables_path}")
        
        return tables


def main():
    """Generate LaTeX tables."""
    
    generator = LaTeXTableGenerator()
    
    print("\n[Step 1] Aggregating metrics from all experiments...")
    aggregates = generator.aggregate_all_metrics()
    
    print("\nAggregated Results:")
    print("-" * 60)
    for system, metrics in aggregates.items():
        print(f"{system}:")
        print(f"  Coverage: {metrics['coverage']:.2f}% ± {metrics['coverage_std']:.2f}")
        print(f"  Growth: {metrics['corpus_growth']:.0f} docs")
        print(f"  Citation F1: {metrics['citation_f1']:.2f}%")
        print(f"  Latency: {metrics['latency_ms']/1000:.1f}s")
        print(f"  N experiments: {metrics['n_experiments']}")
        print()
    
    print("\n[Step 2] Generating LaTeX tables...")
    tables = generator.generate_all_tables()
    
    print("\n" + "="*60)
    print("LATEX TABLE GENERATION COMPLETE")
    print("="*60)
    print(f"\nTables saved to: results/latex_tables/")


if __name__ == '__main__':
    main()
