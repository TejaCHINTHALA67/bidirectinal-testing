"""
Statistical Analysis
====================

Analyzes experimental results and computes statistical significance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import pandas as pd

# Load results directly without importing BidirectionalRAG (avoids ChromaDB/onnxruntime dependency)
def load_results(results_path: str) -> Dict:
    """Load experiment results from JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class StatisticalAnalyzer:
    """Analyzes experimental results."""
    
    def __init__(self, results_dir: str = 'experiments/results/sparse_corpus'):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        results_dir : str
            Directory containing experiment results
        """
        self.results_dir = results_dir
        
        print("="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
    
    def load_experiment_results(self) -> Dict[str, Dict]:
        """Load all experiment results."""
        experiments = {}
        
        experiment_files = {
            'static_rag': 'static_rag_latest.json',
            'naive_writeback': 'naive_writeback_latest.json',
            'bidirectional_rag': 'bidirectional_rag_latest.json',
            'ablation_study': 'ablation_study_latest.json'
        }
        
        for exp_name, filename in experiment_files.items():
            filepath = os.path.join(self.results_dir, filename)
            try:
                experiments[exp_name] = load_results(filepath)
                print(f"[OK] Loaded: {exp_name}")
            except FileNotFoundError:
                print(f"[WARN] Not found: {exp_name}")
        
        return experiments
    
    def compare_coverage(self, experiments: Dict[str, Dict]) -> Dict:
        """
        Compare coverage across experiments.
        
        Coverage = % of queries that retrieved relevant documents
        """
        print("\n" + "-"*60)
        print("COVERAGE COMPARISON")
        print("-"*60)
        
        coverage_results = {}
        
        for exp_name, exp_data in experiments.items():
            # Prefer direct initial_coverage/final_coverage fields if available
            if 'initial_coverage' in exp_data and 'final_coverage' in exp_data:
                initial = exp_data['initial_coverage']
                final = exp_data['final_coverage']
                improvement = exp_data.get('coverage_improvement', final - initial)
                # Get evolution from checkpoints if available
                coverages = [cp['coverage'] for cp in exp_data.get('checkpoints', [])]
            elif 'checkpoints' in exp_data:
                # Fall back to checkpoints if direct fields not available
                coverages = [cp['coverage'] for cp in exp_data['checkpoints']]
                initial = coverages[0] if coverages else 0.0
                final = coverages[-1] if coverages else 0.0
                improvement = (final - initial) if coverages else 0.0
            else:
                continue
            
            coverage_results[exp_name] = {
                'initial': initial,
                'final': final,
                'improvement': improvement,
                'evolution': coverages if 'coverages' in locals() else []
            }
            
            print(f"\n{exp_name}:")
            print(f"  Initial coverage: {coverage_results[exp_name]['initial']:.2%}")
            print(f"  Final coverage: {coverage_results[exp_name]['final']:.2%}")
            print(f"  Improvement: {coverage_results[exp_name]['improvement']:.2%}")
        
        return coverage_results
    
    def compare_corpus_growth(self, experiments: Dict[str, Dict]) -> Dict:
        """Compare corpus growth across experiments."""
        print("\n" + "-"*60)
        print("CORPUS GROWTH COMPARISON")
        print("-"*60)
        
        growth_results = {}
        
        for exp_name, exp_data in experiments.items():
            if 'final_stats' not in exp_data:
                continue
            
            stats = exp_data['final_stats']
            config = exp_data.get('config', {})
            initial = config.get('initial_corpus_size', 0)
            
            growth_results[exp_name] = {
                'initial_size': initial,
                'final_size': stats.get('corpus_size', initial),
                'growth': stats.get('corpus_growth', 0),
                'growth_rate': stats.get('growth_rate', 0.0)
            }
            
            print(f"\n{exp_name}:")
            print(f"  Initial corpus: {growth_results[exp_name]['initial_size']}")
            print(f"  Final corpus: {growth_results[exp_name]['final_size']}")
            print(f"  Growth: +{growth_results[exp_name]['growth']} documents")
            print(f"  Growth rate: {growth_results[exp_name]['growth_rate']:.2%}")
        
        return growth_results
    
    def analyze_acceptance_layer(self, experiments: Dict[str, Dict]) -> Dict:
        """Analyze acceptance layer performance."""
        print("\n" + "-"*60)
        print("ACCEPTANCE LAYER ANALYSIS")
        print("-"*60)
        
        acceptance_results = {}
        
        # Focus on bidirectional RAG
        if 'bidirectional_rag' in experiments:
            exp_data = experiments['bidirectional_rag']
            
            if 'acceptance_stats' in exp_data:
                stats = exp_data['acceptance_stats']
                total = stats['total_accepted'] + stats['total_rejected']
                
                acceptance_results['bidirectional_rag'] = {
                    'total_queries': total,
                    'accepted': stats['total_accepted'],
                    'rejected': stats['total_rejected'],
                    'acceptance_rate': stats['total_accepted'] / total if total > 0 else 0.0,
                    'rejection_breakdown': stats.get('rejection_reasons', {})
                }
                
                print(f"\nBidirectional RAG:")
                print(f"  Total queries: {total}")
                print(f"  Accepted: {stats['total_accepted']} ({stats['total_accepted']/total:.2%})")
                print(f"  Rejected: {stats['total_rejected']} ({stats['total_rejected']/total:.2%})")
                
                print(f"\n  Rejection breakdown:")
                for reason, count in stats.get('rejection_reasons', {}).items():
                    pct = count / stats['total_rejected'] if stats['total_rejected'] > 0 else 0
                    print(f"    {reason}: {count} ({pct:.2%})")
        
        return acceptance_results
    
    def statistical_significance_tests(
        self,
        experiments: Dict[str, Dict]
    ) -> Dict:
        """
        Perform statistical significance tests.
        
        Tests if differences in coverage are statistically significant.
        """
        print("\n" + "-"*60)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("-"*60)
        
        sig_tests = {}
        
        # Get coverage data for each experiment
        coverage_data = {}
        for exp_name, exp_data in experiments.items():
            if 'queries' not in exp_data:
                continue
            
            # Binary: did each query retrieve docs?
            coverage_data[exp_name] = [
                1 if q.get('retrieved_docs', 0) > 0 else 0
                for q in exp_data['queries']
            ]
        
        # Pairwise comparisons
        experiments_list = list(coverage_data.keys())
        
        for i in range(len(experiments_list)):
            for j in range(i + 1, len(experiments_list)):
                exp1 = experiments_list[i]
                exp2 = experiments_list[j]
                
                data1 = coverage_data[exp1]
                data2 = coverage_data[exp2]
                
                # Ensure same length (truncate to shorter)
                min_len = min(len(data1), len(data2))
                data1 = data1[:min_len]
                data2 = data2[:min_len]
                
                # T-test for proportions
                t_stat, p_value = stats.ttest_ind(data1, data2)
                
                # Cohen's d effect size
                mean1, mean2 = np.mean(data1), np.mean(data2)
                std_pooled = np.sqrt((np.var(data1) + np.var(data2)) / 2)
                cohens_d = (mean1 - mean2) / std_pooled if std_pooled > 0 else 0
                
                comparison_key = f"{exp1} vs {exp2}"
                sig_tests[comparison_key] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'cohens_d': float(cohens_d),
                    'coverage_diff': mean1 - mean2
                }
                
                print(f"\n{comparison_key}:")
                print(f"  Coverage diff: {(mean1 - mean2):.4f}")
                print(f"  t-statistic: {t_stat:.4f}")
                print(f"  p-value: {p_value:.4f}")
                print(f"  Significant (p<0.05): {p_value < 0.05}")
                print(f"  Cohen's d: {cohens_d:.4f}")
        
        return sig_tests
    
    def generate_comparison_table(self, experiments: Dict[str, Dict]) -> pd.DataFrame:
        """Generate comparison table (like Table 2 in paper)."""
        print("\n" + "-"*60)
        print("COMPARISON TABLE")
        print("-"*60)
        
        rows = []
        
        for exp_name, exp_data in experiments.items():
            final_stats = exp_data.get('final_stats', {})
            config = exp_data.get('config', {})
            
            # Get final metrics - prefer direct fields, fall back to checkpoints
            if 'initial_coverage' in exp_data and 'final_coverage' in exp_data:
                initial_coverage = exp_data['initial_coverage']
                final_coverage = exp_data['final_coverage']
            elif 'checkpoints' in exp_data:
                checkpoints = exp_data.get('checkpoints', [])
                final_coverage = checkpoints[-1]['coverage'] if checkpoints else 0.0
                initial_coverage = checkpoints[0]['coverage'] if checkpoints else 0.0
            else:
                continue
            
            row = {
                'System': exp_name.replace('_', ' ').title(),
                'Final Coverage': f"{final_coverage:.2%}",
                'Coverage Gain': f"{final_coverage - initial_coverage:.2%}",
                'Corpus Size': final_stats.get('corpus_size', config.get('initial_corpus_size', 0)),
                'Growth': f"+{final_stats.get('corpus_growth', 0)}",
                'Acceptance Rate': f"{final_stats.get('acceptance_rate', 0.0):.2%}" if 'acceptance_rate' in final_stats else 'N/A'
            }
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        print(f"\n{df.to_string(index=False)}")
        
        return df
    
    def analyze_ablation_study(self, experiments: Dict[str, Dict]) -> Dict:
        """Analyze ablation study results."""
        print("\n" + "-"*60)
        print("ABLATION STUDY ANALYSIS")
        print("-"*60)
        
        if 'ablation_study' not in experiments:
            print("[WARN] Ablation study results not found")
            return {}
        
        ablation_data = experiments['ablation_study']
        configurations = ablation_data.get('configurations', {})
        
        ablation_results = {}
        
        for config_name, config_data in configurations.items():
            summary = config_data.get('summary', {})
            ablation_results[config_name] = {
                'description': config_data.get('description', 'N/A'),
                'acceptance_rate': summary.get('acceptance_rate', 0.0),
                'corpus_growth': summary.get('corpus_growth', 0)
            }
            
            print(f"\n{config_name}:")
            print(f"  Description: {config_data.get('description', 'N/A')}")
            print(f"  Acceptance rate: {summary.get('acceptance_rate', 0.0):.2%}")
            print(f"  Corpus growth: +{summary.get('corpus_growth', 0)} docs")
        
        return ablation_results
    
    def save_analysis_results(
        self,
        coverage_results: Dict,
        growth_results: Dict,
        acceptance_results: Dict,
        sig_tests: Dict,
        comparison_table: pd.DataFrame,
        ablation_results: Dict
    ):
        """Save all analysis results."""
        output_dir = self.results_dir
        
        # Save comprehensive results
        all_results = {
            'coverage_comparison': coverage_results,
            'corpus_growth': growth_results,
            'acceptance_analysis': acceptance_results,
            'significance_tests': sig_tests,
            'ablation_study': ablation_results,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj
        
        converted_results = convert_numpy(all_results)
        
        output_path = os.path.join(output_dir, 'statistical_analysis.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"\n[OK] Saved analysis: {output_path}")
        
        # Save comparison table as CSV
        table_path = os.path.join(output_dir, 'comparison_table.csv')
        comparison_table.to_csv(table_path, index=False)
        print(f"[OK] Saved table: {table_path}")


def main():
    """Run statistical analysis."""
    
    analyzer = StatisticalAnalyzer()
    
    # Load results
    print("\n[Step 1] Loading experiment results...")
    experiments = analyzer.load_experiment_results()
    
    if not experiments:
        print("[ERROR] No experiment results found!")
        print("Please run: python experiments/run_experiments.py")
        return
    
    # Run analyses
    print("\n[Step 2] Running analyses...")
    
    coverage_results = analyzer.compare_coverage(experiments)
    growth_results = analyzer.compare_corpus_growth(experiments)
    acceptance_results = analyzer.analyze_acceptance_layer(experiments)
    sig_tests = analyzer.statistical_significance_tests(experiments)
    comparison_table = analyzer.generate_comparison_table(experiments)
    ablation_results = analyzer.analyze_ablation_study(experiments)
    
    # Save results
    print("\n[Step 3] Saving analysis results...")
    analyzer.save_analysis_results(
        coverage_results,
        growth_results,
        acceptance_results,
        sig_tests,
        comparison_table,
        ablation_results
    )
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {analyzer.results_dir}/")
    print(f"\nNext step:")
    print(f"  Generate figures: python experiments/generate_figures.py")


if __name__ == '__main__':
    main()

