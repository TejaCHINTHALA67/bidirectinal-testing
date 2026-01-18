"""
Statistical Significance Analysis for IEEE-Grade RAG Experiments

Implements rigorous statistical tests for comparing systems:
- Paired t-tests (for same test set)
- Wilcoxon signed-rank test (non-parametric alternative)
- Cohen's d effect size
- Bootstrap confidence intervals
- Bonferroni correction for multiple comparisons
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARNING] scipy not available. Install with: pip install scipy")

try:
    from statsmodels.stats.power import TTestIndPower
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("[WARNING] statsmodels not available. Power analysis disabled.")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Statistical analyzer for comparing RAG systems.
    
    Performs rigorous statistical tests to determine if differences
    between systems are statistically significant.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analyzer.
        
        Parameters
        ----------
        alpha : float
            Significance level (default: 0.05)
        """
        self.alpha = alpha
        if not SCIPY_AVAILABLE:
            logger.warning("[StatisticalAnalysis] scipy not available. Limited functionality.")
    
    def compare_systems(
        self,
        system1_results: List[Dict],
        system2_results: List[Dict],
        metric_name: str = 'coverage',
        system1_name: str = 'System 1',
        system2_name: str = 'System 2'
    ) -> Dict:
        """
        Compare two systems on a specific metric.
        
        Parameters
        ----------
        system1_results : List[Dict]
            Results from first system (one dict per query)
        system2_results : List[Dict]
            Results from second system (one dict per query)
        metric_name : str
            Metric to compare (e.g., 'coverage', 'f1_score', 'latency_ms')
        system1_name : str
            Name of first system
        system2_name : str
            Name of second system
        
        Returns
        -------
        Dict
            Statistical comparison results:
            {
                'metric': str,
                'system1_mean': float,
                'system2_mean': float,
                'difference': float,
                't_statistic': float,
                'p_value': float,
                'significant': bool,
                'cohens_d': float,
                'effect_size': str,
                'bootstrap_ci_lower': float,
                'bootstrap_ci_upper': float,
                'wilcoxon_statistic': float,
                'wilcoxon_p_value': float
            }
        """
        # Extract metric values
        values1 = self._extract_metric_values(system1_results, metric_name)
        values2 = self._extract_metric_values(system2_results, metric_name)
        
        if len(values1) != len(values2):
            logger.warning(f"[StatisticalAnalysis] Mismatched sample sizes: {len(values1)} vs {len(values2)}")
            # Use minimum length
            min_len = min(len(values1), len(values2))
            values1 = values1[:min_len]
            values2 = values2[:min_len]
        
        if len(values1) < 2:
            logger.warning(f"[StatisticalAnalysis] Insufficient samples for statistical test: {len(values1)}")
            return self._empty_comparison(metric_name, system1_name, system2_name)
        
        # Compute means
        mean1 = np.mean(values1)
        mean2 = np.mean(values2)
        difference = mean2 - mean1  # system2 - system1
        
        # Paired t-test (since same test set)
        t_stat, p_value = self._paired_ttest(values1, values2)
        significant = p_value < self.alpha
        
        # Cohen's d (effect size)
        cohens_d = self._cohens_d(values1, values2)
        effect_size = self._interpret_effect_size(cohens_d)
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ci(values1, values2, n_bootstrap=1000)
        
        # Wilcoxon signed-rank test (non-parametric)
        wilcoxon_stat, wilcoxon_p = self._wilcoxon_test(values1, values2)
        
        return {
            'metric': metric_name,
            'system1_name': system1_name,
            'system2_name': system2_name,
            'system1_mean': float(mean1),
            'system2_mean': float(mean2),
            'difference': float(difference),
            'n_samples': len(values1),
            't_statistic': float(t_stat) if t_stat is not None else None,
            'p_value': float(p_value) if p_value is not None else None,
            'significant': significant,
            'cohens_d': float(cohens_d) if cohens_d is not None else None,
            'effect_size': effect_size,
            'bootstrap_ci_lower': float(ci_lower) if ci_lower is not None else None,
            'bootstrap_ci_upper': float(ci_upper) if ci_upper is not None else None,
            'bootstrap_ci_level': 0.95,
            'wilcoxon_statistic': float(wilcoxon_stat) if wilcoxon_stat is not None else None,
            'wilcoxon_statistic': float(wilcoxon_stat) if wilcoxon_stat is not None else None,
            'wilcoxon_p_value': float(wilcoxon_p) if wilcoxon_p is not None else None
        }

    def compare_runs_welch(
        self,
        system1_means: List[float],
        system2_means: List[float],
        metric_name: str,
        system1_name: str,
        system2_name: str
    ) -> Dict:
        """
        Compare systems using run-level averages (Welch's t-test).
        Use this when comparing N independent runs (e.g. 20 seeds).
        """
        if len(system1_means) < 2 or len(system2_means) < 2:
            return self._empty_comparison(metric_name, system1_name, system2_name)
            
        # Welch's t-test (independent samples, unequal variance)
        t_stat, p_value = stats.ttest_ind(system1_means, system2_means, equal_var=False)
        significant = p_value < self.alpha
        
        # Effect size
        cohens_d = self._cohens_d(system1_means, system2_means)
        
        # Power Analysis
        power = self._compute_power(len(system1_means), cohens_d, self.alpha)
        
        mean1 = np.mean(system1_means)
        mean2 = np.mean(system2_means)
        
        return {
            'metric': metric_name,
            'test_type': 'Welch\'s t-test (Run-level)',
            'system1_name': system1_name,
            'system2_name': system2_name,
            'system1_mean': float(mean1),
            'system2_mean': float(mean2),
            'difference': float(mean2 - mean1),
            'n_samples': len(system1_means),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': significant,
            'cohens_d': float(cohens_d) if cohens_d else 0.0,
            'effect_size': self._interpret_effect_size(cohens_d),
            'power': float(power)
        }

    def _compute_power(self, nobs: int, effect_size: Optional[float], alpha: float) -> float:
        """Compute statistical power."""
        if not STATSMODELS_AVAILABLE or not effect_size:
            return 0.0
        try:
            analysis = TTestIndPower()
            return analysis.solve_power(effect_size=effect_size, nobs1=nobs, ratio=1.0, alpha=alpha)
        except:
            return 0.0
    
    def _extract_metric_values(self, results: List[Dict], metric_name: str) -> List[float]:
        """Extract metric values from results."""
        values = []
        
        for result in results:
            # Try direct access
            if metric_name in result:
                value = result[metric_name]
                if isinstance(value, (int, float)):
                    values.append(float(value))
            # Try nested access (e.g., result['metrics']['coverage'])
            elif 'metrics' in result and isinstance(result['metrics'], dict):
                if metric_name in result['metrics']:
                    value = result['metrics'][metric_name]
                    if isinstance(value, (int, float)):
                        values.append(float(value))
        
        return values
    
    def _paired_ttest(self, values1: List[float], values2: List[float]) -> Tuple[Optional[float], Optional[float]]:
        """Perform paired t-test."""
        if not SCIPY_AVAILABLE:
            return None, None
        
        try:
            # Paired t-test: test if mean difference is significantly different from 0
            differences = np.array(values2) - np.array(values1)
            t_stat, p_value = stats.ttest_1samp(differences, 0.0)
            return t_stat, p_value
        except Exception as e:
            logger.warning(f"[StatisticalAnalysis] Paired t-test failed: {e}")
            return None, None
    
    def _cohens_d(self, values1: List[float], values2: List[float]) -> Optional[float]:
        """
        Compute Cohen's d effect size.
        
        Cohen's d = (mean2 - mean1) / pooled_std
        
        Interpretation:
        - |d| < 0.2: Negligible
        - 0.2 <= |d| < 0.5: Small
        - 0.5 <= |d| < 0.8: Medium
        - |d| >= 0.8: Large
        """
        try:
            mean1 = np.mean(values1)
            mean2 = np.mean(values2)
            std1 = np.std(values1, ddof=1)  # Sample std
            std2 = np.std(values2, ddof=1)
            
            n1 = len(values1)
            n2 = len(values2)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            
            if pooled_std == 0:
                return None
            
            cohens_d = (mean2 - mean1) / pooled_std
            return cohens_d
        except Exception as e:
            logger.warning(f"[StatisticalAnalysis] Cohen's d computation failed: {e}")
            return None
    
    def _interpret_effect_size(self, cohens_d: Optional[float]) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d is None:
            return "N/A"
        
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def _bootstrap_ci(
        self,
        values1: List[float],
        values2: List[float],
        n_bootstrap: int = 1000,
        ci_level: float = 0.95
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute bootstrap confidence interval for difference in means.
        
        Parameters
        ----------
        values1 : List[float]
            Values from system 1
        values2 : List[float]
            Values from system 2
        n_bootstrap : int
            Number of bootstrap samples
        ci_level : float
            Confidence level (default: 0.95)
        
        Returns
        -------
        Tuple[float, float]
            (lower_bound, upper_bound)
        """
        try:
            differences = np.array(values2) - np.array(values1)
            n = len(differences)
            
            if n < 2:
                return None, None
            
            # Bootstrap sampling
            bootstrap_diffs = []
            for _ in range(n_bootstrap):
                # Resample with replacement
                sample_indices = np.random.choice(n, size=n, replace=True)
                sample_diffs = differences[sample_indices]
                bootstrap_diffs.append(np.mean(sample_diffs))
            
            # Compute confidence interval
            alpha = 1 - ci_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_diffs, lower_percentile)
            ci_upper = np.percentile(bootstrap_diffs, upper_percentile)
            
            return ci_lower, ci_upper
        except Exception as e:
            logger.warning(f"[StatisticalAnalysis] Bootstrap CI computation failed: {e}")
            return None, None
    
    def _wilcoxon_test(self, values1: List[float], values2: List[float]) -> Tuple[Optional[float], Optional[float]]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to t-test).
        
        Useful when data is not normally distributed.
        """
        if not SCIPY_AVAILABLE:
            return None, None
        
        try:
            differences = np.array(values2) - np.array(values1)
            # Remove zeros (Wilcoxon requires non-zero differences)
            differences = differences[differences != 0]
            
            if len(differences) < 2:
                return None, None
            
            stat, p_value = stats.wilcoxon(differences)
            return stat, p_value
        except Exception as e:
            logger.warning(f"[StatisticalAnalysis] Wilcoxon test failed: {e}")
            return None, None
    
    def _empty_comparison(self, metric_name: str, system1_name: str, system2_name: str) -> Dict:
        """Return empty comparison result."""
        return {
            'metric': metric_name,
            'system1_name': system1_name,
            'system2_name': system2_name,
            'system1_mean': 0.0,
            'system2_mean': 0.0,
            'difference': 0.0,
            'n_samples': 0,
            't_statistic': None,
            'p_value': None,
            'significant': False,
            'cohens_d': None,
            'effect_size': 'N/A',
            'bootstrap_ci_lower': None,
            'bootstrap_ci_upper': None,
            'bootstrap_ci_level': 0.95,
            'wilcoxon_statistic': None,
            'wilcoxon_p_value': None
        }
    
    def compare_multiple_systems(
        self,
        all_system_results: Dict[str, List[Dict]],
        metrics: List[str] = None
    ) -> Dict:
        """
        Compare multiple systems on multiple metrics.
        
        Parameters
        ----------
        all_system_results : Dict[str, List[Dict]]
            Dictionary mapping system names to their results
        metrics : List[str]
            List of metrics to compare (default: all common metrics)
        
        Returns
        -------
        Dict
            Comprehensive comparison results with Bonferroni correction
        """
        if metrics is None:
            # Default metrics
            metrics = [
                'coverage',
                'f1_score',
                'exact_match',
                'grounding_score',
                'latency_ms'
            ]
        
        system_names = list(all_system_results.keys())
        
        if len(system_names) < 2:
            logger.warning("[StatisticalAnalysis] Need at least 2 systems for comparison")
            return {}
        
        # Generate all pairwise comparisons
        comparisons = {}
        
        for i, sys1 in enumerate(system_names):
            for sys2 in system_names[i+1:]:
                for metric in metrics:
                    comparison_key = f"{sys1}_vs_{sys2}_{metric}"
                    
                    comparison = self.compare_systems(
                        all_system_results[sys1],
                        all_system_results[sys2],
                        metric_name=metric,
                        system1_name=sys1,
                        system2_name=sys2
                    )
                    
                    comparisons[comparison_key] = comparison
        
        # Apply Bonferroni correction
        n_comparisons = len(comparisons)
        corrected_alpha = self.alpha / n_comparisons if n_comparisons > 0 else self.alpha
        
        # Update significance flags
        for comp_key, comp in comparisons.items():
            if comp['p_value'] is not None:
                comp['significant_uncorrected'] = comp['significant']
                comp['significant'] = comp['p_value'] < corrected_alpha
                comp['bonferroni_corrected_alpha'] = corrected_alpha
        
        return {
            'comparisons': comparisons,
            'n_comparisons': n_comparisons,
            'bonferroni_corrected_alpha': corrected_alpha,
            'original_alpha': self.alpha
        }
    
    def generate_report(
        self,
        comparison_results: Dict,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate human-readable statistical report.
        
        Parameters
        ----------
        comparison_results : Dict
            Results from compare_systems or compare_multiple_systems
        output_path : Optional[Path]
            Path to save report (optional)
        
        Returns
        -------
        str
            Formatted report text
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("STATISTICAL SIGNIFICANCE ANALYSIS")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Single comparison
        if 'metric' in comparison_results:
            comp = comparison_results
            report_lines.append(f"Comparison: {comp['system1_name']} vs {comp['system2_name']}")
            report_lines.append(f"Metric: {comp['metric']}")
            report_lines.append("")
            report_lines.append(f"System 1 Mean: {comp['system1_mean']:.4f}")
            report_lines.append(f"System 2 Mean: {comp['system2_mean']:.4f}")
            report_lines.append(f"Difference: {comp['difference']:+.4f}")
            report_lines.append(f"Sample Size: {comp['n_samples']}")
            report_lines.append("")
            
            if comp['t_statistic'] is not None:
                report_lines.append("Paired t-test:")
                report_lines.append(f"  t-statistic: {comp['t_statistic']:.4f}")
                report_lines.append(f"  p-value: {comp['p_value']:.6f}")
                report_lines.append(f"  Significant (alpha={self.alpha}): {'Yes' if comp['significant'] else 'No'}")
                report_lines.append("")
            
            if comp['cohens_d'] is not None:
                report_lines.append("Effect Size (Cohen's d):")
                report_lines.append(f"  d = {comp['cohens_d']:.4f}")
                report_lines.append(f"  Interpretation: {comp['effect_size']}")
                report_lines.append("")
            
            if comp['bootstrap_ci_lower'] is not None:
                report_lines.append("Bootstrap 95% Confidence Interval:")
                report_lines.append(f"  [{comp['bootstrap_ci_lower']:.4f}, {comp['bootstrap_ci_upper']:.4f}]")
                report_lines.append("")
            
            if comp['wilcoxon_p_value'] is not None:
                report_lines.append("Wilcoxon Signed-Rank Test (non-parametric):")
                report_lines.append(f"  Statistic: {comp['wilcoxon_statistic']:.4f}")
                report_lines.append(f"  p-value: {comp['wilcoxon_p_value']:.6f}")
                report_lines.append("")
        
        # Multiple comparisons
        elif 'comparisons' in comparison_results:
            report_lines.append(f"Multiple System Comparison")
            report_lines.append(f"Number of comparisons: {comparison_results['n_comparisons']}")
            report_lines.append(f"Bonferroni corrected α: {comparison_results['bonferroni_corrected_alpha']:.6f}")
            report_lines.append("")
            
            # Group by metric
            by_metric = {}
            for comp_key, comp in comparison_results['comparisons'].items():
                metric = comp['metric']
                if metric not in by_metric:
                    by_metric[metric] = []
                by_metric[metric].append(comp)
            
            for metric, comps in by_metric.items():
                report_lines.append(f"\n{'-' * 70}")
                report_lines.append(f"Metric: {metric.upper()}")
                report_lines.append(f"{'-' * 70}")
                
                for comp in comps:
                    report_lines.append(f"\n{comp['system1_name']} vs {comp['system2_name']}:")
                    report_lines.append(f"  Difference: {comp['difference']:+.4f}")
                    if comp['p_value'] is not None:
                        report_lines.append(f"  p-value: {comp['p_value']:.6f}")
                        report_lines.append(f"  Significant: {'Yes' if comp['significant'] else 'No'}")
                    if comp['cohens_d'] is not None:
                        report_lines.append(f"  Effect size: {comp['effect_size']} (d={comp['cohens_d']:.4f})")
        
        report_lines.append("")
        report_lines.append("=" * 70)
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"[StatisticalAnalysis] Report saved to {output_path}")
        
        return report_text


def load_results_from_directory(results_dir: Path) -> Dict[str, List[Dict]]:
    """
    Load experiment results from directory structure.
    
    Expected structure:
    results_dir/
        dataset/
            system1/
                seed_42/
                    metrics.json
                    test_results.json (optional)
                seed_43/
                    ...
            system2/
                ...
    
    Parameters
    ----------
    results_dir : Path
        Root directory containing results
    
    Returns
    -------
    Dict[str, List[Dict]]
        Dictionary mapping system names to their results
    """
    all_results = {}
    
    if not results_dir.exists():
        logger.warning(f"[StatisticalAnalysis] Results directory not found: {results_dir}")
        return all_results
    
    # Find all dataset directories
    for dataset_dir in results_dir.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name == 'checkpoints':
            continue
        
        # Find all system directories
        for system_dir in dataset_dir.iterdir():
            if not system_dir.is_dir():
                continue
            
            system_name = f"{dataset_dir.name}_{system_dir.name}"  # dataset_system
            
            if system_name not in all_results:
                all_results[system_name] = []
            
            # Find all seed directories
            for seed_dir in system_dir.iterdir():
                if not seed_dir.is_dir():
                    continue
                
                # Try to load metrics.json (contains aggregated metrics)
                metrics_file = seed_dir / 'metrics.json'
                test_results_file = seed_dir / 'test_results.json'
                
                if metrics_file.exists():
                    try:
                        with open(metrics_file, 'r', encoding='utf-8') as f:
                            metrics = json.load(f)
                        
                        # Convert metrics to result format
                        result = {
                            'metrics': metrics,
                            'seed': seed_dir.name,
                            'dataset': dataset_dir.name,
                            'system': system_dir.name
                        }
                        all_results[system_name].append(result)
                        
                    except Exception as e:
                        logger.warning(f"[StatisticalAnalysis] Failed to load {metrics_file}: {e}")
                
                # Also try test_results.json if it exists
                elif test_results_file.exists():
                    try:
                        with open(test_results_file, 'r', encoding='utf-8') as f:
                            test_results = json.load(f)
                        
                        if isinstance(test_results, list):
                            all_results[system_name].extend(test_results)
                        elif isinstance(test_results, dict) and 'test_results' in test_results:
                            all_results[system_name].extend(test_results['test_results'])
                            
                    except Exception as e:
                        logger.warning(f"[StatisticalAnalysis] Failed to load {test_results_file}: {e}")
    
    # Filter out empty results
    all_results = {k: v for k, v in all_results.items() if v}
    
    return all_results


if __name__ == '__main__':
    # Test statistical analyzer
    print("=" * 70)
    print("TESTING STATISTICAL ANALYZER")
    print("=" * 70)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    # System 1: mean = 0.5, std = 0.1
    system1_results = [
        {'coverage': float(x)} for x in np.random.normal(0.5, 0.1, n_samples)
    ]
    
    # System 2: mean = 0.6, std = 0.1 (significantly better)
    system2_results = [
        {'coverage': float(x)} for x in np.random.normal(0.6, 0.1, n_samples)
    ]
    
    analyzer = StatisticalAnalyzer(alpha=0.05)
    
    # Single comparison
    comparison = analyzer.compare_systems(
        system1_results,
        system2_results,
        metric_name='coverage',
        system1_name='Baseline',
        system2_name='Proposed'
    )
    
    print("\nSingle Comparison:")
    print(f"  Difference: {comparison['difference']:.4f}")
    print(f"  p-value: {comparison['p_value']:.6f}")
    print(f"  Significant: {comparison['significant']}")
    print(f"  Effect size: {comparison['effect_size']} (d={comparison['cohens_d']:.4f})")
    
    # Generate report
    report = analyzer.generate_report(comparison)
    print("\n" + report)
    
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYZER TEST COMPLETE")
    print("=" * 70)

