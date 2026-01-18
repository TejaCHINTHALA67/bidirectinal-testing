"""
Figure Generation
=================

Generates publication-quality figures for the paper.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd

from src.dataset_utils import load_results


# Set publication-quality style
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

sns.set_style("whitegrid")
COLORS = sns.color_palette("Set2", 8)


class FigureGenerator:
    """Generates publication figures."""
    
    def __init__(
        self,
        results_dir: str = 'experiments/results',
        output_dir: str = 'experiments/results/figures'
    ):
        """
        Initialize figure generator.
        
        Parameters
        ----------
        results_dir : str
            Directory containing results
        output_dir : str
            Directory to save figures
        """
        self.results_dir = results_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*60)
        print("PUBLICATION FIGURE GENERATION")
        print("="*60)
        print(f"Output directory: {output_dir}")
    
    def load_experiments(self) -> Dict[str, Dict]:
        """Load experiment results."""
        experiments = {}
        
        experiment_files = {
            'Static RAG': 'static_rag_latest.json',
            'Naive Write-back': 'naive_writeback_latest.json',
            'Bidirectional RAG': 'bidirectional_rag_latest.json'
        }
        
        for exp_name, filename in experiment_files.items():
            filepath = os.path.join(self.results_dir, filename)
            try:
                experiments[exp_name] = load_results(filepath)
                print(f"[OK] Loaded: {exp_name}")
            except FileNotFoundError:
                print(f"[WARN] Not found: {exp_name}")
        
        return experiments
    
    def figure_1_coverage_evolution(self, experiments: Dict[str, Dict]):
        """
        Figure 1: Coverage Evolution Over Time
        
        Shows how coverage changes as queries are processed.
        """
        print("\n[Figure 1] Coverage Evolution...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for idx, (exp_name, exp_data) in enumerate(experiments.items()):
            if 'checkpoints' not in exp_data:
                continue
            
            checkpoints = exp_data['checkpoints']
            query_indices = [cp['query_index'] for cp in checkpoints]
            coverages = [cp['coverage'] * 100 for cp in checkpoints]  # Convert to percentage
            
            ax.plot(
                query_indices,
                coverages,
                marker='o',
                linewidth=2,
                markersize=6,
                label=exp_name,
                color=COLORS[idx]
            )
        
        ax.set_xlabel('Number of Queries Processed', fontweight='bold')
        ax.set_ylabel('Coverage (%)', fontweight='bold')
        ax.set_title('Coverage Evolution Across Query Sequence', fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # Save
        output_path = os.path.join(self.output_dir, 'figure_1_coverage_evolution')
        plt.tight_layout()
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"    [OK] Saved: {output_path}.png/.pdf")
    
    def figure_2_system_comparison(self, experiments: Dict[str, Dict]):
        """
        Figure 2: System Comparison Bar Chart
        
        Compares key metrics across systems.
        """
        print("\n[Figure 2] System Comparison...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        systems = []
        final_coverages = []
        growth_rates = []
        acceptance_rates = []
        
        for exp_name, exp_data in experiments.items():
            systems.append(exp_name)
            
            # Coverage
            checkpoints = exp_data.get('checkpoints', [])
            if checkpoints:
                final_coverages.append(checkpoints[-1]['coverage'] * 100)
            else:
                final_coverages.append(0)
            
            # Growth
            final_stats = exp_data.get('final_stats', {})
            config = exp_data.get('config', {})
            initial = config.get('initial_corpus_size', 1)
            growth = final_stats.get('corpus_growth', 0)
            growth_rates.append(growth / initial * 100 if initial > 0 else 0)
            
            # Acceptance
            acceptance_rate = final_stats.get('acceptance_rate', 0) * 100
            acceptance_rates.append(acceptance_rate)
        
        # Plot 1: Final Coverage
        axes[0].bar(range(len(systems)), final_coverages, color=COLORS[:len(systems)])
        axes[0].set_xticks(range(len(systems)))
        axes[0].set_xticklabels(systems, rotation=15, ha='right')
        axes[0].set_ylabel('Coverage (%)', fontweight='bold')
        axes[0].set_title('Final Coverage', fontweight='bold')
        axes[0].set_ylim([0, max(final_coverages) * 1.2])
        
        # Add value labels
        for i, v in enumerate(final_coverages):
            axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Growth Rate
        axes[1].bar(range(len(systems)), growth_rates, color=COLORS[:len(systems)])
        axes[1].set_xticks(range(len(systems)))
        axes[1].set_xticklabels(systems, rotation=15, ha='right')
        axes[1].set_ylabel('Growth Rate (%)', fontweight='bold')
        axes[1].set_title('Corpus Growth', fontweight='bold')
        
        for i, v in enumerate(growth_rates):
            axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Acceptance Rate
        axes[2].bar(range(len(systems)), acceptance_rates, color=COLORS[:len(systems)])
        axes[2].set_xticks(range(len(systems)))
        axes[2].set_xticklabels(systems, rotation=15, ha='right')
        axes[2].set_ylabel('Acceptance Rate (%)', fontweight='bold')
        axes[2].set_title('Write-back Acceptance', fontweight='bold')
        
        for i, v in enumerate(acceptance_rates):
            if v > 0:
                axes[2].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Overall title
        fig.suptitle('System Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.output_dir, 'figure_2_system_comparison')
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"    [OK] Saved: {output_path}.png/.pdf")
    
    def figure_3_rejection_breakdown(self, experiments: Dict[str, Dict]):
        """
        Figure 3: Rejection Reason Breakdown
        
        Shows why responses were rejected by acceptance layer.
        """
        print("\n[Figure 3] Rejection Breakdown...")
        
        # Get bidirectional RAG data
        if 'Bidirectional RAG' not in experiments:
            print("    [SKIP] Bidirectional RAG data not available")
            return
        
        exp_data = experiments['Bidirectional RAG']
        acceptance_stats = exp_data.get('acceptance_stats', {})
        rejection_reasons = acceptance_stats.get('rejection_reasons', {})
        
        if not rejection_reasons or sum(rejection_reasons.values()) == 0:
            print("    [SKIP] No rejection data available")
            return
        
        # Prepare data
        reasons = list(rejection_reasons.keys())
        counts = list(rejection_reasons.values())
        total = sum(counts)
        percentages = [c / total * 100 for c in counts]
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=reasons,
            autopct='%1.1f%%',
            startangle=90,
            colors=COLORS[:len(reasons)],
            explode=[0.05] * len(reasons),
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
            autotext.set_fontweight('bold')
        
        ax.set_title(
            'Rejection Reason Breakdown\n(Bidirectional RAG Acceptance Layer)',
            fontweight='bold',
            fontsize=14,
            pad=20
        )
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.output_dir, 'figure_3_rejection_breakdown')
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"    [OK] Saved: {output_path}.png/.pdf")
    
    def figure_4_corpus_growth(self, experiments: Dict[str, Dict]):
        """
        Figure 4: Corpus Size Over Time
        
        Shows how corpus grows during query processing.
        """
        print("\n[Figure 4] Corpus Growth...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for idx, (exp_name, exp_data) in enumerate(experiments.items()):
            if 'checkpoints' not in exp_data:
                continue
            
            checkpoints = exp_data['checkpoints']
            query_indices = [cp['query_index'] for cp in checkpoints]
            corpus_sizes = [cp.get('corpus_size', 0) for cp in checkpoints]
            
            # Skip if no growth
            if max(corpus_sizes) == min(corpus_sizes):
                continue
            
            ax.plot(
                query_indices,
                corpus_sizes,
                marker='s',
                linewidth=2,
                markersize=6,
                label=exp_name,
                color=COLORS[idx]
            )
        
        ax.set_xlabel('Number of Queries Processed', fontweight='bold')
        ax.set_ylabel('Corpus Size (Documents)', fontweight='bold')
        ax.set_title('Corpus Growth Over Query Sequence', fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.output_dir, 'figure_4_corpus_growth')
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"    [OK] Saved: {output_path}.png/.pdf")
    
    def figure_5_ablation_study(self):
        """
        Figure 5: Ablation Study Results
        
        Shows impact of removing each safety mechanism.
        """
        print("\n[Figure 5] Ablation Study...")
        
        # Load ablation results
        ablation_path = os.path.join(self.results_dir, 'ablation_study_latest.json')
        try:
            ablation_data = load_results(ablation_path)
        except FileNotFoundError:
            print("    [SKIP] Ablation study data not available")
            return
        
        configurations = ablation_data.get('configurations', {})
        
        if not configurations:
            print("    [SKIP] No ablation configurations found")
            return
        
        # Prepare data
        config_names = list(configurations.keys())
        acceptance_rates = [c['acceptance_rate'] * 100 for c in configurations.values()]
        corpus_growths = [c['corpus_growth'] for c in configurations.values()]
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Acceptance Rates
        axes[0].barh(range(len(config_names)), acceptance_rates, color=COLORS[:len(config_names)])
        axes[0].set_yticks(range(len(config_names)))
        axes[0].set_yticklabels([c.replace('_', ' ').title() for c in config_names])
        axes[0].set_xlabel('Acceptance Rate (%)', fontweight='bold')
        axes[0].set_title('Acceptance Rates by Configuration', fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(acceptance_rates):
            axes[0].text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
        
        # Plot 2: Corpus Growth
        axes[1].barh(range(len(config_names)), corpus_growths, color=COLORS[:len(config_names)])
        axes[1].set_yticks(range(len(config_names)))
        axes[1].set_yticklabels([c.replace('_', ' ').title() for c in config_names])
        axes[1].set_xlabel('Corpus Growth (Documents)', fontweight='bold')
        axes[1].set_title('Corpus Growth by Configuration', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(corpus_growths):
            axes[1].text(v + 2, i, f'{v}', va='center', fontweight='bold')
        
        fig.suptitle('Ablation Study: Impact of Safety Mechanisms', fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.output_dir, 'figure_5_ablation_study')
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"    [OK] Saved: {output_path}.png/.pdf")


def main():
    """Generate all figures."""
    
    generator = FigureGenerator()
    
    # Load experiments
    print("\n[Step 1] Loading experiment results...")
    experiments = generator.load_experiments()
    
    if not experiments:
        print("[ERROR] No experiment results found!")
        print("Please run: python experiments/run_experiments.py")
        return
    
    # Generate figures
    print("\n[Step 2] Generating figures...")
    
    generator.figure_1_coverage_evolution(experiments)
    generator.figure_2_system_comparison(experiments)
    generator.figure_3_rejection_breakdown(experiments)
    generator.figure_4_corpus_growth(experiments)
    generator.figure_5_ablation_study()
    
    print("\n" + "="*60)
    print("FIGURE GENERATION COMPLETE")
    print("="*60)
    print(f"\nFigures saved to: {generator.output_dir}/")
    print(f"\nGenerated figures:")
    print(f"  1. Coverage evolution (line plot)")
    print(f"  2. System comparison (bar charts)")
    print(f"  3. Rejection breakdown (pie chart)")
    print(f"  4. Corpus growth (line plot)")
    print(f"  5. Ablation study (horizontal bars)")
    print(f"\nAll figures available in PNG (300 DPI) and PDF formats")


if __name__ == '__main__':
    main()

