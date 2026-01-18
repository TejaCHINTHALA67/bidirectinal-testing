"""
Trajectory Analysis Script
==========================

Generates trajectory plots for long-term experiments (5000 queries).
Visualizes:
1. Coverage (Grounding Score / Relevance) over time.
2. Corpus Composition (Growth of Experience Store).

Usage:
    python scripts/plot_trajectories.py --results-dir results/long_term/seed_42
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
from pathlib import Path

def load_trajectory_data(results_dir):
    """Load query logs and extract time-series data."""
    logs = []
    
    # Locate query logs
    # Pattern: results_dir/realistic/nq/bidirectional_rag/42/query_logs/*.json
    search_pattern = os.path.join(results_dir, '**', 'query_logs', '*.json')
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"No logs found in {results_dir}")
        return None
        
    print(f"Found {len(files)} log files. Loading...")
    
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                entry = json.load(fh)
                logs.append({
                    'query_index': entry.get('query_index', 0),
                    'grounding_score': entry.get('grounding_score', 0.0),
                    'accepted': entry.get('accepted', False),
                    # Need corpus size? We might infer it or it should be logged.
                    # If not logged, we count cumulative accepted (assuming 1 doc added per accept)
                    'timestamp': entry.get('timestamp')
                })
        except:
            pass
            
    df = pd.DataFrame(logs)
    if not df.empty:
        df = df.sort_values('query_index')
        df['cumulative_experiences'] = df['accepted'].cumsum()
        
        # Rolling average for grounding score
        df['grounding_ma_50'] = df['grounding_score'].rolling(window=50).mean()
        
    return df

def plot_trajectories(df, output_dir):
    """Generate plots."""
    if df is None or df.empty:
        return
        
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Corpus Growth
    plt.figure(figsize=(10, 6))
    plt.plot(df['query_index'], df['cumulative_experiences'], label='Synthetic Experiences', color='blue')
    plt.title('Experience Store Growth over Time')
    plt.xlabel('Queries Processed')
    plt.ylabel('Number of Stored Experiences')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path / 'corpus_growth.png')
    plt.close()
    
    # 2. Coverage / Quality (Grounding Score)
    plt.figure(figsize=(10, 6))
    plt.plot(df['query_index'], df['grounding_score'], alpha=0.2, color='gray', label='Raw Score')
    plt.plot(df['query_index'], df['grounding_ma_50'], color='red', linewidth=2, label='Moving Avg (50)')
    plt.title('Grounding Score Trajectory (Coverage proxy)')
    plt.xlabel('Queries Processed')
    plt.ylabel('Grounding Score (0-1)')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path / 'grounding_trajectory.png')
    plt.close()
    
    print(f"Plots saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True, help='Path to experiment results')
    args = parser.parse_args()
    
    df = load_trajectory_data(args.results_dir)
    if df is not None:
        plot_trajectories(df, args.results_dir)

if __name__ == '__main__':
    main()
