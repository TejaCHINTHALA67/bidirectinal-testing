"""
Model Scaling Experiments
=========================

Runs comparative experiments between small (3B) and medium (8B) models
to quantify the impact of model size on RAG performance and Hallucination rates.

Configurations:
- Small: llama3.2:3b
- Medium: llama3.1:8b (or equivalent available 8b model)

Usage:
    python scripts/model_scaling_experiments.py --num-queries 50 --seeds 42 43
"""

import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path

def run_experiment(model_name: str, output_dir: str, num_queries: int, seeds: list):
    """Run experiment for a specific model."""
    print(f"\n{'='*60}")
    print(f"Running Scaling Experiment: {model_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    # Construct command
    # Using venv python if available
    python_exe = sys.executable
    
    cmd = [
        python_exe, "main.py",
        "--systems", "bidirectional_rag",
        "--datasets", "nq",
        "--corpus_type", "sparse",
        "--num_queries", str(num_queries),
        "--seeds", *map(str, seeds),
        "--output_dir", output_dir,
        "--llm_model", model_name,
        "--offline",
        "--max_workers", "1" # Serial execution for safety
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] Completed run for {model_name}")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed run for {model_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run model scaling experiments')
    parser.add_argument('--num-queries', type=int, default=50, help='Number of queries per run')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42], help='Seeds to run')
    parser.add_argument('--models', type=str, nargs='+', default=['llama3.2:3b', 'llama3.1:8b'], help='Models to compare')
    parser.add_argument('--output-root', type=str, default='results/scaling', help='Root output directory')
    
    args = parser.parse_args()
    
    # Run for each model
    for model in args.models:
        # Sanitize model name for directory
        model_slug = model.replace(':', '_').replace('.', '_')
        output_dir = os.path.join(args.output_root, model_slug)
        
        run_experiment(model, output_dir, args.num_queries, args.seeds)

if __name__ == '__main__':
    # Ensure we are in project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    main()
