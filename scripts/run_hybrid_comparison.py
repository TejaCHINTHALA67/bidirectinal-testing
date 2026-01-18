"""
Hybrid Baseline Comparison Runner
=================================

Runs head-to-head comparison between Bidirectional RAG and 
Hybrid Baselines (Self-RAG+WB, FLARE+WB, CRAG+WB).

Goal: Prove that Bidirectional RAG's acceptance layer provides 
value over simply adding write-back to SOTA retrieval methods.

Usage:
    python scripts/run_hybrid_comparison.py --num-queries 50 --seed 42
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Run hybrid comparison')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-queries', type=int, default=50, help='Queries per system')
    parser.add_argument('--datasets', nargs='+', default=['nq'], help='Datasets')
    
    args = parser.parse_args()
    
    # Systems to compare
    systems = [
        'bidirectional_rag',
        'self_rag_wb',
        'flare_wb', 
        'crag_wb'
    ]
    
    output_dir = f"results/hybrid_comparison/seed_{args.seed}"
    
    print(f"\n{'='*60}")
    print(f"Starting Hybrid Comparison Experiment")
    print(f"Systems: {systems}")
    print(f"Corpus: Realistic")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    python_exe = sys.executable
    
    # Run all systems in one go (or loop if memory issues)
    # Using main.py orchestration
    cmd = [
        python_exe, "main.py",
        "--systems", *systems,
        "--datasets", *args.datasets,
        "--corpus_type", "realistic",
        "--num_queries", str(args.num_queries),
        "--seeds", str(args.seed),
        "--output_dir", output_dir,
        "--offline",
        "--max_workers", "1" # Serial execution for fair comparison / safety
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n[SUCCESS] Hybrid comparison complete.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    main()
