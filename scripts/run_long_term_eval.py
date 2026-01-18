"""
Long-Term Evaluation Runner
===========================

Runs the 5000-query long-term learning experiment for Bidirectional RAG.
This evaluates the stability and learning trajectory of the Experience Store
over an extended period.

Usage:
    python scripts/run_long_term_eval.py --seed 42
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Run long-term evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-queries', type=int, default=5000, help='Total queries')
    parser.add_argument('--checkpoint-every', type=int, default=100, help='Checkpoint frequency')
    
    args = parser.parse_args()
    
    output_dir = f"results/long_term/seed_{args.seed}"
    
    print(f"\n{'='*60}")
    print(f"Starting Long-Term Evaluation (5000 Queries)")
    print(f"System: Bidirectional RAG")
    print(f"Corpus: Realistic (Wikipedia + SO)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    python_exe = sys.executable
    
    cmd = [
        python_exe, "main.py",
        "--systems", "bidirectional_rag",
        "--datasets", "nq", # NQ is best for open-domain questions
        "--corpus_type", "realistic",
        "--num_queries", str(args.num_queries),
        "--checkpoint_every", str(args.checkpoint_every),
        "--seeds", str(args.seed),
        "--output_dir", output_dir,
        "--offline",
        "--max_workers", "1" # STRICT SERIAL for long stability
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n[SUCCESS] Long-term evaluation complete.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    main()
