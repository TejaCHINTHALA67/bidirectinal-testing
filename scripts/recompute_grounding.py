"""
Recompute grounding scores using probability-based NLI (softmax).

Usage:
    python scripts/recompute_grounding.py --results_dir results
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Ensure offline mode if needed
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.evaluation.metrics import MetricsCalculator


def recompute_grounding(results_dir: str = "results"):
    base = Path(results_dir)
    calculator = MetricsCalculator(corpus=[])
    updated = 0

    for metrics_path in base.rglob("metrics.json"):
        test_results_path = metrics_path.with_name("test_results.json")
        if not test_results_path.exists():
            continue

        try:
            with open(test_results_path, "r", encoding="utf-8") as f:
                test_results = json.load(f)
        except Exception:
            continue

        # Recompute per-query grounding
        new_scores = []
        for r in test_results:
            ctx = r.get("retrieved_docs", [])
            # retrieved_docs may be list of strings or dicts
            if ctx and isinstance(ctx[0], dict):
                ctx = [d.get("text", "") for d in ctx]
            score = calculator._compute_single_grounding(
                r.get("response", ""),
                ctx or []
            )
            r["grounding_score"] = score
            new_scores.append(score)

        if not new_scores:
            continue

        # Update metrics.json
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        metrics["grounding_score"] = float(np.mean(new_scores))

        with open(test_results_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        updated += 1

    print(f"[recompute_grounding] Updated {updated} experiments.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results", type=str)
    args = parser.parse_args()
    recompute_grounding(args.results_dir)

