"""
Recompute Answer F1 / EM using improved extraction + normalization.

Usage:
    python scripts/recompute_answer_f1.py --results_dir results
"""

import argparse
import json
from pathlib import Path

from src.evaluation.metrics import MetricsCalculator


def recompute_answer_f1(results_dir: str = "results"):
    base = Path(results_dir)
    calc = MetricsCalculator(corpus=[])
    updated = 0

    for metrics_path in base.rglob("metrics.json"):
        test_results_path = metrics_path.with_name("test_results.json")
        if not test_results_path.exists():
            continue

        with open(test_results_path, "r", encoding="utf-8") as f:
            test_results = json.load(f)

        # Recompute metrics (uses improved F1/EM)
        metrics = calc.compute_all(test_results)

        # Preserve any existing fields not recomputed (e.g., latency)
        with open(metrics_path, "r", encoding="utf-8") as f:
            old_metrics = json.load(f)
        for k, v in old_metrics.items():
            if k not in metrics:
                metrics[k] = v

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        with open(test_results_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)

        updated += 1

    print(f"[recompute_answer_f1] Updated {updated} experiments.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results", type=str)
    args = parser.parse_args()
    recompute_answer_f1(args.results_dir)

