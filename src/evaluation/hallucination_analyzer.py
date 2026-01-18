import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.evaluation.metrics import MetricsCalculator


class HallucinationAnalyzer:
    """
    Aggregate hallucination rates across datasets/systems/seeds from query_logs.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.out_dir = self.results_dir / "hallucination_analysis"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def analyze_all_experiments(self, results_dir: str = None):
        base_dir = Path(results_dir) if results_dir else self.results_dir
        rows = []
        calculator = MetricsCalculator(corpus=[])

        datasets = [p for p in base_dir.iterdir() if p.is_dir()]
        for dataset_dir in tqdm(datasets, desc="Datasets"):
            dataset = dataset_dir.name
            systems = [p for p in dataset_dir.iterdir() if p.is_dir()]
            for system_dir in tqdm(systems, desc=f"{dataset} systems", leave=False):
                system = system_dir.name
                seeds = [p for p in system_dir.iterdir() if p.is_dir()]
                for seed_dir in seeds:
                    seed = seed_dir.name
                    query_logs = sorted((seed_dir / "query_logs").glob("query_*.json"))
                    if not query_logs:
                        continue

                    hall_rates: List[float] = []
                    for qfile in query_logs:
                        try:
                            with open(qfile, "r", encoding="utf-8") as f:
                                qd = json.load(f)
                            hall = calculator.compute_hallucination_rate(
                                query=qd.get("query", ""),
                                retrieved_docs=qd.get("retrieved_docs", []),
                                generated_response=qd.get("generated_response", ""),
                            )
                            hall_rates.append(hall)
                        except Exception as e:
                            self._log_error(f"{qfile}: {e}")

                    if hall_rates:
                        rows.append(
                            {
                                "dataset": dataset,
                                "system": system,
                                "seed": seed,
                                "hallucination_rate": float(np.mean(hall_rates)),
                                "std_error": float(np.std(hall_rates) / np.sqrt(len(hall_rates))),
                                "count": len(hall_rates),
                            }
                        )

        if not rows:
            print("[HallucinationAnalyzer] No query logs found.")
            return

        # Save CSV
        csv_path = self.out_dir / "hallucination_rates.csv"
        import csv

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["dataset", "system", "seed", "hallucination_rate", "std_error", "count"],
            )
            writer.writeheader()
            writer.writerows(rows)

        # Plot comparison (mean over seeds per system)
        self._plot_comparison(rows)
        print(f"[HallucinationAnalyzer] Saved results to {csv_path}")

    def _plot_comparison(self, rows: List[Dict]):
        # Aggregate by system
        by_system: Dict[str, List[float]] = {}
        for r in rows:
            by_system.setdefault(r["system"], []).append(r["hallucination_rate"])

        systems = sorted(by_system.keys())
        means = [np.mean(by_system[s]) for s in systems]
        std_errs = [np.std(by_system[s]) / np.sqrt(len(by_system[s])) for s in systems]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(systems))
        bars = ax.bar(x, means, yerr=std_errs, capsize=5, color="#2c7fb8", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=45, ha="right")
        ax.set_ylabel("Hallucination Rate (%)")
        ax.set_title("Hallucination Rate Comparison (lower is better)")
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        out_path = self.out_dir / "hallucination_comparison.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        fig.savefig(self.out_dir / "hallucination_comparison.pdf", bbox_inches="tight")
        plt.close(fig)

    def _log_error(self, message: str):
        log_path = self.out_dir / "hallucination_errors.log"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{message}\n")
        except Exception:
            pass


if __name__ == "__main__":
    HallucinationAnalyzer().analyze_all_experiments("results")
"""
Hallucination Analyzer
----------------------

Loads per-query logs and aggregates hallucination rates across
datasets/systems/seeds. Generates a CSV summary and a comparison plot.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation.metrics import MetricsCalculator


class HallucinationAnalyzer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "hallucination_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.error_log = self.output_dir / "hallucination_errors.log"

    def _log_error(self, msg: str):
        with open(self.error_log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def analyze_all_experiments(self, results_dir: str = None):
        base_dir = Path(results_dir) if results_dir else self.results_dir
        rows: List[Dict] = []

        # Preload a metrics calculator (minimal corpus placeholder)
        metrics_calc = MetricsCalculator(corpus=[])

        # Traverse dataset/system/seed
        dataset_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
        for dataset_path in dataset_dirs:
            dataset = dataset_path.name
            for system_path in dataset_path.iterdir():
                if not system_path.is_dir():
                    continue
                system = system_path.name
                for seed_path in system_path.iterdir():
                    if not seed_path.is_dir():
                        continue
                    seed = seed_path.name
                    query_logs = sorted((seed_path / "query_logs").glob("query_*.json"))
                    if not query_logs:
                        continue

                    rates = []
                    for qfile in tqdm(query_logs, desc=f"{dataset}/{system}/seed{seed}", leave=False):
                        try:
                            with open(qfile, "r", encoding="utf-8") as f:
                                qd = json.load(f)
                            rate = qd.get("hallucination_rate")
                            if rate is None:
                                rate = metrics_calc.compute_hallucination_rate(
                                    query=qd.get("query", ""),
                                    retrieved_docs=[d.get("text", "") for d in qd.get("retrieved_docs", [])],
                                    generated_response=qd.get("generated_response", "")
                                )
                            rates.append(rate)
                        except Exception as e:
                            self._log_error(f"{qfile}: {e}")

                    if not rates:
                        continue

                    rows.append({
                        "dataset": dataset,
                        "system": system,
                        "seed": seed,
                        "hallucination_rate": float(np.mean(rates)),
                        "num_queries": len(rates)
                    })

        if not rows:
            print("[HallucinationAnalyzer] No query logs found.")
            return

        df = pd.DataFrame(rows)
        csv_path = self.output_dir / "hallucination_rates.csv"
        df.to_csv(csv_path, index=False)
        print(f"[HallucinationAnalyzer] Saved summary to {csv_path}")

        # Aggregate per system
        sys_df = df.groupby("system")["hallucination_rate"].agg(['mean', 'std']).reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(sys_df["system"], sys_df["mean"], yerr=sys_df["std"], capsize=5)
        ax.set_ylabel("Hallucination Rate (%)")
        ax.set_xlabel("System")
        ax.set_title("Hallucination Rate Comparison (lower is better)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig_path = self.output_dir / "hallucination_comparison.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"[HallucinationAnalyzer] Saved plot to {fig_path}")


if __name__ == "__main__":
    analyzer = HallucinationAnalyzer()
    analyzer.analyze_all_experiments("results/")

