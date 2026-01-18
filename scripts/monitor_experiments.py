"""
Real-time experiment monitoring.

Run in separate terminal while experiments are running.

Usage:
    python scripts/monitor_experiments.py
"""

import time
from pathlib import Path
import json


def monitor_experiments():
    """Monitor experiment progress."""
    results_dir = Path("results")
    
    # Expected experiments: 3 systems x 4 datasets x 3 seeds = 36
    total_experiments = 3 * 4 * 3
    
    print("=" * 60)
    print("Experiment Monitor Started")
    print(f"Total expected experiments: {total_experiments}")
    print("=" * 60)
    print()
    
    while True:
        # Count completed experiments (those with metrics.json)
        completed = len(list(results_dir.rglob("metrics.json")))
        progress = (completed / total_experiments) * 100
        
        # Count experiments with query logs
        with_logs = len(list(results_dir.rglob("query_logs")))
        
        # Get breakdown by system
        systems = {}
        for metrics_file in results_dir.rglob("metrics.json"):
            parts = metrics_file.parts
            if len(parts) >= 4:
                system = parts[-3]
                systems[system] = systems.get(system, 0) + 1
        
        # Display status
        status = f"[OK] Completed: {completed}/{total_experiments} ({progress:.1f}%) | With logs: {with_logs}"
        
        print(f"\r{status}", end="", flush=True)
        
        # Check if done
        if completed >= total_experiments:
            print("\n")
            print("=" * 60)
            print("All experiments complete!")
            print("=" * 60)
            print("\nBreakdown by system:")
            for sys, count in sorted(systems.items()):
                print(f"  {sys}: {count} experiments")
            break
        
        time.sleep(10)  # Update every 10 seconds


if __name__ == "__main__":
    try:
        monitor_experiments()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

