"""
Master Experiment Runner for IEEE-Grade RAG Experiments

Orchestrates comprehensive experiments across systems, datasets, and seeds.

Usage:
    python main.py --systems bidirectional_rag standard_rag --datasets stackoverflow --num_queries 1000 --seed 42
    python main.py --systems all --datasets all --seeds 42 43 44 45 46 --num_queries 1000
"""

# CRITICAL: Fix ONNX import issue BEFORE any other imports
import fix_onnx_import

import argparse
import json
import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tqdm import tqdm


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive RAG experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--systems',
        nargs='+',
        default=['all'],
        help='Systems to evaluate (simplified to 3 core baselines): all, standard_rag, naive_writeback, bidirectional_rag'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['nq', 'triviaqa', 'hotpotqa', 'stackoverflow'],
        help='Datasets to use: nq, triviaqa, hotpotqa, stackoverflow, or all'
    )
    
    parser.add_argument(
        '--corpus_type',
        type=str,
        default='sparse',
        choices=['sparse', 'moderate', 'realistic'],
        help='Corpus initialization: sparse (2024 SO), moderate (+500 Wiki), realistic (+2000 Wiki)'
    )

    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=list(range(42, 62)),  # 20 seeds: 42-61
        help='Random seeds for reproducibility (20 seeds for statistical power)'
    )
    
    parser.add_argument(
        '--num_queries',
        type=int,
        default=1000,
        help='Total number of queries (800 train + 200 test)'
    )
    
    parser.add_argument(
        '--checkpoint_every',
        type=int,
        default=250,
        help='Save checkpoint every N queries'
    )
    
    parser.add_argument(
        '--max_workers',
        type=int,
        default=None,
        help='Maximum parallel workers (default: CPU count)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip completed experiments'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--offline',
        action='store_true',
        help='Enable offline mode (HF_HUB_OFFLINE=1)'
    )
    
    parser.add_argument(
        '--disable_experience_store',
        action='store_true',
        help='Disable experience store (ablation study)'
    )
    
    parser.add_argument(
        '--llm_model',
        type=str,
        default='llama3.2:3b',
        help='Ollama model to use (e.g. llama3.2:3b, llama3.1:8b)'
    )
    
    return parser.parse_args()


def get_all_systems() -> List[str]:
    """
    Get list of all available systems.
    
    IEEE Access paper compares 6 systems:
    1. Standard RAG - Traditional retrieve-and-generate with static corpus
    2. Self-RAG - Adaptive retrieval with reflection tokens
    3. FLARE - Generate-then-retrieve on uncertainty  
    4. CRAG - Corrective retrieval via relevance evaluation
    5. Naive Write-back - Write-back without validation
    6. Bidirectional RAG (ours) - Validated write-back with multi-stage acceptance
    """
    return [
        'standard_rag',
        'self_rag',
        'flare',
        'crag',
        'naive_writeback',
        'bidirectional_rag'
    ]


def get_all_datasets() -> List[str]:
    """Get list of all available datasets."""
    return ['nq', 'triviaqa', 'hotpotqa', 'stackoverflow']


def expand_systems(systems: List[str]) -> List[str]:
    """Expand 'all' to list of all systems."""
    if 'all' in systems:
        return get_all_systems()
    return systems


def expand_datasets(datasets: List[str]) -> List[str]:
    """Expand 'all' to list of all datasets."""
    if 'all' in datasets:
        return get_all_datasets()
    return datasets


def is_experiment_complete(output_dir: str, lambda_corpus_type: str, dataset: str, system: str, seed: int) -> bool:
    """Check if experiment is already complete."""
    result_dir = Path(output_dir) / lambda_corpus_type / dataset / system / str(seed)
    metrics_file = result_dir / 'metrics.json'
    return metrics_file.exists() and metrics_file.stat().st_size > 0


def run_single_experiment(
    dataset: str,
    system: str,
    seed: int,
    corpus_type: str,
    num_queries: int,
    checkpoint_every: int,
    output_dir: str,
    offline: bool,
    disable_experience_store: bool = False,
    llm_model: str = "llama3.2:3b"
) -> Dict[str, Any]:
    """
    Run a single experiment configuration.
    
    Returns:
        Dictionary with experiment results or error information
    """
    # Set offline mode if requested
    if offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
    # Import here to avoid issues with multiprocessing
    from src.data.dataset_loader import DatasetLoader
    from src.systems.baselines import get_system_class
    from src.evaluation.metrics import MetricsCalculator
    import json
    from experiments.corpus_configurations import CorpusConfigurator

    # CRITICAL FIX: The entire original function body was indented under 'if offline:'.
    # To fix the 'return None' bug without rewriting 200 lines of indentation,
    # we wrap the execution logic in 'if True:'.
    if True:
        
        # Load dataset queries only (we'll swap the corpus)
        loader = DatasetLoader(cache_dir='data/raw')
        _, queries = loader.load_and_process(dataset)
        
        # Load Configured Corpus
        configurator = CorpusConfigurator(data_dir='data')
        corpus_data = configurator.get_corpus(corpus_type, seed=seed)
        
        # Save corpus to ensure path exists for system init
        # We save it to a specific run path to avoid overwrites if running parallel diff configs
        corpus_path = Path(f'data/processed/{dataset}_{corpus_type}_{seed}_corpus.json')
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, indent=2, ensure_ascii=False)
        
        # Limit queries to num_queries if specified
        import random
        random.seed(seed)
        random.shuffle(queries)
        if num_queries > 0 and num_queries < len(queries):
            queries = queries[:num_queries]
        
        # Split queries (80/20 train/test)
        split_idx = int(len(queries) * 0.8)
        train_queries = queries[:split_idx]
        test_queries = queries[split_idx:]

        # Prepare output directories early
        # Configurable output path: results/sparse/nq/standard_rag/42
        # Prepare output directories early
        # Configurable output path: results/sparse/nq/standard_rag/42
        result_dir = Path(output_dir) / corpus_type / dataset / system / str(seed)
        query_logs_dir = result_dir / 'query_logs'
        query_logs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics calculator once (reused for hallucination scoring)
        metrics_calc = MetricsCalculator(corpus=corpus_data)
        
        # Initialize system with metadata for query persistence
        SystemClass = get_system_class(system)
        system_output_dir = Path(output_dir) / corpus_type
        system_instance = SystemClass(
            corpus_path=str(corpus_path),
            dataset_name=dataset,
            seed=seed,
            system_slug=system,
            output_dir=str(system_output_dir),
            enable_query_logging=True,
            enable_experience_store=not disable_experience_store,
            llm_model=llm_model,
        )
        
        # Process training queries
        training_results = []
        print(f"[{dataset}/{system}/seed_{seed}] Processing {len(train_queries)} training queries...")
        for i, query_data in enumerate(train_queries):
            result = system_instance.query(query_data['question'])
            training_results.append(result)
            
            # Print progress every 10 queries or at checkpoints
            if (i + 1) % 10 == 0 or (i + 1) % checkpoint_every == 0:
                print(f"  [{dataset}/{system}/seed_{seed}] Training query {i+1}/{len(train_queries)}")
            
            # Checkpoint
            if (i + 1) % checkpoint_every == 0:
                save_checkpoint(
                    output_dir, dataset, system, seed,
                    i + 1, training_results, system_instance
                )
        
        # Final checkpoint
        save_checkpoint(
            output_dir, dataset, system, seed,
            len(train_queries), training_results, system_instance
        )
        
        # Evaluate on test set
        print(f"[{dataset}/{system}/seed_{seed}] Evaluating on {len(test_queries)} test queries...")
        test_results = []
        for i, query_data in enumerate(test_queries):
            result = system_instance.query(query_data['question'])
            # top_distance should already be in result from _retrieve()
            # If not present, estimate from retrieved_docs presence
            if 'top_distance' not in result or result.get('top_distance') is None:
                if 'retrieved_docs' in result and result['retrieved_docs']:
                    result['top_distance'] = 0.3  # Assume relevant if retrieved
                else:
                    result['top_distance'] = 1.0  # No retrieval = no coverage
            # Compute hallucination rate per query
            hallucination_rate = metrics_calc.compute_hallucination_rate(
                query=query_data['question'],
                retrieved_docs=result.get('retrieved_docs', []),
                generated_response=result.get('response', '')
            )
            result['hallucination_rate'] = hallucination_rate

            enriched_result = {
                **result,
                'ground_truth': query_data.get('answer', '')
            }
            test_results.append(enriched_result)

            # Persist full query log for later analysis
            try:
                # Use retrieved_ids if present, otherwise enumerate
                retrieved_ids = result.get('retrieved_ids') or [
                    f"doc_{idx}" for idx, _ in enumerate(result.get('retrieved_docs', []))
                ]
                retrieved_docs = result.get('retrieved_docs', [])
                retrieved_docs_struct = [
                    {"id": rid, "text": doc}
                    for rid, doc in zip(retrieved_ids, retrieved_docs)
                ]

                query_log = {
                    "query": query_data['question'],
                    "query_index": i,
                    "dataset": dataset,
                    "system": system,
                    "seed": seed,
                    "retrieved_docs": retrieved_docs_struct,
                    "generated_response": result.get('response', ''),
                    "citations": result.get('citations', []),
                    "ground_truth": query_data.get('answer', ''),
                    "hallucination_rate": hallucination_rate,
                    "grounding_score": result.get('grounding_score'),
                    "rejection_reason": result.get('rejection_reason'),
                    "timestamp": time.time()
                }

                with open(query_logs_dir / f"query_{i}.json", "w", encoding="utf-8") as f:
                    json.dump(query_log, f, indent=2)
            except Exception as e:
                print(f"[WARN] Failed to save query log {i}: {e}")
            
            # Print progress every 10 queries
            if (i + 1) % 10 == 0:
                print(f"  [{dataset}/{system}/seed_{seed}] Test query {i+1}/{len(test_queries)}")
        
        # Compute metrics
        print(f"[{dataset}/{system}/seed_{seed}] Computing metrics...")
        metrics = metrics_calc.compute_all(test_results)
        hall_values = [tr.get('hallucination_rate', 0.0) for tr in test_results]
        metrics['hallucination_rate'] = float(np.mean(hall_values)) if hall_values else 0.0
        
        # Add system-specific metrics
        if hasattr(system_instance, 'get_statistics'):
            stats = system_instance.get_statistics()
            metrics.update({
                'corpus_growth': stats.get('documents_added', 0),
                'acceptance_rate': stats.get('acceptance_rate', 0.0)
            })
        
        # Save results
        print(f"[{dataset}/{system}/seed_{seed}] Saving results...")
        result_dir = Path(output_dir) / dataset / system / str(seed)
        result_dir.mkdir(parents=True, exist_ok=True)
        
        with open(result_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save test results for statistical analysis
        with open(result_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        with open(result_dir / 'config.json', 'w') as f:
            json.dump({
                'dataset': dataset,
                'system': system,
                'seed': seed,
                'num_queries': num_queries,
                'checkpoint_every': checkpoint_every,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"[{dataset}/{system}/seed_{seed}] ✓ Experiment complete!")
        
        return {
            'status': 'success',
            'dataset': dataset,
            'system': system,
            'seed': seed,
            'metrics': metrics
        }
        



def save_checkpoint(
    output_dir: str,
    dataset: str,
    system: str,
    seed: int,
    query_index: int,
    results: List[Dict],
    system_instance: Any
):
    """Save checkpoint during experiment."""
    result_dir = Path(output_dir) / dataset / system / str(seed) / 'checkpoints'
    result_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'query_index': query_index,
        'timestamp': datetime.now().isoformat(),
        'num_results': len(results)
    }
    
    if hasattr(system_instance, 'get_statistics'):
        stats = system_instance.get_statistics()
        checkpoint.update(stats)
    
    with open(result_dir / f'checkpoint_{query_index}.json', 'w') as f:
        json.dump(checkpoint, f, indent=2)


def main():
    """Main execution."""
    args = parse_arguments()
    
    # Expand 'all' options
    systems = expand_systems(args.systems)
    datasets = expand_datasets(args.datasets)
    
    # Create experiment grid
    experiments = []
    for dataset in datasets:
        for system in systems:
            for seed in args.seeds:
                # Skip if resume and already complete
                if args.resume and is_experiment_complete(
                    args.output_dir, args.corpus_type, dataset, system, seed
                ):
                    print(f"[SKIP] {args.corpus_type}/{dataset}/{system}/seed_{seed} (already complete)")
                    continue
                
                experiments.append((dataset, system, seed))
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT RUNNER")
    print(f"{'='*70}")
    print(f"Systems: {len(systems)} ({', '.join(systems)})")
    print(f"Datasets: {len(datasets)} ({', '.join(datasets)})")
    print(f"Seeds: {len(args.seeds)} ({', '.join(map(str, args.seeds))})")
    print(f"Total Experiments: {len(experiments)}")
    train_count = int(args.num_queries * 0.8) if args.num_queries > 0 else 0
    test_count = args.num_queries - train_count if args.num_queries > 0 else 0
    print(f"Queries per Experiment: {args.num_queries} ({train_count} train + {test_count} test)")
    print(f"Output Directory: {args.output_dir}")
    print(f"{'='*70}\n")
    
    if not experiments:
        print("No experiments to run (all complete or invalid configuration)")
        return
    
    # Run experiments
    max_workers = args.max_workers or os.cpu_count() or 1
    print(f"Running {len(experiments)} experiments with {max_workers} workers...\n")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_exp = {
            executor.submit(
                run_single_experiment,
                dataset, system, seed,
                args.corpus_type,
                args.num_queries,
                args.checkpoint_every,
                args.output_dir,
                args.offline,
                args.disable_experience_store,
                args.llm_model
            ): (dataset, system, seed)
            for dataset, system, seed in experiments
        }
        
        # Process results with progress bar
        with tqdm(total=len(experiments), desc="Experiments") as pbar:
            for future in as_completed(future_to_exp):
                exp = future_to_exp[future]
                try:
                    result = future.result()
                    
                    if result is None:
                        raise RuntimeError("Experiment returned None (implicit return?)")

                    results.append(result)
                    
                    if result['status'] == 'success':
                        pbar.set_postfix({
                            'last': f"{result['dataset']}/{result['system']}/s{result['seed']}",
                            'success': sum(1 for r in results if r['status'] == 'success')
                        })
                    else:
                        print(f"\n[ERROR] {exp[0]}/{exp[1]}/seed_{exp[2]}: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"\n[FATAL] {exp[0]}/{exp[1]}/seed_{exp[2]}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Only append failure if we didn't already append success
                    # (This is tricky if append happened above. But here exception matches [FATAL])
                    # We assume if exception caught here, result wasn't appended successfully above
                    results.append({
                        'status': 'fatal',
                        'dataset': exp[0],
                        'system': exp[1],
                        'seed': exp[2],
                        'error': str(e)
                    })
                
                pbar.update(1)
    
    # Summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\nFailed Experiments:")
        for r in results:
            if r['status'] != 'success':
                print(f"  - {r['dataset']}/{r['system']}/seed_{r['seed']}: {r.get('error', 'Unknown')}")
    
    # Save summary
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.output_dir) / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(experiments),
            'successful': successful,
            'failed': failed,
            'results': results
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    
    # Run post-experiment statistical analysis
    if successful > 0:
        print("\n" + "="*70)
        print("RUNNING POST-EXPERIMENT STATISTICAL ANALYSIS")
        print("="*70)
        
        try:
            from src.evaluation.statistical_analysis import StatisticalAnalyzer, load_results_from_directory
            
            analyzer = StatisticalAnalyzer(alpha=0.05)
            results_dir = Path(args.output_dir)
            
            # Load all results
            all_system_results = load_results_from_directory(results_dir)
            
            if len(all_system_results) >= 2:
                # Compare systems on key metrics
                metrics_to_compare = ['coverage', 'f1_score', 'grounding_score', 'latency_ms']
                
                comparison_results = analyzer.compare_multiple_systems(
                    all_system_results,
                    metrics=metrics_to_compare
                )
                
                # Generate report
                report_path = results_dir / 'statistical_analysis_report.txt'
                report = analyzer.generate_report(comparison_results, output_path=report_path)
                
                print(f"\n[OK] Statistical analysis complete")
                print(f"Report saved to: {report_path}")
                
                # Save JSON results
                analysis_json_path = results_dir / 'statistical_analysis.json'
                with open(analysis_json_path, 'w') as f:
                    json.dump(comparison_results, f, indent=2, default=str)
                print(f"Results saved to: {analysis_json_path}")
            else:
                print(f"[WARN] Need at least 2 systems for comparison (found {len(all_system_results)})")
                
        except Exception as e:
            print(f"[WARN] Statistical analysis failed: {e}")
            print("You can run it manually later:")
            print("  python -c \"from src.evaluation.statistical_analysis import StatisticalAnalyzer; ...\"")
    
    print("\n" + "="*70)
    print("EXPERIMENT RUNNER COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review results in:", args.output_dir)
    print("  2. Check statistical analysis report")
    print("  3. Generate figures: python experiments/generate_publication_figures.py")
    print("  4. Generate LaTeX tables: python experiments/generate_latex_tables.py")


if __name__ == '__main__':
    main()

