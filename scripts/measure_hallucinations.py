
import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.hallucination_metrics import HallucinationEvaluator

def load_query_logs(results_dir: Path, sample_size: int = 50) -> Dict[str, List[Dict]]:
    """
    Load query logs grouped by system.
    """
    system_logs = {}
    
    # Walk through results directory
    for root, dirs, files in os.walk(results_dir):
        if 'query_logs' in dirs:
            log_dir = Path(root) / 'query_logs'
            # Determine system from path structure: .../system/seed/query_logs
            try:
                # Path(root) is .../system/seed
                system = log_dir.parent.parent.name
                seed = log_dir.parent.name
                
                # Check if system is relevant
                if system not in system_logs:
                    system_logs[system] = []
                
                # Load JSONs
                json_files = list(log_dir.glob('*.json'))
                if not json_files:
                    continue
                    
                # Sample if needed
                if len(json_files) > sample_size // 5: # Distribute sample across seeds? 
                    # Just load all and sample later globally
                    pass
                
                for jf in json_files:
                    try:
                        with open(jf, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            data['seed'] = seed # Inject seed
                            system_logs[system].append(data)
                    except:
                        pass
                        
            except Exception as e:
                print(f"Error processing {log_dir}: {e}")
                
    # Sample globally per system
    for system in system_logs:
        if len(system_logs[system]) > sample_size:
            random.seed(42)
            system_logs[system] = random.sample(system_logs[system], sample_size)
            
    return system_logs

def main():
    parser = argparse.ArgumentParser(description='Measure hallucination rates using LLM-based FActScore')
    parser.add_argument('--results-dir', type=str, required=True, help='Path to results directory (e.g. results/sparse/nq)')
    parser.add_argument('--output-file', type=str, default='hallucination_analysis_results.json', help='Output JSON file')
    parser.add_argument('--sample-size', type=int, default=50, help='Number of queries to evaluate per system')
    parser.add_argument('--model', type=str, default='llama3.2:3b', help='Ollama model to use')
    
    args = parser.parse_args()
    
    print(f"Loading query logs from {args.results_dir}...")
    system_logs = load_query_logs(Path(args.results_dir), args.sample_size)
    
    if not system_logs:
        print("No query logs found!")
        return
        
    print(f"Found systems: {list(system_logs.keys())}")
    
    evaluator = HallucinationEvaluator(model_name=args.model)
    results = {}
    
    for system, logs in system_logs.items():
        print(f"\nEvaluating system: {system} ({len(logs)} samples)")
        
        generations = [log['generated_response'] for log in logs]
        
        # Extract source texts from retrieved_docs structure
        # retrieved_docs is list of {id, text} dicts
        knowledge_sources = []
        for log in logs:
            docs = log.get('retrieved_docs', [])
            texts = [d['text'] for d in docs] if docs else []
            knowledge_sources.append(texts)
            
        print("  Computing FActScore (this may take a while)...")
        scores = evaluator.compute_fact_score(generations, knowledge_sources)
        
        # Aggregate
        fact_scores = [s['fact_score'] for s in scores]
        avg_fact_score = np.mean(fact_scores) if fact_scores else 0.0
        
        print(f"  Average FActScore: {avg_fact_score:.3f}")
        
        results[system] = {
            'samples': len(logs),
            'avg_fact_score': avg_fact_score,
            'scores': scores,
            'logs': logs  # Include logs? Too big.
        }
        
    # Save
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {args.output_file}")

if __name__ == '__main__':
    main()
