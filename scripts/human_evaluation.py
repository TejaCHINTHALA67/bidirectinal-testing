"""
Human Evaluation Sampling Script
================================

Randomly samples query-response pairs from experiment results for human evaluation.
Generates an anonymized CSV/JSON form for blind review.
"""

import argparse
import json
import os
import random
import csv
from pathlib import Path
from typing import List, Dict, Any

def load_system_outputs(results_dir: Path, systems: List[str]) -> Dict[str, List[Dict]]:
    """Load results for specified systems."""
    loaded_data = {}
    
    for system in systems:
        # Search for latest result file
        # Pattern: results_dir/.../system/.../query_logs ?
        # Or look for aggregated results if available.
        # Let's assume we can walk the directory and find query logs.
        
        logs = []
        for root, dirs, files in os.walk(results_dir):
             if system in root and 'query_logs' in dirs:
                 log_dir = Path(root) / 'query_logs'
                 # Load up to 50 logs per seed/shard
                 for jf in list(log_dir.glob('*.json'))[:50]:
                     try:
                         with open(jf, 'r', encoding='utf-8') as f:
                             d = json.load(f)
                             d['system_id'] = system
                             logs.append(d)
                     except: 
                        pass
        
        if logs:
            loaded_data[system] = logs
        else:
            print(f"Warning: No logs found for system {system}")
            
    return loaded_data

def generate_evaluation_form(
    data: Dict[str, List[Dict]], 
    output_file: str, 
    samples_per_system: int = 50
):
    """Generate blind evaluation form (CSV)."""
    
    # Flatten and sample
    all_samples = []
    
    for system, logs in data.items():
        if len(logs) > samples_per_system:
            selection = random.sample(logs, samples_per_system)
        else:
            selection = logs
            
        for log in selection:
            # Create evaluation entry
            entry = {
                'id': f"{system[:3]}_{log.get('query_hash', random.randint(1000,9999))}",
                'system_hidden': system, # Hidden from annotator if possible, but kept for matching
                'query': log.get('query', ''),
                'response': log.get('generated_response', ''),
                'retrieved_context_summary': " | ".join([d['text'][:100] + "..." for d in log.get('retrieved_docs', [])])[:500],
                # Rating columns
                'relevance_1_5': '',
                'factuality_1_5': '',
                'coherence_1_5': '',
                'comments': ''
            }
            all_samples.append(entry)
            
    # Shuffle to blind specific system blocks (though system_hidden reveals it if looked at)
    random.shuffle(all_samples)
    
    # Write to CSV
    keys = ['id', 'query', 'response', 'retrieved_context_summary', 'relevance_1_5', 'factuality_1_5', 'coherence_1_5', 'comments', 'system_hidden']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_samples)
        
    print(f"Generated {len(all_samples)} samples in {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate human evaluation set')
    parser.add_argument('--results-dir', type=str, required=True, help='Root results directory')
    parser.add_argument('--systems', nargs='+', default=['bidirectional_rag', 'standard_rag', 'naive_writeback'], help='Systems to compare')
    parser.add_argument('--output', type=str, default='human_eval_form.csv', help='Output CSV file')
    parser.add_argument('--sample-size', type=int, default=20, help='Samples per system')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    data = load_system_outputs(Path(args.results_dir), args.systems)
    if data:
        generate_evaluation_form(data, args.output, args.sample_size)
    else:
        print("No data found to sample.")

if __name__ == '__main__':
    main()
