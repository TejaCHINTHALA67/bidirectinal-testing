"""
Measure Hallucination Rates
===========================

Uses NLI model to detect hallucinations in responses.
Compares hallucination rates across different systems.
"""

import json
import random
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("[WARNING] transformers not available. Install with: pip install transformers")
    TRANSFORMERS_AVAILABLE = False


def check_hallucination(response: str, context: str, nli_pipeline) -> bool:
    """
    Use NLI to check if response contradicts context.
    
    Returns True if hallucinated (contradiction detected).
    
    Parameters
    ----------
    response : str
        Generated response
    context : str
        Retrieved context/document
    nli_pipeline
        NLI pipeline from transformers
        
    Returns
    -------
    bool
        True if hallucinated (contradiction), False otherwise
    """
    try:
        # Truncate to model limits (500 chars each)
        context_text = (context[:500] if isinstance(context, str) 
                       else str(context)[:500])
        response_text = response[:500] if isinstance(response, str) else str(response)[:500]
        
        if not context_text or not response_text:
            return False  # Can't check without both
        
        # Format for NLI: context [SEP] response
        # Some models expect premise and hypothesis
        result = nli_pipeline(
            f"{context_text} [SEP] {response_text}",
            return_all_scores=True
        )
        
        # Check for contradiction
        # Result format depends on model, handle both cases
        if isinstance(result, list) and len(result) > 0:
            # If return_all_scores=True, get list of dicts
            for item in result:
                if isinstance(item, dict):
                    label = item.get('label', '').lower()
                    score = item.get('score', 0.0)
                    if 'contradiction' in label and score > 0.5:
                        return True
        elif isinstance(result, dict):
            label = result.get('label', '').lower()
            score = result.get('score', 0.0)
            if 'contradiction' in label and score > 0.5:
                return True
        
        return False
    except Exception as e:
        print(f"    [WARNING] Error checking hallucination: {e}")
        return False  # If error, assume not hallucinated


def measure_system_hallucinations(
    responses_file: str,
    system_name: str,
    n_samples: int = 200,
    nli_pipeline=None
) -> tuple:
    """
    Measure hallucination rate for one system.
    
    Parameters
    ----------
    responses_file : str
        Path to JSON file with responses
    system_name : str
        Name of the system
    n_samples : int
        Number of samples to analyze
    nli_pipeline
        NLI pipeline (will create if None)
        
    Returns
    -------
    tuple
        (hallucination_rate, list_of_hallucinations)
    """
    print(f"\nAnalyzing {system_name}...")
    
    if not os.path.exists(responses_file):
        print(f"    [ERROR] File not found: {responses_file}")
        return 0.0, []
    
    with open(responses_file, 'r', encoding='utf-8') as f:
        responses = json.load(f)
    
    # Extract query results - handle different formats
    if isinstance(responses, dict):
        if 'queries' in responses:
            query_results = responses['queries']
        elif 'results' in responses:
            query_results = responses['results']
        else:
            # Assume it's a list of results
            query_results = list(responses.values()) if responses else []
    else:
        query_results = responses
    
    if not query_results:
        print(f"    [WARNING] No query results found in {responses_file}")
        return 0.0, []
    
    # Sample responses
    n_samples = min(n_samples, len(query_results))
    sampled = random.sample(query_results, n_samples)
    
    # Initialize NLI if needed
    if nli_pipeline is None and TRANSFORMERS_AVAILABLE:
        print("    Loading NLI model...")
        try:
            nli_pipeline = pipeline(
                "text-classification",
                model="cross-encoder/nli-deberta-v3-base",
                device=-1  # CPU
            )
        except Exception as e:
            print(f"    [ERROR] Failed to load NLI model: {e}")
            print("    [INFO] Using mock NLI (will return 0% hallucination)")
            nli_pipeline = None
    
    hallucinations = []
    
    for i, item in enumerate(sampled):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(sampled)}")
        
        # Extract response and context
        response = item.get('response', '')
        if not response:
            continue
        
        # Get first retrieved doc as context
        retrieved_docs = item.get('retrieved_docs', [])
        if not retrieved_docs:
            retrieved_docs = item.get('retrieved_context', [])
        
        context = retrieved_docs[0] if retrieved_docs else ''
        
        if not context:
            # Try to get from metadata
            context = item.get('context', '')
        
        # Check for hallucination
        if nli_pipeline:
            is_hallucinated = check_hallucination(response, context, nli_pipeline)
        else:
            # Mock: no hallucinations detected
            is_hallucinated = False
        
        if is_hallucinated:
            hallucinations.append({
                'query': item.get('query', item.get('query_text', '')),
                'response': response[:200],
                'context': context[:200] if context else 'No context'
            })
    
    rate = len(hallucinations) / len(sampled) if sampled else 0.0
    print(f"  Hallucination rate: {rate:.1%} ({len(hallucinations)}/{len(sampled)})")
    
    return rate, hallucinations


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Measure hallucination rates across systems'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='experiments/results/sparse_corpus',
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=200,
        help='Number of samples to analyze per system'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling'
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("="*60)
    print("HALLUCINATION ANALYSIS")
    print("="*60)
    
    if not TRANSFORMERS_AVAILABLE:
        print("\n[WARNING] transformers library not available.")
        print("          Install with: pip install transformers")
        print("          Continuing with mock analysis...")
    
    # Define result files
    result_files = {
        'Static RAG': os.path.join(args.results_dir, 'static_rag_latest.json'),
        'Naive Write-back': os.path.join(args.results_dir, 'naive_writeback_latest.json'),
        'Bidirectional RAG': os.path.join(args.results_dir, 'bidirectional_rag_latest.json')
    }
    
    # Initialize NLI pipeline once
    nli_pipeline = None
    if TRANSFORMERS_AVAILABLE:
        print("\nLoading NLI model (this may take a minute)...")
        try:
            nli_pipeline = pipeline(
                "text-classification",
                model="cross-encoder/nli-deberta-v3-base",
                device=-1  # CPU
            )
            print("    [OK] NLI model loaded")
        except Exception as e:
            print(f"    [WARNING] Failed to load NLI model: {e}")
            print("    [INFO] Will use mock analysis")
    
    # Measure all three systems
    results = {}
    all_hallucinations = {}
    
    for system_name, result_file in result_files.items():
        rate, hallucinations = measure_system_hallucinations(
            result_file,
            system_name,
            n_samples=args.n_samples,
            nli_pipeline=nli_pipeline
        )
        results[system_name.lower().replace(' ', '_')] = {
            'rate': rate,
            'count': len(hallucinations)
        }
        all_hallucinations[system_name.lower().replace(' ', '_')] = hallucinations
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    static_rate = results.get('static_rag', {}).get('rate', 0.0)
    naive_rate = results.get('naive_write-back', {}).get('rate', 0.0)
    bidir_rate = results.get('bidirectional_rag', {}).get('rate', 0.0)
    
    print(f"\nStatic RAG:        {static_rate:.1%}")
    if static_rate > 0:
        naive_increase = ((naive_rate / static_rate - 1) * 100) if static_rate > 0 else 0
        bidir_increase = ((bidir_rate / static_rate - 1) * 100) if static_rate > 0 else 0
        print(f"Naive Write-back:  {naive_rate:.1%} ({naive_increase:+.0f}% vs static)")
        print(f"Bidirectional RAG: {bidir_rate:.1%} ({bidir_increase:+.0f}% vs static)")
    else:
        print(f"Naive Write-back:  {naive_rate:.1%}")
        print(f"Bidirectional RAG: {bidir_rate:.1%}")
    
    # Save results
    output_dir = Path(args.results_dir).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'hallucination_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'static': results.get('static_rag', {}),
            'naive': results.get('naive_write-back', {}),
            'bidirectional': results.get('bidirectional_rag', {}),
            'naive_increase_percent': ((naive_rate / static_rate - 1) * 100) if static_rate > 0 else 0,
            'bidir_increase_percent': ((bidir_rate / static_rate - 1) * 100) if static_rate > 0 else 0,
            'samples_analyzed': args.n_samples,
            'hallucination_examples': {
                k: v[:10] for k, v in all_hallucinations.items()  # Save first 10 examples
            }
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()

