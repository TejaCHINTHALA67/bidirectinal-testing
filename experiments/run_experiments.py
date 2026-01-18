"""
Experiment Runner
=================

Runs comprehensive experiments comparing different RAG configurations:
1. Static RAG (baseline)
2. Naive Write-back (no acceptance layer)
3. Bidirectional RAG (full system)
4. Ablation studies
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import random
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

from tqdm import tqdm
import numpy as np

from src.bidirectional_rag import BidirectionalRAG
from src.dataset_utils import (
    load_stackoverflow_data,
    format_documents_for_rag,
    save_results
)


class ExperimentRunner:
    """
    Runs and tracks experiments for RAG systems.
    """
    
    def __init__(
        self,
        output_dir: str = 'experiments/results',
        checkpoint_interval: int = 250
    ):
        """
        Initialize experiment runner.
        
        Parameters
        ----------
        output_dir : str
            Directory to save results
        checkpoint_interval : int
            Interval for checkpointing metrics
        """
        self.output_dir = output_dir
        self.checkpoint_interval = checkpoint_interval
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*60)
        print("EXPERIMENT RUNNER INITIALIZED")
        print("="*60)
        print(f"Output directory: {output_dir}")
        print(f"Checkpoint interval: {checkpoint_interval}")
    
    def run_static_rag_baseline(
        self,
        train_docs: List[Dict],
        train_queries: List[Dict],
        test_queries: List[Dict],
        experiment_name: str = "static_rag"
    ) -> Dict:
        """
        Run Static RAG baseline (no write-back).
        
        This is the traditional RAG that never adds new documents.
        """
        print("\n" + "="*60)
        print(f"EXPERIMENT: {experiment_name.upper()}")
        print("="*60)
        print("Configuration: Traditional RAG (no write-back)")
        
        # Initialize system with REAL Ollama LLM
        print("\n[1/4] Initializing system with Ollama Llama3.2:3b...")
        rag = BidirectionalRAG(
            llm_provider='ollama',
            model_name='llama3.2:3b',
            chroma_persist_dir=f'./chroma_{experiment_name}',
            use_cpu=True
        )
        
        # Add initial corpus
        print(f"\n[2/4] Loading initial corpus ({len(train_docs)} documents)...")
        formatted_docs = format_documents_for_rag(train_docs)
        rag.add_documents(formatted_docs)
        
        # Use provided global split
        print(f"\n[2.5/4] Using global split: {len(train_queries)} training, {len(test_queries)} test")
        print(f"           Test IDs (first 5): {[q.get('query_id') for q in test_queries[:5]]}")
        
        # STEP 1: Measure INITIAL coverage on TEST set (before any processing)
        print(f"\n[3/4] Measuring initial coverage on test set ({len(test_queries)} queries)...")
        initial_coverage = self._measure_coverage_on_queries(rag, test_queries)
        print(f"    Initial coverage (test set): {initial_coverage:.2%}")
        
        # STEP 2: Process TRAINING queries (NO WRITE-BACK for static RAG)
        print(f"\n[3.5/4] Processing {len(train_queries)} training queries (static, no write-back)...")
        
        results = {
            'experiment': experiment_name,
            'config': {
                'write_back_enabled': False,
                'initial_corpus_size': len(train_docs),
                'train_queries': len(train_queries),
                'test_queries': len(test_queries)
            },
            'checkpoints': [],
            'queries': [],
            'initial_coverage': initial_coverage,
            'final_coverage': 0.0,
            'coverage_improvement': 0.0
        }
        
        for idx, query_data in enumerate(tqdm(train_queries, desc="Processing training queries")):
            query = query_data['query']
            
            # Get response (but don't write back)
            result = rag.query(query)
            
            # Track result
            query_result = {
                'query_id': query_data['query_id'],
                'query': query,
                'retrieved_docs': len(result.retrieved_docs),
                'response_length': len(result.response),
                'top_distance': result.top_distance
            }
            results['queries'].append(query_result)
            
            # Checkpoint metrics
            if (idx + 1) % self.checkpoint_interval == 0 or idx == len(train_queries) - 1:
                checkpoint = {
                    'query_index': idx + 1,
                    'corpus_size': len(train_docs),  # Static - no growth
                    'coverage': initial_coverage,  # Same as initial (no write-back)
                    'coverage_percentage': initial_coverage * 100
                }
                results['checkpoints'].append(checkpoint)
        
        # STEP 3: Measure FINAL coverage on TEST set (after processing training queries)
        print(f"\n[3.75/4] Measuring final coverage on test set...")
        final_coverage = self._measure_coverage_on_queries(rag, test_queries)
        coverage_improvement = final_coverage - initial_coverage
        
        results['final_coverage'] = final_coverage
        results['coverage_improvement'] = coverage_improvement
        
        print(f"    Final coverage (test set): {final_coverage:.2%}")
        print(f"    Coverage improvement: {coverage_improvement:+.2%}")
        
        # Add final checkpoint
        results['checkpoints'].append({
            'query_index': len(train_queries),
            'corpus_size': len(train_docs),
            'coverage': final_coverage,
            'coverage_percentage': final_coverage * 100
        })
        
        # Save results
        print(f"\n[4/4] Saving results...")
        self._save_experiment_results(results, experiment_name)
        
        print(f"\n[COMPLETE] {experiment_name}")
        return results
    
    def run_naive_writeback(
        self,
        train_docs: List[Dict],
        train_queries: List[Dict],
        test_queries: List[Dict],
        experiment_name: str = "naive_writeback"
    ) -> Dict:
        """
        Run Naive Write-back (no acceptance layer).
        
        Accepts ALL generated responses without validation.
        """
        print("\n" + "="*60)
        print(f"EXPERIMENT: {experiment_name.upper()}")
        print("="*60)
        print("Configuration: Naive write-back (NO safety checks)")
        
        # Initialize system with DISABLED acceptance layer
        print("\n[1/4] Initializing system with Ollama Llama3.2:3b (safety disabled)...")
        rag = BidirectionalRAG(
            grounding_threshold=0.0,  # Accept everything
            novelty_threshold=1.0,    # Accept everything
            llm_provider='ollama',
            model_name='llama3.2:3b',
            chroma_persist_dir=f'./chroma_{experiment_name}',
            use_cpu=True
        )
        
        # Add initial corpus
        print(f"\n[2/4] Loading initial corpus ({len(train_docs)} documents)...")
        formatted_docs = format_documents_for_rag(train_docs)
        rag.add_documents(formatted_docs)
        
        initial_size = len(train_docs)
        
        # Use provided global split
        print(f"\n[2.5/4] Using global split: {len(train_queries)} training, {len(test_queries)} test")
        print(f"           Test IDs (first 5): {[q.get('query_id') for q in test_queries[:5]]}")
        
        # STEP 1: Measure INITIAL coverage on TEST set
        print(f"\n[3/4] Measuring initial coverage on test set ({len(test_queries)} queries)...")
        initial_coverage = self._measure_coverage_on_queries(rag, test_queries)
        print(f"    Initial coverage (test set): {initial_coverage:.2%}")
        
        # STEP 2: Process TRAINING queries WITH NAIVE WRITE-BACK
        print(f"\n[3.5/4] Processing {len(train_queries)} training queries (naive write-back)...")
        
        results = {
            'experiment': experiment_name,
            'config': {
                'write_back_enabled': True,
                'acceptance_layer_enabled': False,
                'initial_corpus_size': initial_size,
                'train_queries': len(train_queries),
                'test_queries': len(test_queries)
            },
            'checkpoints': [],
            'queries': [],
            'acceptance_stats': {
                'total_accepted': 0,
                'total_rejected': 0
            },
            'initial_coverage': initial_coverage,
            'final_coverage': 0.0,
            'coverage_improvement': 0.0
        }
        
        for idx, query_data in enumerate(tqdm(train_queries, desc="Processing training queries")):
            query = query_data['query']
            
            # Get response WITH write-back
            result = rag.query(query)
            
            # Track result
            query_result = {
                'query_id': query_data['query_id'],
                'query': query,
                'retrieved_docs': len(result.retrieved_docs),
                'written_back': result.written_back,
                'response_length': len(result.response),
                'top_distance': result.top_distance
            }
            results['queries'].append(query_result)
            
            if result.written_back:
                results['acceptance_stats']['total_accepted'] += 1
            
            # Checkpoint metrics
            if (idx + 1) % self.checkpoint_interval == 0 or idx == len(train_queries) - 1:
                stats = rag.get_statistics()
                checkpoint = {
                    'query_index': idx + 1,
                    'corpus_size': stats['documents_added'],
                    'model_generated': stats['model_generated'],
                    'coverage': initial_coverage,  # Will update after final measurement
                    'coverage_percentage': initial_coverage * 100
                }
                results['checkpoints'].append(checkpoint)
        
        # STEP 3: Measure FINAL coverage on TEST set (after corpus growth)
        print(f"\n[3.75/4] Measuring final coverage on test set...")
        final_coverage = self._measure_coverage_on_queries(rag, test_queries)
        coverage_improvement = final_coverage - initial_coverage
        
        results['final_coverage'] = final_coverage
        results['coverage_improvement'] = coverage_improvement
        
        print(f"    Final coverage (test set): {final_coverage:.2%}")
        print(f"    Coverage improvement: {coverage_improvement:+.2%}")
        
        # Update final checkpoint
        if results['checkpoints']:
            results['checkpoints'][-1]['coverage'] = final_coverage
            results['checkpoints'][-1]['coverage_percentage'] = final_coverage * 100
        
        # Final stats
        stats = rag.get_statistics()
        results['final_stats'] = {
            'corpus_size': stats['documents_added'],
            'corpus_growth': stats['model_generated'],
            'growth_rate': stats['model_generated'] / initial_size if initial_size > 0 else 0
        }
        
        # Save results
        print(f"\n[4/4] Saving results...")
        self._save_experiment_results(results, experiment_name)
        
        print(f"\n[COMPLETE] {experiment_name}")
        return results
    
    def run_bidirectional_rag(
        self,
        train_docs: List[Dict],
        train_queries: List[Dict],
        test_queries: List[Dict],
        experiment_name: str = "bidirectional_rag"
    ) -> Dict:
        """
        Run full Bidirectional RAG with acceptance layer.
        
        This is our proposed system with all safety mechanisms.
        """
        print("\n" + "="*60)
        print(f"EXPERIMENT: {experiment_name.upper()}")
        print("="*60)
        print("Configuration: Full Bidirectional RAG (WITH safety checks)")
        
        # Initialize system with FULL acceptance layer
        # Note: Thresholds calibrated for CrossEncoder NLI model
        print("\n[1/4] Initializing system with Ollama Llama3.2:3b (full safety)...")
        rag = BidirectionalRAG(
            grounding_threshold=0.60,  # Calibrated for cross-encoder sigmoid scores
            novelty_threshold=0.90,
            composition_ratio_cap=0.30,
            llm_provider='ollama',
            model_name='llama3.2:3b',
            chroma_persist_dir=f'./chroma_{experiment_name}',
            use_cpu=True
        )
        
        # Add initial corpus
        print(f"\n[2/4] Loading initial corpus ({len(train_docs)} documents)...")
        formatted_docs = format_documents_for_rag(train_docs)
        rag.add_documents(formatted_docs)
        
        initial_size = len(train_docs)
        
        # Use provided global split
        print(f"\n[2.5/4] Using global split: {len(train_queries)} training, {len(test_queries)} test")
        print(f"           Test IDs (first 5): {[q.get('query_id') for q in test_queries[:5]]}")
        
        # STEP 1: Measure INITIAL coverage on TEST set
        print(f"\n[3/4] Measuring initial coverage on test set ({len(test_queries)} queries)...")
        initial_coverage = self._measure_coverage_on_queries(rag, test_queries)
        print(f"    Initial coverage (test set): {initial_coverage:.2%}")
        
        # STEP 2: Process TRAINING queries WITH FULL VALIDATION
        print(f"\n[3.5/4] Processing {len(train_queries)} training queries (bidirectional RAG)...")
        
        results = {
            'experiment': experiment_name,
            'config': {
                'write_back_enabled': True,
                'acceptance_layer_enabled': True,
                'grounding_threshold': 0.60,
                'novelty_threshold': 0.90,
                'composition_ratio_cap': 0.30,
                'initial_corpus_size': initial_size,
                'train_queries': len(train_queries),
                'test_queries': len(test_queries)
            },
            'checkpoints': [],
            'queries': [],
            'acceptance_stats': {
                'total_accepted': 0,
                'total_rejected': 0,
                'rejection_reasons': {
                    'grounding': 0,
                    'attribution': 0,
                    'novelty': 0
                }
            },
            'samples': {
                'accepted': [],
                'rejected': []
            },
            'initial_coverage': initial_coverage,
            'final_coverage': 0.0,
            'coverage_improvement': 0.0
        }
        
        for idx, query_data in enumerate(tqdm(train_queries, desc="Processing training queries")):
            query = query_data['query']
            
            # Get response WITH validation
            result = rag.query(query)
            
            # Track result
            query_result = {
                'query_id': query_data['query_id'],
                'query': query,
                'retrieved_docs': len(result.retrieved_docs),
                'written_back': result.written_back,
                'accepted': result.acceptance_result.accepted,
                'grounding_score': result.acceptance_result.grounding_score,
                'has_attribution': result.acceptance_result.has_attribution,
                'novelty_score': result.acceptance_result.novelty_score,
                'rejection_reason': result.acceptance_result.rejection_reason,
                'response_length': len(result.response),
                'top_distance': result.top_distance  # Store distance for coverage calculation
            }
            results['queries'].append(query_result)
            
            # Update acceptance stats
            if result.written_back:
                results['acceptance_stats']['total_accepted'] += 1
                # Save sample (first 50)
                if len(results['samples']['accepted']) < 50:
                    results['samples']['accepted'].append({
                        'query': query,
                        'response': result.response[:200],
                        'scores': {
                            'grounding': result.acceptance_result.grounding_score,
                            'novelty': result.acceptance_result.novelty_score
                        }
                    })
            else:
                results['acceptance_stats']['total_rejected'] += 1
                # Track rejection reason
                if result.acceptance_result.rejection_reason:
                    if 'grounding' in result.acceptance_result.rejection_reason.lower():
                        results['acceptance_stats']['rejection_reasons']['grounding'] += 1
                    elif 'attribution' in result.acceptance_result.rejection_reason.lower():
                        results['acceptance_stats']['rejection_reasons']['attribution'] += 1
                    elif 'novelty' in result.acceptance_result.rejection_reason.lower():
                        results['acceptance_stats']['rejection_reasons']['novelty'] += 1
                
                # Save sample (first 50)
                if len(results['samples']['rejected']) < 50:
                    results['samples']['rejected'].append({
                        'query': query,
                        'response': result.response[:200],
                        'reason': result.acceptance_result.rejection_reason
                    })
            
            # Checkpoint metrics
            if (idx + 1) % self.checkpoint_interval == 0 or idx == len(train_queries) - 1:
                stats = rag.get_statistics()
                checkpoint = {
                    'query_index': idx + 1,
                    'corpus_size': stats['documents_added'],
                    'model_generated': stats['model_generated'],
                    'acceptance_rate': stats.get('acceptance_rate', 0.0),
                    'coverage': initial_coverage,  # Will update after final measurement
                    'coverage_percentage': initial_coverage * 100
                }
                results['checkpoints'].append(checkpoint)
        
        # STEP 3: Measure FINAL coverage on TEST set (after corpus growth)
        print(f"\n[3.75/4] Measuring final coverage on test set...")
        final_coverage = self._measure_coverage_on_queries(rag, test_queries)
        coverage_improvement = final_coverage - initial_coverage
        
        results['final_coverage'] = final_coverage
        results['coverage_improvement'] = coverage_improvement
        
        print(f"    Final coverage (test set): {final_coverage:.2%}")
        print(f"    Coverage improvement: {coverage_improvement:+.2%}")
        
        # Update final checkpoint
        if results['checkpoints']:
            results['checkpoints'][-1]['coverage'] = final_coverage
            results['checkpoints'][-1]['coverage_percentage'] = final_coverage * 100
        
        # Final stats
        stats = rag.get_statistics()
        results['final_stats'] = {
            'corpus_size': stats['documents_added'],
            'corpus_growth': stats['model_generated'],
            'growth_rate': stats['model_generated'] / initial_size if initial_size > 0 else 0,
            'acceptance_rate': stats.get('acceptance_rate', 0.0),
            'rejection_rate': stats.get('rejection_rate', 0.0)
        }
        
        # Save results
        print(f"\n[4/4] Saving results...")
        self._save_experiment_results(results, experiment_name)
        
        print(f"\n[COMPLETE] {experiment_name}")
        return results
    
    def run_ablation_study(
        self,
        train_docs: List[Dict],
        queries: List[Dict],
        num_queries_per_ablation: int = 500
    ) -> Dict:
        """
        Run ablation study: test impact of each safety mechanism.
        """
        print("\n" + "="*60)
        print("ABLATION STUDY")
        print("="*60)
        
        # Limit queries for ablation (faster)
        ablation_queries = queries[:num_queries_per_ablation]
        
        configurations = {
            'no_grounding': {
                'description': 'No grounding check',
                'grounding_threshold': 0.60,
                'novelty_threshold': 0.90,
                'use_grounding': False,
                'use_attribution': True,
                'use_novelty': True
            },
            'no_attribution': {
                'description': 'No attribution check',
                'grounding_threshold': 0.60,
                'novelty_threshold': 0.90,
                'use_grounding': True,
                'use_attribution': False,
                'use_novelty': True
            },
            'no_novelty': {
                'description': 'No novelty check',
                'grounding_threshold': 0.60,
                'novelty_threshold': 1.0,
                'use_grounding': True,
                'use_attribution': True,
                'use_novelty': False
            }
        }
        
        ablation_results = {
            'experiment': 'ablation_study',
            'num_queries': len(ablation_queries),
            'configurations': {}
        }
        
        for config_name, config in configurations.items():
            print(f"\n[Ablation] Testing: {config['description']}")
            
            # Initialize with ablated configuration using REAL LLM
            rag = BidirectionalRAG(
                grounding_threshold=config.get('grounding_threshold', 0.85),
                novelty_threshold=config.get('novelty_threshold', 0.90),
                llm_provider='ollama',
                model_name='llama3.2:3b',
                chroma_persist_dir=f'./chroma_ablation_{config_name}',
                use_cpu=True,
                use_grounding=config.get('use_grounding', True),
                use_attribution=config.get('use_attribution', True),
                use_novelty=config.get('use_novelty', True)
            )
            
            # Add documents
            formatted_docs = format_documents_for_rag(train_docs)
            rag.add_documents(formatted_docs)
            
            accepted = 0
            rejected = 0
            query_records = []
            variant_gate_settings = {
                'use_grounding': config.get('use_grounding', True),
                'use_attribution': config.get('use_attribution', True),
                'use_novelty': config.get('use_novelty', True)
            }
            gate_thresholds = {
                'grounding_threshold': config.get('grounding_threshold', 0.85),
                'novelty_threshold': config.get('novelty_threshold', 0.90)
            }

            for query_data in tqdm(ablation_queries, desc=f"  {config_name}"):
                result = rag.query(query_data['query'])
                if result.written_back:
                    accepted += 1
                else:
                    rejected += 1

                acceptance = result.acceptance_result
                query_records.append({
                    'variant': config_name,
                    'query_id': query_data.get('query_id'),
                    'query': query_data['query'],
                    'response': result.response,
                    'accepted': result.written_back,
                    'grounding_score': acceptance.grounding_score,
                    'has_attribution': acceptance.has_attribution,
                    'novelty_score': acceptance.novelty_score,
                    'rejection_reason': acceptance.rejection_reason,
                    'retrieved_ids': result.retrieved_ids,
                    'timestamp': result.timestamp,
                    'gate_settings': variant_gate_settings.copy(),
                    'gate_thresholds': gate_thresholds.copy()
                })

            # Get stats
            stats = rag.get_statistics()
            
            total_queries = len(ablation_queries)
            acceptance_rate = accepted / total_queries if total_queries else 0.0

            ablation_results['configurations'][config_name] = {
                'description': config['description'],
                'config': config,
                'summary': {
                    'accepted': accepted,
                    'rejected': rejected,
                    'acceptance_rate': acceptance_rate,
                    'corpus_growth': stats['model_generated'],
                    'final_corpus_size': stats['documents_added'],
                    'statistics': stats
                },
                'queries': query_records
            }
            
            print(f"    Accepted: {accepted}/{len(ablation_queries)} ({accepted/len(ablation_queries):.2%})")
        
        # Save ablation results
        self._save_experiment_results(ablation_results, 'ablation_study')
        
        print(f"\n[COMPLETE] Ablation study")
        return ablation_results
    
    def _measure_coverage_on_queries(
        self,
        rag: BidirectionalRAG,
        queries: List[Dict],
        relevance_threshold: float = 0.4
    ) -> float:
        """
        Measure coverage on a set of queries by retrieving (without processing).
        
        This is used for test set evaluation - we just check if queries can
        retrieve relevant documents, not generate responses.
        
        Parameters
        ----------
        rag : BidirectionalRAG
            RAG system to test
        queries : List[Dict]
            List of query dictionaries
        relevance_threshold : float
            Maximum distance for relevance (default 0.4)
            
        Returns
        -------
        float
            Coverage percentage (0.0 to 1.0)
        """
        if not queries:
            return 0.0
        
        covered = 0
        for query_data in queries:
            query_text = query_data.get('query', '')
            if not query_text:
                continue
            
            # Just retrieve, don't generate
            retrieved_docs, retrieved_ids, retrieved_meta, top_distance = rag.retrieve(query_text, k=5)
            
            # Check if relevant document found
            if retrieved_docs and top_distance < relevance_threshold:
                covered += 1
        
        return covered / len(queries) if queries else 0.0
    
    def _split_queries_train_test(
        self,
        queries: List[Dict],
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split queries into training and test sets.
        
        Parameters
        ----------
        queries : List[Dict]
            All queries
        train_ratio : float
            Proportion for training (default 0.8 = 80%)
        seed : int
            Random seed for reproducibility
            
        Returns
        -------
        Tuple[List[Dict], List[Dict]]
            (train_queries, test_queries)
        """
        queries_copy = queries.copy()
        random.seed(seed)
        random.shuffle(queries_copy)
        
        split_idx = int(len(queries_copy) * train_ratio)
        train_queries = queries_copy[:split_idx]
        test_queries = queries_copy[split_idx:]
        
        return train_queries, test_queries
    
    def _compute_coverage(
        self, 
        query_results: List[Dict],
        relevance_threshold: float = 0.4
    ) -> float:
        """
        Compute coverage: % of queries that retrieved relevant documents.
        
        Uses relevance threshold: only count as covered if top document
        has distance < threshold (lower distance = more relevant).
        
        Distance interpretation:
        - Distance < 0.3: Very relevant (same topic, similar question)
        - Distance 0.3-0.5: Somewhat relevant (same domain)
        - Distance > 0.5: Poor match (different topic entirely)
        
        When corpus has gaps (e.g., no JavaScript docs):
        - JavaScript query will retrieve Python/SQL docs
        - Top distance will be ~0.6-0.8 (not relevant)
        - Coverage = FAIL for that query
        
        Parameters
        ----------
        query_results : List[Dict]
            List of query result dictionaries
        relevance_threshold : float
            Maximum distance for a document to be considered relevant
            (default 0.4 - adjust based on your embedding model)
            
        Returns
        -------
        float
            Coverage percentage (0.0 to 1.0)
        """
        if not query_results:
            return 0.0
        
        # Count queries with relevant documents
        covered = 0
        for q in query_results:
            retrieved_docs = q.get('retrieved_docs', 0)
            if retrieved_docs == 0:
                # No documents retrieved = not covered
                continue
            
            # Get distance to top document
            top_distance = q.get('top_distance', 1.0)  # Default to max distance if not available
            
            # Check if document is relevant enough
            if top_distance < relevance_threshold:
                # Document is relevant (low distance = high similarity)
                covered += 1
            # else: document not relevant enough, don't count as covered
        
        return covered / len(query_results)
    
    def _is_experiment_complete(self, experiment_name: str, expected_queries: int) -> bool:
        """Check if an experiment is already complete."""
        latest_filepath = os.path.join(self.output_dir, f"{experiment_name}_latest.json")
        
        if not os.path.exists(latest_filepath):
            return False
        
        try:
            with open(latest_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if experiment processed expected number of queries
            queries_processed = len(data.get('queries', []))
            has_final_coverage = 'final_coverage' in data and data.get('final_coverage', 0) > 0
            
            # Experiment is complete if it processed all expected queries and has final coverage
            is_complete = queries_processed >= expected_queries and has_final_coverage
            
            if is_complete:
                print(f"    [SKIP] {experiment_name} already complete ({queries_processed} queries processed)")
            
            return is_complete
        except Exception as e:
            print(f"    [WARN] Could not check {experiment_name} completion: {e}")
            return False
    
    def _save_experiment_results(self, results: Dict, experiment_name: str):
        """Save experiment results to JSON."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        save_results(results, filepath)
        
        # Also save as "latest"
        latest_filepath = os.path.join(self.output_dir, f"{experiment_name}_latest.json")
        save_results(results, latest_filepath)
        
        print(f"    [OK] Saved: {filepath}")


def main():
    """Run all experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run bidirectional RAG experiments')
    parser.add_argument(
        '--sparse-corpus',
        action='store_true',
        help='Use sparse corpus instead of full corpus (for coverage experiments)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--queries',
        type=int,
        default=None,
        help='Limit number of queries to process (for testing). If not specified, uses all queries.'
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("BIDIRECTIONAL RAG - EXPERIMENTAL EVALUATION")
    print("="*60)
    
    if args.sparse_corpus:
        print("\n[INFO] Using SPARSE CORPUS (for coverage experiments)")
        print("       This corpus has intentional knowledge gaps (~60% coverage)")
    
    # Load dataset
    print("\n[Step 1] Loading dataset...")
    try:
        train_docs, test_docs, queries = load_stackoverflow_data(
            use_sparse_corpus=args.sparse_corpus
        )
        
        # Note: We use MIXED queries (not filtered) to get realistic ~50% initial coverage
        # Mixed = 50% Python/SQL (covered) + 50% JavaScript/Git (not covered)
        # This allows us to demonstrate coverage improvement from ~50% → ~75%
        if args.sparse_corpus:
            print(f"    [INFO] Using MIXED queries (Python/SQL + JavaScript/Git)")
            print(f"          Expected: ~50% initial coverage (Python/SQL covered, JS/Git not)")
        
        # Limit queries if specified (for testing)
        if args.queries and args.queries < len(queries):
            queries = queries[:args.queries]
            print(f"    [INFO] Limited to {args.queries} queries for testing")
        
        corpus_type = "sparse" if args.sparse_corpus else "full"
        print(f"    [OK] Loaded {len(train_docs)} train docs ({corpus_type}), {len(queries)} queries")
    except FileNotFoundError as e:
        print(f"    [ERROR] Dataset not found: {e}")
        if args.sparse_corpus:
            print("    Please run: python data/create_sparse_corpus.py")
        else:
            print("    Please run: python data/prepare_stackoverflow.py")
        return
    
    # Initialize experiment runner
    output_dir = args.output_dir
    if args.sparse_corpus:
        # Save sparse corpus results in separate directory
        output_dir = os.path.join(args.output_dir, 'sparse_corpus')
    
    runner = ExperimentRunner(
        output_dir=output_dir,
        checkpoint_interval=250
    )
    
    # Run experiments
    print("\n[Step 2] Running experiments...")

    # Global train/test split (SPLIT ONCE for all systems)
    import random
    random.seed(42)
    shuffled = queries.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.8)
    train_queries = shuffled[:split_idx]
    test_queries = shuffled[split_idx:]
    print(f"\n[Split] Global split: {len(train_queries)} train, {len(test_queries)} test")
    print(f"[Split] Test IDs (first 5): {[q.get('query_id') for q in test_queries[:5]]}")
    
    # Check which experiments are already complete
    print("\n[Step 2.5] Checking for completed experiments...")
    expected_train_queries = len(train_queries)
    
    static_complete = runner._is_experiment_complete('static_rag', expected_train_queries)
    naive_complete = runner._is_experiment_complete('naive_writeback', expected_train_queries)
    bidirectional_complete = runner._is_experiment_complete('bidirectional_rag', expected_train_queries)
    
    # Experiment 1: Static RAG (baseline)
    print("\n" + "-"*60)
    if static_complete:
        print("[SKIP] Static RAG - already complete, loading results...")
        latest_filepath = os.path.join(output_dir, 'static_rag_latest.json')
        with open(latest_filepath, 'r', encoding='utf-8') as f:
            static_results = json.load(f)
        print(f"    [OK] Loaded existing results: {len(static_results.get('queries', []))} queries")
    else:
        static_results = runner.run_static_rag_baseline(train_docs, train_queries, test_queries)
    
    # Experiment 2: Naive Write-back
    print("\n" + "-"*60)
    if naive_complete:
        print("[SKIP] Naive Write-back - already complete, loading results...")
        latest_filepath = os.path.join(output_dir, 'naive_writeback_latest.json')
        with open(latest_filepath, 'r', encoding='utf-8') as f:
            naive_results = json.load(f)
        print(f"    [OK] Loaded existing results: {len(naive_results.get('queries', []))} queries")
    else:
        naive_results = runner.run_naive_writeback(train_docs, train_queries, test_queries)
    
    # Experiment 3: Bidirectional RAG (proposed)
    print("\n" + "-"*60)
    if bidirectional_complete:
        print("[SKIP] Bidirectional RAG - already complete, loading results...")
        latest_filepath = os.path.join(output_dir, 'bidirectional_rag_latest.json')
        with open(latest_filepath, 'r', encoding='utf-8') as f:
            bidirectional_results = json.load(f)
        print(f"    [OK] Loaded existing results: {len(bidirectional_results.get('queries', []))} queries")
    else:
        bidirectional_results = runner.run_bidirectional_rag(train_docs, train_queries, test_queries)
    
    # Experiment 4: Ablation Study
    print("\n" + "-"*60)
    ablation_results = runner.run_ablation_study(train_docs, queries, num_queries_per_ablation=500)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENTAL EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: experiments/results/")
    print(f"\nNext steps:")
    print(f"  1. Run analysis: python experiments/statistical_analysis.py")
    print(f"  2. Generate figures: python experiments/generate_figures.py")


if __name__ == '__main__':
    main()

