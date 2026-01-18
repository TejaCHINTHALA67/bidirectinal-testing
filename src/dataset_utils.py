"""
Dataset Utilities
==================

Utility functions for loading and manipulating datasets.
"""

import json
import os
from typing import Dict, List, Tuple, Optional
import random
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_stackoverflow_data(
    data_dir: str = 'data',
    use_sparse_corpus: bool = False
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load Stack Overflow dataset from JSON files.
    
    Parameters
    ----------
    data_dir : str
        Directory containing dataset files
    use_sparse_corpus : bool
        If True, load sparse corpus instead of full training corpus
        
    Returns
    -------
    Tuple[List[Dict], List[Dict], List[Dict]]
        Training documents, test documents, queries
    """
    if use_sparse_corpus:
        # Load sparse corpus
        sparse_path = os.path.join(data_dir, 'sparse', 'corpus.json')
        if not os.path.exists(sparse_path):
            raise FileNotFoundError(
                f"Sparse corpus not found at {sparse_path}. "
                f"Please run: python data/create_sparse_corpus.py"
            )
        train_path = sparse_path
    else:
        train_path = os.path.join(data_dir, 'stackoverflow_corpus_train.json')
    
    test_path = os.path.join(data_dir, 'stackoverflow_corpus_test.json')
    queries_path = os.path.join(data_dir, 'stackoverflow_queries.json')
    
    # Load files
    with open(train_path, 'r', encoding='utf-8') as f:
        train_docs = json.load(f)
    
    with open(test_path, 'r', encoding='utf-8') as f:
        test_docs = json.load(f)
    
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    return train_docs, test_docs, queries


def load_statistics(data_dir: str = 'data') -> Dict:
    """Load dataset statistics."""
    stats_path = os.path.join(data_dir, 'dataset_statistics.json')
    
    with open(stats_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    return stats


def create_evaluation_splits(
    queries: List[Dict],
    n_splits: int = 5,
    seed: int = 42
) -> List[Tuple[List[Dict], List[Dict]]]:
    """
    Create cross-validation splits for queries.
    
    Parameters
    ----------
    queries : List[Dict]
        List of query dictionaries
    n_splits : int
        Number of folds
    seed : int
        Random seed
        
    Returns
    -------
    List[Tuple[List[Dict], List[Dict]]]
        List of (train_queries, eval_queries) tuples
    """
    random.seed(seed)
    queries_shuffled = queries.copy()
    random.shuffle(queries_shuffled)
    
    fold_size = len(queries) // n_splits
    splits = []
    
    for i in range(n_splits):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < n_splits - 1 else len(queries)
        
        eval_queries = queries_shuffled[start_idx:end_idx]
        train_queries = queries_shuffled[:start_idx] + queries_shuffled[end_idx:]
        
        splits.append((train_queries, eval_queries))
    
    return splits


def sample_subset(
    documents: List[Dict],
    n: int,
    seed: int = 42
) -> List[Dict]:
    """
    Sample a random subset of documents.
    
    Parameters
    ----------
    documents : List[Dict]
        List of documents
    n : int
        Number of documents to sample
    seed : int
        Random seed
        
    Returns
    -------
    List[Dict]
        Sampled documents
    """
    random.seed(seed)
    
    if n >= len(documents):
        return documents.copy()
    
    return random.sample(documents, n)


def compute_dataset_statistics(documents: List[Dict]) -> Dict:
    """
    Compute statistics for a document collection.
    
    Parameters
    ----------
    documents : List[Dict]
        List of documents
        
    Returns
    -------
    Dict
        Statistics dictionary
    """
    # Length statistics
    lengths = [len(doc['content']) for doc in documents]
    
    # Topic distribution if available
    topics = {}
    for doc in documents:
        topic = doc.get('metadata', {}).get('topic', 'Unknown')
        topics[topic] = topics.get(topic, 0) + 1
    
    # Convert numpy types to native Python types
    stats = {
        'num_documents': int(len(documents)),
        'length_stats': {
            'mean': float(np.mean(lengths)),
            'median': float(np.median(lengths)),
            'std': float(np.std(lengths)),
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths))
        },
        'topic_distribution': topics
    }
    
    return stats


def format_documents_for_rag(documents: List[Dict]) -> List[Dict]:
    """
    Format documents for RAG system ingestion.
    
    Ensures all documents have required fields:
    - doc_id
    - content
    - metadata (optional)
    
    Parameters
    ----------
    documents : List[Dict]
        Raw documents
        
    Returns
    -------
    List[Dict]
        Formatted documents
    """
    formatted = []
    
    for doc in documents:
        formatted_doc = {
            'doc_id': doc.get('doc_id', f"doc_{len(formatted)}"),
            'content': doc.get('content', doc.get('text', '')),
            'metadata': doc.get('metadata', {})
        }
        
        # Ensure metadata has origin field
        if 'origin' not in formatted_doc['metadata']:
            formatted_doc['metadata']['origin'] = 'human_authored'
        
        formatted.append(formatted_doc)
    
    return formatted


def batch_queries(
    queries: List[Dict],
    batch_size: int
) -> List[List[Dict]]:
    """
    Batch queries for processing.
    
    Parameters
    ----------
    queries : List[Dict]
        List of queries
    batch_size : int
        Size of each batch
        
    Returns
    -------
    List[List[Dict]]
        Batched queries
    """
    batches = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        batches.append(batch)
    
    return batches


def save_results(
    results: Dict,
    output_path: str,
    pretty: bool = True
):
    """
    Save results to JSON file.
    
    Parameters
    ----------
    results : Dict
        Results dictionary
    output_path : str
        Output file path
    pretty : bool
        Whether to pretty-print JSON
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        else:
            json.dump(results, f, ensure_ascii=False, cls=NumpyEncoder)


def load_results(results_path: str) -> Dict:
    """Load results from JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results

