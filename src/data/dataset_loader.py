"""
Dataset Loader for IEEE-Grade RAG Experiments

Loads and processes 4 standard benchmarks:
1. Natural Questions (NQ)
2. TriviaQA
3. HotpotQA
4. Stack Overflow (existing)

Creates sparse versions with intentional knowledge gaps for coverage evaluation.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
    
    # Windows workaround: Patch HuggingFace file system glob to handle ** patterns
    import sys
    if sys.platform == 'win32':
        try:
            import huggingface_hub.hf_file_system
            original_glob = huggingface_hub.hf_file_system.HfFileSystem.glob
            
            def patched_glob(self, path, **kwargs):
                # Fix ** patterns that aren't at path boundaries
                if isinstance(path, str) and '**' in path:
                    import re
                    # Split path and fix each component
                    parts = path.split('/')
                    fixed_parts = []
                    for part in parts:
                        if '**' in part and part != '**':
                            # Replace ** with * in middle of component
                            part = part.replace('**', '*')
                        fixed_parts.append(part)
                    path = '/'.join(fixed_parts)
                return original_glob(self, path, **kwargs)
            
            huggingface_hub.hf_file_system.HfFileSystem.glob = patched_glob
            
            # Also patch fsspec.utils.glob_translate as backup
            try:
                import fsspec.utils
                original_glob_translate = fsspec.utils.glob_translate
                
                def patched_glob_translate(pattern):
                    if '**' in pattern:
                        parts = pattern.split('/')
                        fixed_parts = []
                        for part in parts:
                            if '**' in part and part != '**':
                                part = part.replace('**', '*')
                            fixed_parts.append(part)
                        pattern = '/'.join(fixed_parts)
                    return original_glob_translate(pattern)
                
                fsspec.utils.glob_translate = patched_glob_translate
            except Exception:
                pass
        except Exception as e:
            # If patching fails, continue without it
            pass
except ImportError:
    print("[WARNING] datasets library not available. Install with: pip install datasets")
    DATASETS_AVAILABLE = False


class DatasetLoader:
    """
    Load and process datasets for RAG experiments.
    
    Creates sparse corpora with intentional knowledge gaps to evaluate
    coverage improvement through corpus growth.
    """
    
    def __init__(self, cache_dir: str = "data/raw"):
        """
        Initialize dataset loader.
        
        Parameters
        ----------
        cache_dir : str
            Directory to cache downloaded datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.splits_dir = Path("data/splits")
        self.splits_dir.mkdir(parents=True, exist_ok=True)
    
    def load_and_process(
        self,
        dataset_name: str,
        num_docs: int = 2024,
        num_queries: int = 1000,
        sparse_ratio: float = 0.5,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load and process a dataset.
        
        Parameters
        ----------
        dataset_name : str
            Name of dataset: 'nq', 'triviaqa', 'hotpotqa', 'stackoverflow'
        num_docs : int
            Target number of documents in sparse corpus
        num_queries : int
            Target number of queries
        sparse_ratio : float
            Ratio of topics to keep (0.5 = keep 50%, remove 50%)
        seed : int
            Random seed for reproducibility
        
        Returns
        -------
        tuple
            (sparse_corpus: List[Dict], queries: List[Dict])
            Each doc: {"id": str, "text": str, "topic": str}
            Each query: {"question": str, "topic": str, "answer": str}
        """
        random.seed(seed)
        
        if dataset_name == 'nq':
            return self._load_natural_questions(num_docs, num_queries, sparse_ratio, seed)
        elif dataset_name == 'triviaqa':
            return self._load_triviaqa(num_docs, num_queries, seed)
        elif dataset_name == 'hotpotqa':
            return self._load_hotpotqa(num_docs, num_queries, seed)
        elif dataset_name == 'stackoverflow':
            return self._load_stackoverflow(num_docs, num_queries, sparse_ratio, seed)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _load_natural_questions(
        self,
        num_docs: int,
        num_queries: int,
        sparse_ratio: float,
        seed: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load Natural Questions dataset.
        
        Creates sparse corpus by keeping only 50% of topics.
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        print(f"[NQ] Loading Natural Questions dataset...")
        
        # Check for local pre-processed files first
        local_corpus = self.cache_dir.parent / "processed" / "natural_questions_corpus.json"
        local_queries = self.cache_dir.parent / "processed" / "natural_questions_queries.json"
        
        if local_corpus.exists() and local_queries.exists():
            print("[NQ] Loading from local pre-processed files...")
            with open(local_corpus, 'r', encoding='utf-8') as f:
                corpus = json.load(f)
            with open(local_queries, 'r', encoding='utf-8') as f:
                queries = json.load(f)
            print(f"[NQ] Loaded {len(corpus)} docs, {len(queries)} queries from local files")
            return corpus[:num_docs], queries[:num_queries]
        
        # Try to load from HuggingFace (Windows glob pattern issue should be patched)
        try:
            print("[NQ] Using streaming mode...")
            # Note: trust_remote_code not needed for Parquet-based datasets
            dataset_stream = load_dataset(
                'natural_questions',
                split='train',
                cache_dir=str(self.cache_dir),
                streaming=True
            )
            # Convert streaming dataset to list (take first 4000)
            dataset_list = []
            print("[NQ] Streaming data (this may take a few minutes)...")
            for i, item in enumerate(dataset_stream):
                if i >= 4000:
                    break
                if (i + 1) % 500 == 0:
                    print(f"[NQ] Loaded {i + 1}/4000 items...")
                dataset_list.append(item)
            # Convert to DatasetDict format
            from datasets import Dataset
            dataset = Dataset.from_list(dataset_list)
            print(f"[NQ] Successfully loaded {len(dataset_list)} items")
        except Exception as e:
            print(f"[ERROR] Failed to load Natural Questions: {e}")
            print("[INFO] Manual download required. See MANUAL_DATASET_DOWNLOAD.md")
            print(f"[INFO] Expected files:")
            print(f"  - {local_corpus}")
            print(f"  - {local_queries}")
            raise
        
        # Extract documents and questions
        all_docs = []
        all_queries = []
        
        for item in dataset:
            # Extract document
            if 'document' in item and 'text' in item['document']:
                doc_text = item['document']['text']
                if doc_text and len(doc_text) > 50:  # Filter very short docs
                    all_docs.append({
                        'id': f"nq_doc_{item.get('id', len(all_docs))}",
                        'text': doc_text[:2000],  # Limit length
                        'topic': self._extract_topic_nq(doc_text)
                    })
            
            # Extract question
            if 'question' in item and 'text' in item['question']:
                question_text = item['question']['text']
                if question_text:
                    all_queries.append({
                        'question': question_text,
                        'topic': self._extract_topic_nq(question_text),
                        'answer': item.get('annotations', [{}])[0].get('short_answers', [{}])[0].get('text', '') if item.get('annotations') else ''
                    })
        
        print(f"[NQ] Loaded {len(all_docs)} documents, {len(all_queries)} queries")
        
        # Create sparse corpus (keep 50% of topics)
        sparse_corpus, queries = self._create_sparse_corpus(
            all_docs, all_queries, num_docs, num_queries, sparse_ratio, seed
        )
        
        return sparse_corpus, queries
    
    def _load_triviaqa(
        self,
        num_docs: int,
        num_queries: int,
        seed: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load TriviaQA dataset.
        
        TriviaQA already has coverage variation, so we just sample.
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        print(f"[TriviaQA] Loading TriviaQA dataset...")
        
        # Check for local pre-processed files first
        local_corpus = self.cache_dir.parent / "processed" / "triviaqa_corpus.json"
        local_queries = self.cache_dir.parent / "processed" / "triviaqa_queries.json"
        
        if local_corpus.exists() and local_queries.exists():
            print("[TriviaQA] Loading from local pre-processed files...")
            with open(local_corpus, 'r', encoding='utf-8') as f:
                corpus = json.load(f)
            with open(local_queries, 'r', encoding='utf-8') as f:
                queries = json.load(f)
            print(f"[TriviaQA] Loaded {len(corpus)} docs, {len(queries)} queries from local files")
            return corpus[:num_docs], queries[:num_queries]
        
        # Try to load from HuggingFace (Windows glob pattern issue should be patched)
        try:
            print("[TriviaQA] Using streaming mode...")
            # Note: trust_remote_code not needed for Parquet-based datasets
            dataset_stream = load_dataset(
                'trivia_qa',
                'rc.nocontext',
                split='train',
                cache_dir=str(self.cache_dir),
                streaming=True
            )
            # Convert streaming dataset to list (sample first 5000 for processing)
            dataset_list = []
            print("[TriviaQA] Streaming data (this may take a few minutes)...")
            for i, item in enumerate(dataset_stream):
                if i >= 5000:  # Enough for sampling
                    break
                if (i + 1) % 1000 == 0:
                    print(f"[TriviaQA] Loaded {i + 1}/5000 items...")
                dataset_list.append(item)
            from datasets import Dataset
            dataset = Dataset.from_list(dataset_list)
            print(f"[TriviaQA] Successfully loaded {len(dataset_list)} items")
        except Exception as e:
            print(f"[ERROR] Failed to load TriviaQA: {e}")
            print("[INFO] Manual download required. See MANUAL_DATASET_DOWNLOAD.md")
            print(f"[INFO] Expected files:")
            print(f"  - {local_corpus}")
            print(f"  - {local_queries}")
            raise
        
        # Sample documents and queries
        all_items = list(dataset)
        random.seed(seed)
        random.shuffle(all_items)
        
        sampled = all_items[:num_docs + num_queries]
        
        # Split into docs and queries
        docs = []
        queries = []
        
        for i, item in enumerate(sampled):
            if i < num_docs:
                # Create document from question + answer
                doc_text = f"Question: {item.get('question', '')}\nAnswer: {item.get('answer', {}).get('value', '')}"
                docs.append({
                    'id': f"triviaqa_doc_{i}",
                    'text': doc_text[:2000],
                    'topic': self._extract_topic_triviaqa(item.get('question', ''))
                })
            else:
                queries.append({
                    'question': item.get('question', ''),
                    'topic': self._extract_topic_triviaqa(item.get('question', '')),
                    'answer': item.get('answer', {}).get('value', '')
                })
        
        print(f"[TriviaQA] Created {len(docs)} documents, {len(queries)} queries")
        
        return docs, queries[:num_queries]
    
    def _load_hotpotqa(
        self,
        num_docs: int,
        num_queries: int,
        seed: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load HotpotQA dataset.
        
        Filters for single-hop questions for fair comparison.
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        print(f"[HotpotQA] Loading HotpotQA dataset...")
        
        # Check for local pre-processed files first
        local_corpus = self.cache_dir.parent / "processed" / "hotpotqa_corpus.json"
        local_queries = self.cache_dir.parent / "processed" / "hotpotqa_queries.json"
        
        if local_corpus.exists() and local_queries.exists():
            print("[HotpotQA] Loading from local pre-processed files...")
            with open(local_corpus, 'r', encoding='utf-8') as f:
                corpus = json.load(f)
            with open(local_queries, 'r', encoding='utf-8') as f:
                queries = json.load(f)
            print(f"[HotpotQA] Loaded {len(corpus)} docs, {len(queries)} queries from local files")
            return corpus[:num_docs], queries[:num_queries]
        
        # Try to load from HuggingFace (Windows glob pattern issue should be patched)
        try:
            print("[HotpotQA] Using streaming mode...")
            # Note: trust_remote_code not needed for Parquet-based datasets
            dataset_stream = load_dataset(
                'hotpot_qa',
                'distractor',
                split='train',
                cache_dir=str(self.cache_dir),
                streaming=True
            )
            # Convert streaming dataset to list (sample first 5000 for processing)
            dataset_list = []
            print("[HotpotQA] Streaming data (this may take a few minutes)...")
            for i, item in enumerate(dataset_stream):
                if i >= 5000:  # Enough for sampling
                    break
                if (i + 1) % 1000 == 0:
                    print(f"[HotpotQA] Loaded {i + 1}/5000 items...")
                dataset_list.append(item)
            from datasets import Dataset
            dataset = Dataset.from_list(dataset_list)
            print(f"[HotpotQA] Successfully loaded {len(dataset_list)} items")
        except Exception as e:
            print(f"[ERROR] Failed to load HotpotQA: {e}")
            print("[INFO] Manual download required. See MANUAL_DATASET_DOWNLOAD.md")
            print(f"[INFO] Expected files:")
            print(f"  - {local_corpus}")
            print(f"  - {local_queries}")
            raise
        
        # Filter single-hop questions (level == 'easy' or has only 1 supporting fact)
        single_hop = []
        for item in dataset:
            # Check if single-hop (simplified: has 'level' or check supporting facts)
            if item.get('level') == 'easy' or len(item.get('supporting_facts', [])) <= 1:
                single_hop.append(item)
        
        print(f"[HotpotQA] Found {len(single_hop)} single-hop questions")
        
        # Sample
        random.seed(seed)
        random.shuffle(single_hop)
        
        sampled = single_hop[:num_docs + num_queries]
        
        # Create documents from contexts
        docs = []
        queries = []
        
        for i, item in enumerate(sampled):
            if i < num_docs:
                # Combine contexts into document
                contexts = item.get('context', [])
                doc_text = "\n\n".join([
                    f"{ctx.get('title', '')}\n{ctx.get('sentences', [''])[0]}"
                    for ctx in contexts[:3]  # Limit to 3 contexts
                ])
                
                docs.append({
                    'id': f"hotpotqa_doc_{i}",
                    'text': doc_text[:2000],
                    'topic': self._extract_topic_hotpotqa(item.get('question', ''))
                })
            else:
                queries.append({
                    'question': item.get('question', ''),
                    'topic': self._extract_topic_hotpotqa(item.get('question', '')),
                    'answer': item.get('answer', '')
                })
        
        print(f"[HotpotQA] Created {len(docs)} documents, {len(queries)} queries")
        
        return docs, queries[:num_queries]
    
    def _load_stackoverflow(
        self,
        num_docs: int,
        num_queries: int,
        sparse_ratio: float,
        seed: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load Stack Overflow dataset (existing implementation).
        """
        print(f"[StackOverflow] Loading Stack Overflow dataset...")
        
        # Check if existing sparse corpus exists (try multiple locations)
        possible_corpus_paths = [
            Path("data/sparse/stackoverflow_sparse_corpus.json"),
            Path("data/processed/stackoverflow_corpus.json"),
            Path("data/stackoverflow_corpus_train.json")
        ]
        possible_query_paths = [
            Path("data/stackoverflow_queries.json"),
            Path("data/processed/stackoverflow_queries.json")
        ]
        
        sparse_corpus_path = None
        queries_path = None
        
        for path in possible_corpus_paths:
            if path.exists():
                sparse_corpus_path = path
                break
        
        for path in possible_query_paths:
            if path.exists():
                queries_path = path
                break
        
        if sparse_corpus_path and sparse_corpus_path.exists() and queries_path and queries_path.exists():
            print(f"[StackOverflow] Loading from existing files...")
            with open(sparse_corpus_path, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
            
            with open(queries_path, 'r', encoding='utf-8') as f:
                queries_data = json.load(f)
            
            # Convert to our format
            corpus = [
                {
                    'id': doc.get('id', f"so_doc_{i}"),
                    'text': doc.get('content', doc.get('text', '')),
                    'topic': doc.get('topic', doc.get('tags', ['unknown'])[0] if doc.get('tags') else 'unknown')
                }
                for i, doc in enumerate(corpus_data[:num_docs])
            ]
            
            queries = [
                {
                    'question': q.get('query', q.get('question', '')),
                    'topic': q.get('topic', 'unknown'),
                    'answer': q.get('answer', '')
                }
                for q in queries_data[:num_queries]
            ]
            
            return corpus, queries
        
        # Otherwise, use existing create_sparse_corpus.py
        print(f"[StackOverflow] Creating sparse corpus...")
        import subprocess
        result = subprocess.run(
            ['python', 'data/create_sparse_corpus.py'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"[WARNING] Failed to create sparse corpus: {result.stderr}")
            print("[INFO] Using fallback: loading from existing files if available")
        
        # Try loading again
        if sparse_corpus_path.exists():
            return self._load_stackoverflow(num_docs, num_queries, sparse_ratio, seed)
        else:
            raise FileNotFoundError(
                "Stack Overflow sparse corpus not found. "
                "Run: python data/create_sparse_corpus.py"
            )
    
    def _create_sparse_corpus(
        self,
        all_docs: List[Dict],
        all_queries: List[Dict],
        num_docs: int,
        num_queries: int,
        sparse_ratio: float,
        seed: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Create sparse corpus by keeping only sparse_ratio of topics.
        
        Similar to Stack Overflow experiment: keep some topics, remove others.
        """
        random.seed(seed)
        
        # Group by topic
        docs_by_topic = {}
        queries_by_topic = {}
        
        for doc in all_docs:
            topic = doc.get('topic', 'unknown')
            if topic not in docs_by_topic:
                docs_by_topic[topic] = []
            docs_by_topic[topic].append(doc)
        
        for query in all_queries:
            topic = query.get('topic', 'unknown')
            if topic not in queries_by_topic:
                queries_by_topic[topic] = []
            queries_by_topic[topic].append(query)
        
        # Select topics to keep (sparse_ratio of all topics)
        all_topics = list(docs_by_topic.keys())
        random.shuffle(all_topics)
        num_topics_to_keep = max(1, int(len(all_topics) * sparse_ratio))
        kept_topics = set(all_topics[:num_topics_to_keep])
        removed_topics = set(all_topics[num_topics_to_keep:])
        
        print(f"[Sparse] Keeping {len(kept_topics)} topics, removing {len(removed_topics)} topics")
        
        # Build sparse corpus (only kept topics)
        sparse_corpus = []
        for topic in kept_topics:
            sparse_corpus.extend(docs_by_topic.get(topic, []))
        
        # Sample to target size
        random.shuffle(sparse_corpus)
        sparse_corpus = sparse_corpus[:num_docs]
        
        # Create queries: 50% from kept topics, 50% from removed topics
        kept_queries = []
        gap_queries = []
        
        for topic in kept_topics:
            kept_queries.extend(queries_by_topic.get(topic, []))
        
        for topic in removed_topics:
            gap_queries.extend(queries_by_topic.get(topic, []))
        
        random.shuffle(kept_queries)
        random.shuffle(gap_queries)
        
        # Mix: 50% kept, 50% gap
        num_kept = num_queries // 2
        num_gap = num_queries - num_kept
        
        queries = kept_queries[:num_kept] + gap_queries[:num_gap]
        random.shuffle(queries)
        
        print(f"[Sparse] Corpus: {len(sparse_corpus)} docs, Queries: {len(queries)} ({num_kept} kept, {num_gap} gap)")
        
        return sparse_corpus, queries
    
    def _extract_topic_nq(self, text: str) -> str:
        """Extract topic from Natural Questions text."""
        # Simple keyword-based topic extraction
        text_lower = text.lower()
        if any(word in text_lower for word in ['python', 'code', 'programming', 'function']):
            return 'programming'
        elif any(word in text_lower for word in ['who', 'person', 'born', 'died']):
            return 'people'
        elif any(word in text_lower for word in ['where', 'location', 'city', 'country']):
            return 'geography'
        elif any(word in text_lower for word in ['when', 'year', 'date', 'time']):
            return 'history'
        else:
            return 'general'
    
    def _extract_topic_triviaqa(self, text: str) -> str:
        """Extract topic from TriviaQA question."""
        return self._extract_topic_nq(text)  # Reuse same logic
    
    def _extract_topic_hotpotqa(self, text: str) -> str:
        """Extract topic from HotpotQA question."""
        return self._extract_topic_nq(text)  # Reuse same logic
    
    def save_to_disk(
        self,
        corpus: List[Dict],
        queries: List[Dict],
        dataset_name: str
    ):
        """
        Save processed corpus and queries to disk.
        
        Parameters
        ----------
        corpus : List[Dict]
            Processed corpus
        queries : List[Dict]
            Processed queries
        dataset_name : str
            Name of dataset
        """
        # Save corpus
        corpus_path = self.processed_dir / f"{dataset_name}_corpus.json"
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] Corpus saved to: {corpus_path} ({len(corpus)} docs)")
        
        # Save queries
        queries_path = self.processed_dir / f"{dataset_name}_queries.json"
        with open(queries_path, 'w', encoding='utf-8') as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] Queries saved to: {queries_path} ({len(queries)} queries)")
        
        # Create deterministic train/test split
        self._create_splits(queries, dataset_name, seed=42)
    
    def _create_splits(
        self,
        queries: List[Dict],
        dataset_name: str,
        seed: int = 42
    ):
        """
        Create deterministic train/test splits.
        
        Parameters
        ----------
        queries : List[Dict]
            All queries
        dataset_name : str
            Name of dataset
        seed : int
            Random seed for reproducibility
        """
        random.seed(seed)
        shuffled = queries.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * 0.8)
        train_queries = shuffled[:split_idx]
        test_queries = shuffled[split_idx:]
        
        # Save splits
        train_path = self.splits_dir / f"{dataset_name}_train.json"
        test_path = self.splits_dir / f"{dataset_name}_test.json"
        
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_queries, f, indent=2, ensure_ascii=False)
        
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_queries, f, indent=2, ensure_ascii=False)
        
        print(f"[SPLIT] Train: {len(train_queries)}, Test: {len(test_queries)}")
        print(f"[SAVE] Splits saved to: {train_path}, {test_path}")


def main():
    """Main execution: download and process all datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and process datasets')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['nq', 'triviaqa', 'hotpotqa', 'stackoverflow', 'all'],
        default='all',
        help='Dataset to process'
    )
    parser.add_argument(
        '--download_all',
        action='store_true',
        help='Download all datasets'
    )
    parser.add_argument(
        '--num_docs',
        type=int,
        default=2024,
        help='Number of documents in sparse corpus'
    )
    parser.add_argument(
        '--num_queries',
        type=int,
        default=1000,
        help='Number of queries'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    loader = DatasetLoader(cache_dir='data/raw')
    
    datasets_to_process = ['nq', 'triviaqa', 'hotpotqa', 'stackoverflow'] if args.dataset == 'all' else [args.dataset]
    
    print("="*70)
    print("DATASET PREPARATION")
    print("="*70)
    print(f"Datasets: {', '.join(datasets_to_process)}")
    print(f"Target docs: {args.num_docs}, Target queries: {args.num_queries}")
    print(f"Seed: {args.seed}")
    print("="*70)
    print()
    
    for dataset_name in datasets_to_process:
        print(f"\n{'='*70}")
        print(f"Processing: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        try:
            corpus, queries = loader.load_and_process(
                dataset_name=dataset_name,
                num_docs=args.num_docs,
                num_queries=args.num_queries,
                sparse_ratio=0.5,
                seed=args.seed
            )
            
            loader.save_to_disk(corpus, queries, dataset_name)
            
            print(f"[OK] {dataset_name.upper()} processed successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("DATASET PREPARATION COMPLETE")
    print(f"{'='*70}")
    print("\nNext step: Implement baseline systems (PROMPT 1)")


if __name__ == '__main__':
    main()

