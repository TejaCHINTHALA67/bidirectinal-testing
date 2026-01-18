"""
Corpus Configuration Manager
============================

Implements the multi-level corpus initialization strategy:
1. SPARSE: Stack Overflow only (already exists)
2. MODERATE: Stack Overflow + 500 Wikipedia articles
3. REALISTIC: Stack Overflow + 2000 Wikipedia articles

Handles downloading Wikipedia articles and mixing them with the base corpus.

Usage:
    python scripts/corpus_configurations.py --config realistic
"""

import argparse
import json
import random
import sys
import os
from pathlib import Path
from typing import List, Dict

try:
    import wikipediaapi
    WIKI_AVAILABLE = True
except ImportError:
    WIKI_AVAILABLE = False
    print("[WARNING] wikipedia-api not installed. Run: pip install wikipedia-api")

def download_wikipedia_articles(num_articles: int, topics: List[str], seed: int = 42) -> List[Dict]:
    """Download Wikipedia articles relevant to topics."""
    if not WIKI_AVAILABLE:
        print("[ERROR] Cannot download articles: wikipedia-api missing.")
        return []
    
    print(f"[WIKI] Downloading {num_articles} articles for topics: {topics}")
    
    # Initialize API with user agent as per policy
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent='BidirectionalRagResearch/1.0 (research@example.com)',
        language='en'
    )
    
    articles = []
    seen_titles = set()
    
    # Deterministic shuffling of topics
    random.seed(seed)
    
    # BFS-style crawling strategy to get diverse pages
    # Start with broad topics
    queue = topics.copy()
    random.shuffle(queue)
    
    while len(articles) < num_articles and queue:
        page_title = queue.pop(0)
        if page_title in seen_titles:
            continue
            
        page = wiki_wiki.page(page_title)
        seen_titles.add(page_title)
        
        if not page.exists():
            continue
            
        # Add valid article
        if len(page.text) > 500: # Filter stubs
            articles.append({
                'id': f"wiki_{page.pageid}",
                'title': page.title,
                'text': page.text[:2000], # Trucate for reasonable index size
                'url': page.fullurl,
                'origin': 'wikipedia',
                'topic': 'general_knowledge' # Simplification
            })
            print(f"  + {page.title}")
            
        # Add links to queue to explore graph
        links = list(page.links.keys())
        random.shuffle(links)
        queue.extend(links[:5]) # Expand breadth
        
    return articles[:num_articles]

def mix_corpora(base_corpus_path: str, wiki_articles: List[Dict], output_path: str):
    """Mix base corpus with Wikipedia articles."""
    with open(base_corpus_path, 'r', encoding='utf-8') as f:
        base_corpus = json.load(f)
        
    print(f"[MIX] Base: {len(base_corpus)} docs, Wiki: {len(wiki_articles)} docs")
    
    mixed_corpus = base_corpus + wiki_articles
    
    # Shuffle
    random.seed(42)
    random.shuffle(mixed_corpus)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mixed_corpus, f, indent=2, ensure_ascii=False)
        
    print(f"[SAVE] Mixed corpus saved to {output_path} ({len(mixed_corpus)} docs)")

def main():
    parser = argparse.ArgumentParser(description='Create corpus configurations')
    parser.add_argument('--config', type=str, choices=['sparse', 'moderate', 'realistic'], required=True)
    parser.add_argument('--base-corpus', type=str, default='data/processed/stackoverflow_corpus.json')
    parser.add_argument('--output-dir', type=str, default='data/processed')
    args = parser.parse_args()
    
    # Ensure base corpus exists
    if not os.path.exists(args.base_corpus):
        print(f"[ERROR] Base corpus not found: {args.base_corpus}")
        sys.exit(1)
        
    if args.config == 'sparse':
        print("[INFO] Sparse config uses base corpus directly.")
        return
        
    # Define Wiki requirements
    if args.config == 'moderate':
        num_wiki = 500
    else: # realistic
        num_wiki = 2000
        
    topics = ['Python (programming language)', 'Algorithm', 'Data science', 'Artificial intelligence', 'Machine learning', 'History of science', 'Geography']
    
    # Check if we have cached wiki data to avoid re-downloading
    wiki_cache_path = Path('data/raw/wiki_cache.json')
    wiki_articles = []
    
    if wiki_cache_path.exists():
        print("[INFO] Loading Wikipedia articles from cache...")
        with open(wiki_cache_path, 'r', encoding='utf-8') as f:
            all_wiki = json.load(f)
            wiki_articles = all_wiki[:num_wiki]
    else:
        # Try download
        if WIKI_AVAILABLE:
            wiki_articles = download_wikipedia_articles(num_wiki, topics)
            # Cache it
            if wiki_articles:
                wiki_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(wiki_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(wiki_articles, f, indent=2)
        else:
            print("[WARNING] Using mock Wikipedia data (library missing).")
            # Mock data for reproducibility/offline mode
            for i in range(num_wiki):
                wiki_articles.append({
                    'id': f"wiki_mock_{i}",
                    'title': f"Mock Article {i}",
                    'text': f"This is a mock Wikipedia article {i} about general knowledge used for testing the realistic corpus configuration.",
                    'origin': 'wikipedia',
                    'topic': 'mock'
                })
    
    output_path = Path(args.output_dir) / f"corpus_{args.config}.json"
    mix_corpora(args.base_corpus, wiki_articles, str(output_path))

if __name__ == '__main__':
    main()
