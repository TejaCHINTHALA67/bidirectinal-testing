"""
Corpus Configurations for Bidirectional RAG Experiments
=====================================================

Implements the 3-level corpus realistic initialization strategy:
1. SPARSE: 2,024 Stack Overflow documents (Current Baseline)
2. MODERATE: 2,024 SO + 500 Wikipedia articles (Mixed Domain)
3. REALISTIC: 2,024 SO + 2,000 Wikipedia articles (Broad Domain)

Handles fetching Wikipedia articles relevant to NQ/TriviaQA/HotpotQA domains
to create realistic "partial knowledge" scenarios.
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional
import time

try:
    import wikipediaapi
    WIKI_AVAILABLE = True
except ImportError:
    WIKI_AVAILABLE = False

logger = logging.getLogger(__name__)

class CorpusConfigurator:
    """Manages creation and loading of different corpus configurations."""
    
    CONFIGS = {
        'sparse': {'so_count': 2024, 'wiki_count': 0},
        'moderate': {'so_count': 2024, 'wiki_count': 500},
        'realistic': {'so_count': 2024, 'wiki_count': 2000}
    }
    
    # Categories to sample from for broad domain coverage
    # Expanded list to ensure sufficient unique articles for realistic corpus (2000+)
    WIKI_CATEGORIES = [
        # Science & Technology
        "Category:History_of_science",
        "Category:Biology",
        "Category:Physics", 
        "Category:Chemistry",
        "Category:Computer_science",
        "Category:Astronomy",
        "Category:Mathematics",
        "Category:Engineering",
        "Category:Medicine",
        "Category:Genetics",
        # History
        "Category:20th-century_history",
        "Category:21st-century_history",
        "Category:Ancient_history",
        "Category:Medieval_history",
        "Category:World_War_II",
        "Category:World_War_I",
        "Category:American_Civil_War",
        "Category:French_Revolution",
        # Geography
        "Category:Geography_of_Europe",
        "Category:Geography_of_North_America",
        "Category:Geography_of_Asia",
        "Category:Geography_of_Africa",
        "Category:Countries",
        "Category:Capital_cities",
        "Category:Rivers",
        "Category:Mountains",
        # Culture & Arts
        "Category:Literature",
        "Category:Pop_culture",
        "Category:Music",
        "Category:Films",
        "Category:Television",
        "Category:Art",
        "Category:Architecture",
        # Sports
        "Category:Sports",
        "Category:Olympic_Games",
        "Category:Football",
        "Category:Baseball",
        # People
        "Category:Scientists",
        "Category:Politicians",
        "Category:Writers",
        "Category:Musicians",
        "Category:Athletes",
        # Additional domains for NQ/TriviaQA coverage
        "Category:Universities",
        "Category:Companies",
        "Category:Awards",
        "Category:Languages",
        "Category:Religions",
    ]

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.variants_dir = self.data_dir / "corpus_variants"
        self.variants_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Wikipedia API with proper user agent
        if WIKI_AVAILABLE:
            self.wiki = wikipediaapi.Wikipedia(
                user_agent='BidirectionalRAGResearch/1.0 (contact: research@example.com)',
                language='en'
            )
        else:
            logger.warning("wikipedia-api not installed. Wikipedia fetching will be mocked/failed.")
            self.wiki = None

    def get_corpus(self, config_name: str, seed: int = 42) -> List[Dict]:
        """
        Get the corpus for a specific configuration.
        
        Parameters
        ----------
        config_name : str
            One of 'sparse', 'moderate', 'realistic'
        seed : int
            Random seed for reproducibility
            
        Returns
        -------
        List[Dict]
            Combined list of documents
        """
        if config_name not in self.CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Options: {list(self.CONFIGS.keys())}")
            
        config = self.CONFIGS[config_name]
        logger.info(f"Loading corpus configuration: {config_name.upper()} "
                   f"(SO: {config['so_count']}, Wiki: {config['wiki_count']})")
        
        # Check if pre-built variant exists
        variant_path = self.variants_dir / f"corpus_{config_name}.json"
        if variant_path.exists():
            logger.info(f"Loading cached corpus from {variant_path}")
            with open(variant_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        # Build if not exists
        return self._build_corpus(config_name, config, variant_path, seed)

    def _build_corpus(self, name: str, config: Dict, save_path: Path, seed: int) -> List[Dict]:
        """Build and save a corpus configuration."""
        random.seed(seed)
        
        # 1. Load Base Stack Overflow Corpus
        so_corpus = self._load_so_corpus(config['so_count'])
        
        # 2. valid Wiki Corpus (if needed)
        wiki_corpus = []
        if config['wiki_count'] > 0:
            wiki_corpus = self._get_wiki_docs(config['wiki_count'], seed)
            
        # 3. Combine
        full_corpus = so_corpus + wiki_corpus
        random.shuffle(full_corpus)
        
        # 4. Save
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(full_corpus, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Built and saved {name} corpus with {len(full_corpus)} docs")
        return full_corpus

    def _load_so_corpus(self, count: int) -> List[Dict]:
        """Load base Stack Overflow documents."""
        # Try multiple paths for SO corpus
        paths = [
            self.data_dir / "processed" / "stackoverflow_corpus.json",
            self.data_dir / "sparse" / "stackoverflow_sparse_corpus.json",
            self.data_dir / "stackoverflow_corpus_train.json"
        ]
        
        for p in paths:
            if p.exists():
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        docs = json.load(f)
                        # Normalize format
                        normalized = []
                        for d in docs[:count]:
                            normalized.append({
                                'id': d.get('id', str(d.get('doc_id', 'unknown'))),
                                'text': d.get('text', d.get('content', '')),
                                'source': 'stackoverflow',
                                'topic': d.get('topic', 'programming')
                            })
                        return normalized
                except Exception as e:
                    logger.warning(f"Failed to load SO corpus from {p}: {e}")
                    
        raise FileNotFoundError("Could not find base Stack Overflow corpus! Run basic setup first.")

    def _get_wiki_docs(self, count: int, seed: int) -> List[Dict]:
        """Get Wikipedia documents, either from cache or fetch new."""
        wiki_cache_path = self.data_dir / "raw" / "wiki_pool.json"
        
        # Try loading existing pool
        pool = []
        if wiki_cache_path.exists():
            with open(wiki_cache_path, 'r', encoding='utf-8') as f:
                pool = json.load(f)
                
        # Fetch if needed
        if len(pool) < count:
            if not self.wiki:
                raise ImportError("wikipedia-api not available and not enough cached docs!")
            
            logger.info(f"Fetching {count - len(pool)} new Wikipedia articles...")
            new_docs = self._fetch_wiki_articles(count - len(pool) + 100, seed) # Fetch extra
            pool.extend(new_docs)
            
            # Save updated pool
            wiki_cache_path.parent.mkdir(exist_ok=True, parents=True)
            with open(wiki_cache_path, 'w', encoding='utf-8') as f:
                json.dump(pool, f, indent=2, ensure_ascii=False)
                
        # Sample (handle case where we still don't have enough after fetching)
        if len(pool) < count:
            logger.warning(f"Only have {len(pool)} Wikipedia articles, need {count}. Using all available.")
            selected = pool.copy()
        else:
            selected = random.sample(pool, count)
        return selected

    def _fetch_wiki_articles(self, count: int, seed: int, max_retries: int = 3) -> List[Dict]:
        """Fetch random articles from categories with retry logic and rate limiting.
        
        Uses exponential backoff for transient failures - essential for reproducible
        research experiments.
        """
        docs = []
        seen_titles = set()
        
        def _fetch_with_retry(fetch_func, *args, retries=max_retries, base_delay=1.0):
            """Execute fetch with exponential backoff retry."""
            for attempt in range(retries):
                try:
                    return fetch_func(*args)
                except Exception as e:
                    if attempt < retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Retry {attempt+1}/{retries} after {delay:.1f}s: {e}")
                        time.sleep(delay)
                    else:
                        raise
            return None
        
        # Simple BFS-like fetch from categories
        for category in self.WIKI_CATEGORIES:
            if len(docs) >= count:
                break
            
            try:
                cat_page = _fetch_with_retry(self.wiki.page, category)
                if cat_page is None or not cat_page.exists():
                    continue
            except Exception as e:
                logger.warning(f"Failed to fetch category {category}: {e}")
                continue
            
            try:
                members = list(cat_page.categorymembers.values())
            except Exception as e:
                logger.warning(f"Failed to get members of {category}: {e}")
                continue
                
            for member in members:
                if len(docs) >= count:
                    break
                    
                if member.ns == wikipediaapi.Namespace.MAIN and member.title not in seen_titles:
                    try:
                        # Rate limiting: add small delay between requests
                        time.sleep(0.2)  # 200ms between requests
                        
                        # Get summary + first section with retry
                        text = f"{member.summary}\n\n"
                        for section in member.sections:
                            text += f"{section.title}\n{section.text}\n"
                            if len(text) > 2000: # Limit length
                                break
                                
                        docs.append({
                            'id': f"wiki_{member.pageid}",
                            'text': text[:3000],
                            'source': 'wikipedia',
                            'title': member.title,
                            'topic': category.replace("Category:", "")
                        })
                        seen_titles.add(member.title)
                        
                        if len(docs) % 50 == 0:
                            logger.info(f"Fetched {len(docs)}/{count} articles...")
                            
                    except Exception as e:
                        logger.warning(f"Error fetching {member.title}: {e}")
                        time.sleep(1.0)  # Extra delay after errors
                        
        return docs
