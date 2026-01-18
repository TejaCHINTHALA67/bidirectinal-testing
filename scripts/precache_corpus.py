#!/usr/bin/env python3
"""
Pre-cache Corpus for IEEE Access Experiments
=============================================

This script pre-builds and caches the corpus variants BEFORE running
parallel experiments. This ensures:
1. All experiments use the exact same corpus (reproducibility)
2. No Wikipedia API rate limiting issues during experiments
3. Faster experiment execution

Run this ONCE before running the main experiments.
"""

import sys
import logging
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.corpus_configurations import CorpusConfigurator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def precache_all_corpora():
    """Pre-cache all corpus variants for reproducible experiments."""
    
    configurator = CorpusConfigurator(data_dir='data')
    
    configs_to_cache = ['sparse', 'moderate', 'realistic']
    
    for config_name in configs_to_cache:
        logger.info(f"\n{'='*60}")
        logger.info(f"Pre-caching corpus: {config_name.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            corpus = configurator.get_corpus(config_name, seed=42)
            logger.info(f"✓ {config_name}: {len(corpus)} documents cached")
        except Exception as e:
            logger.error(f"✗ Failed to cache {config_name}: {e}")
            return False
    
    logger.info(f"\n{'='*60}")
    logger.info("All corpora cached successfully!")
    logger.info("You can now run experiments with --corpus_type realistic")
    logger.info(f"{'='*60}")
    
    return True


if __name__ == "__main__":
    success = precache_all_corpora()
    sys.exit(0 if success else 1)
