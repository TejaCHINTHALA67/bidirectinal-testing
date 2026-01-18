"""
Verify Hybrid Systems
=====================

Smoke test for hybrid baselines (Self-RAG+WB, FLARE+WB, CRAG+WB).
Ensures classes instantiate and basic query API words.
"""

import sys
import os
import shutil
from pathlib import Path

# Add src
projects_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(projects_root, 'src'))
sys.path.insert(0, projects_root) # For other modules

from src.systems.baselines import get_system_class

def test_system(name, corpus_path):
    print(f"\nTesting {name}...")
    try:
        SysClass = get_system_class(name)
        # Mocking embedding model for speed if possible, but simpler to just use generic one
        system = SysClass(
            corpus_path=corpus_path,
            embedding_model='all-MiniLM-L6-v2',
            chroma_persist_dir=f'./temp_verify_{name}',
            debug=True
        )
        
        # Test basic retrieval
        print("  Retrieving...")
        docs, _, _, _ = system._retrieve("test query")
        print(f"  Got {len(docs)} docs")
        
        # We won't generate to save time/cost/ollama calls, just check class structure
        print(f"  [SUCCESS] Initialized {name}")
        
    except Exception as e:
        print(f"  [FAILURE] {name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            shutil.rmtree(f'./temp_verify_{name}')
        except:
            pass

def main():
    # Ensure dummy corpus exists
    corpus_path = "tests/mock_corpus.json"
    if not os.path.exists(corpus_path):
        import json
        with open(corpus_path, 'w') as f:
            json.dump([{"id": "1", "text": "This is a test document."}], f)
            
    systems = ['self_rag_wb', 'flare_wb', 'crag_wb']
    
    for s in systems:
        test_system(s, corpus_path)

if __name__ == '__main__':
    main()
