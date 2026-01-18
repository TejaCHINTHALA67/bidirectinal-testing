"""
Hybrid Baselines with Naive Writeback
=====================================

Implements SOTA baselines (Self-RAG, FLARE, CRAG) augmented with 
Naive Writeback memory. This tests whether simply adding memory to 
advanced retrieval systems is sufficient, or if Bidirectional RAG's 
acceptance layer is necessary.
"""

import uuid
import time
from datetime import datetime
from typing import Dict
import logging

# Import base classes
from src.systems.baselines import SelfRAGAdapter, FLARERAG, CRAGRAG, _lock

logger = logging.getLogger(__name__)

class WritebackMixin:
    """Mixin to add naive writeback capability."""
    
    def _perform_writeback(self, question: str, response: str):
        """Write response back to corpus blindly."""
        try:
            with _lock:
                doc_id = f"{self.system_slug}_doc_{uuid.uuid4()}"
                embedding = self.embedding_model.encode(response, convert_to_tensor=False)
                
                self.collection.add(
                    ids=[doc_id],
                    documents=[response],
                    embeddings=[embedding.tolist()],
                    metadatas=[{
                        'origin': 'model_generated',
                        'topic': 'unknown',
                        'timestamp': datetime.now().isoformat(),
                        'source_query': question,
                        'system': self.system_name
                    }]
                )
                
                self.stats['documents_added'] += 1
                return True
        except Exception as e:
            logger.error(f"[{self.system_name}] Writeback failed: {e}")
            return False

class SelfRAGWriteback(SelfRAGAdapter, WritebackMixin):
    """Self-RAG + Naive Writeback."""
    
    def query(self, question: str) -> Dict:
        result = super().query(question)
        
        # Write back
        start_wb = time.perf_counter()
        written = self._perform_writeback(question, result['response'])
        
        result['written_back'] = written
        # Adjust latency (optional, but realistic)
        result['latency_ms'] += (time.perf_counter() - start_wb) * 1000
        
        return result

class FLAREWriteback(FLARERAG, WritebackMixin):
    """FLARE + Naive Writeback."""
    
    def query(self, question: str) -> Dict:
        result = super().query(question)
        
        # Write back
        start_wb = time.perf_counter()
        written = self._perform_writeback(question, result['response'])
        
        result['written_back'] = written
        result['latency_ms'] += (time.perf_counter() - start_wb) * 1000
        
        return result

class CRAGWriteback(CRAGRAG, WritebackMixin):
    """CRAG + Naive Writeback."""
    
    def query(self, question: str) -> Dict:
        result = super().query(question)
        
        # Write back
        start_wb = time.perf_counter()
        written = self._perform_writeback(question, result['response'])
        
        result['written_back'] = written
        result['latency_ms'] += (time.perf_counter() - start_wb) * 1000
        
        return result
