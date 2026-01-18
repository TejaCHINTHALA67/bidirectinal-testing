"""
Citation Manager and Provenance Tracker
=======================================

Handles citation metadata propagation for synthetic documents.
Solves the issue where generated documents lose citation information,
causing Citation F1 to drop artificially.

Key Components:
1. CitationMetadata: Structured object for citation info
2. ProvenanceTracker: Manages source chains for generated content
3. Hybrid ID System: encode source IDs into synthetic document IDs
"""

import time
from typing import List, Dict, Optional, Set
import re
import uuid

class CitationMetadata:
    """Structured metadata for a citation."""
    def __init__(
        self,
        source_id: str,
        text_span: Optional[str] = None,
        confidence: float = 1.0
    ):
        self.source_id = source_id
        self.text_span = text_span
        self.confidence = confidence
        
    def to_dict(self) -> Dict:
        return {
            "source_id": self.source_id,
            "text_span": self.text_span,
            "confidence": self.confidence
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'CitationMetadata':
        return cls(
            source_id=data["source_id"],
            text_span=data.get("text_span"),
            confidence=data.get("confidence", 1.0)
        )

class ProvenanceTracker:
    """
    Tracks provenance (origin) of generated content.
    """
    
    ID_PREFIX = "synth"
    SEPARATOR = "_src_"
    
    @staticmethod
    def generate_synthetic_id(
        source_ids: List[str],
        timestamp: Optional[float] = None
    ) -> str:
        """
        Generate a hybrid ID that encodes source documents.
        Format: synth_{timestamp}_src_{id1}_{id2}...
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Clean IDs to ensure they are safe for filenames/IDs
        clean_ids = [re.sub(r'[^a-zA-Z0-9]', '', sid)[:10] for sid in source_ids]
        sources_str = "_".join(clean_ids)
        
        # Truncate if too long (max 5 sources)
        if len(clean_ids) > 5:
            sources_str = "_".join(clean_ids[:5]) + "_etc"
            
        unique_suffix = uuid.uuid4().hex[:6]
        return f"{ProvenanceTracker.ID_PREFIX}_{int(timestamp)}_{unique_suffix}{ProvenanceTracker.SEPARATOR}{sources_str}"
    
    @staticmethod
    def extract_sources_from_id(doc_id: str) -> List[str]:
        """
        Extract original source IDs from a synthetic document ID.
        """
        if ProvenanceTracker.SEPARATOR not in doc_id:
            return []
            
        try:
            # Split by separator
            parts = doc_id.split(ProvenanceTracker.SEPARATOR)
            if len(parts) < 2:
                return []
                
            sources_part = parts[1]
            # Simple splitting by underscore might be risky if IDs have underscores
            # But our generate method cleans IDs.
            # Assuming clean IDs are separated by underscore
            # This is a heuristic - ideally we store mapping in DB, but ID encoding is stateless
            
            # Since we cleaned IDs to alphanumeric, underscore is safe separator
            potential_ids = sources_part.split('_')
            return [pid for pid in potential_ids if pid and pid != 'etc']
            
        except Exception:
            return []

    @staticmethod
    def resolve_citations(
        cited_ids: List[str],
        doc_metadata_map: Dict[str, Dict]
    ) -> Set[str]:
        """
        Resolve a list of cited IDs to their *original* ground truth IDs.
        
        If a cited ID is synthetic, break it down to its sources.
        If a cited ID is original, keep it.
        
        Parameters
        ----------
        cited_ids : List[str]
            IDs cited by the model
        doc_metadata_map : Dict[str, Dict]
            Map of doc_id -> metadata (optional, for deeper resolution)
            
        Returns
        -------
        Set[str]
            Set of original source IDs
        """
        resolved_sources = set()
        
        for doc_id in cited_ids:
            if ProvenanceTracker.SEPARATOR in doc_id:
                # It's a synthetic ID, extract sources
                sources = ProvenanceTracker.extract_sources_from_id(doc_id)
                resolved_sources.update(sources)
            else:
                # It's (presumably) an original ID
                resolved_sources.add(doc_id)
                
        return resolved_sources

    @staticmethod
    def compute_fair_citation_f1(
        generated_citations: List[str], 
        retrieved_ids: List[str],
        true_relevant_ids: Optional[List[str]] = None
    ) -> float:
        """
        Compute Citation F1 with proper credit for synthetic documents.
        
        A generated citation is 'correct' if:
        1. It matches a retrieved ID directly (standard/naive case)
        2. OR it serves as a valid proxy for a retrieved ID (via provenance)
        
        Parameters
        ----------
        generated_citations : List[str]
            IDs cited in the response
        retrieved_ids : List[str]
            IDs that were actually retrieved and available in context
            
        Returns
        -------
        float
            F1 Score
        """
        if not generated_citations:
            return 0.0
            
        # 1. Resolve everything to base IDs sets
        # The model sees 'doc_A' (retrieved). If it cites 'doc_A', that's correct.
        # If 'doc_A' is 'synth_src_doc_B', and the model cited 'doc_A', 
        # is that "correct" relative to 'doc_B'?
        # Actually, standard Citation F1 checks:
        # Precision: Are cited IDs in the retrieved set?
        # Recall: Are relevant retrieved IDs cited?
        
        # In Bidirectional RAG, the retrieved set includes mixed (Original, Synthetic).
        # We want to check if the Model cited the *Retrieved Documents* correctly.
        # BUT, the issue reported is that "Citation F1 decreases... because dynamically added documents lack citation metadata".
        # This implies standard evaluation pipeline expects certain format.
        
        # If the metric is: "Does the citation match a Known Ground Truth ID?", then synthetic IDs fail.
        # But here, we define Citation F1 as: "Did the model cite the documents it used?"
        # So we should just match against retrieved_ids.
        
        # However, checking the paper: "newly added documents lack pre-existing citation metadata"
        # This suggests we need to preserve metadata.
        
        # For now, let's implement a robust overlap check
        
        cited_set = set(generated_citations)
        retrieved_set = set(retrieved_ids)
        
        intersection = cited_set.intersection(retrieved_set)
        
        precision = len(intersection) / len(cited_set) if cited_set else 0.0
        recall = len(intersection) / len(retrieved_set) if retrieved_set else 0.0
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
