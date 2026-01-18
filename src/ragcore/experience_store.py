"""
Experience Store for Bidirectional RAG 2.0
===========================================

Stores structured experience logs (critiques) instead of raw model outputs
to prevent model collapse and enable "learning from mistakes."
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ExperienceStore:
    """
    Separate ChromaDB collection for storing experience logs (critiques).
    
    Unlike the main corpus which stores facts, this stores meta-cognitive
    information about past attempts, failures, and successes.
    """
    
    def __init__(
        self,
        chroma_persist_dir: str = './chroma_experience_store',
        embedding_model: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize the Experience Store.
        
        Parameters
        ----------
        chroma_persist_dir : str
            Directory for ChromaDB persistence
        embedding_model : str
            Sentence-transformers model for embeddings
        """
        self.chroma_persist_dir = chroma_persist_dir
        self.embedding_model_name = embedding_model
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        Path(chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=chroma_persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="experience_logs",
            metadata={"description": "Experience logs for Bidirectional RAG 2.0"}
        )
        
        logger.info(f"Experience Store initialized at {chroma_persist_dir}")
    
    def add_experience(
        self,
        query: str,
        response: str,
        grounding_score: float,
        attribution_check: bool,
        critique_text: str,
        verdict: str,
        retrieved_ids: Optional[List[str]] = None
    ) -> str:
        """
        Add an experience log to the store.
        
        Parameters
        ----------
        query : str
            Original user query
        response : str
            Generated response (for summary)
        grounding_score : float
            Grounding verification score
        attribution_check : bool
            Whether attribution check passed
        critique_text : str
            Critique explaining why it passed/failed
        verdict : str
            Verdict label: "VERIFIED_SYNTHESIS", "WARNING_HALLUCINATION", or "PARTIAL_SUCCESS"
        retrieved_ids : List[str], optional
            IDs of retrieved documents used
        
        Returns
        -------
        str
            ID of the added experience log
        """
        exp_id = f"exp_{uuid.uuid4().hex[:12]}"
        
        # Create summary content (what the model tried to do)
        summary = f"Summary: User asked about '{query[:100]}...'. Model attempted to answer."
        
        # Create full content for embedding
        content = f"{summary}\n\nCritique: {critique_text}"
        
        # Generate embedding
        embedding = self.embedding_model.encode(
            content,
            convert_to_tensor=False
        ).tolist()
        
        # Create metadata
        metadata = {
            'type': 'experience_log',
            'original_query': query,
            'response_summary': response[:200] if response else '',  # First 200 chars
            'grounding_score': str(grounding_score),
            'attribution_check': str(attribution_check),
            'critique_text': critique_text,
            'verdict': verdict,
            'timestamp': datetime.now().isoformat(),
            'retrieved_ids': ','.join(retrieved_ids) if retrieved_ids else ''
        }
        
        # Add to collection
        self.collection.add(
            ids=[exp_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )
        
        logger.info(f"[EXPERIENCE STORE] Added experience log: {exp_id} (verdict: {verdict})")
        return exp_id
    
    def retrieve_experiences(
        self,
        query: str,
        n_results: int = 3
    ) -> List[Dict]:
        """
        Retrieve relevant experience logs for a query.
        
        Parameters
        ----------
        query : str
            User query
        n_results : int
            Number of experience logs to retrieve
        
        Returns
        -------
        List[Dict]
            List of experience log dictionaries with:
            - content: The experience log content
            - metadata: Full metadata including verdict, critique, etc.
            - distance: Similarity distance
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=False
        ).tolist()
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['ids'] or not results['ids'][0]:
            return []
        
        # Format results
        experiences = []
        for i, exp_id in enumerate(results['ids'][0]):
            experiences.append({
                'id': exp_id,
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if results['distances'] else 1.0
            })
        
        logger.debug(f"Retrieved {len(experiences)} experience logs for query")
        return experiences
    
    def get_statistics(self) -> Dict:
        """Get statistics about the experience store."""
        count = self.collection.count()
        
        # Count by verdict type
        all_metadatas = self.collection.get(include=['metadatas'])['metadatas']
        verdict_counts = {}
        for meta in all_metadatas:
            verdict = meta.get('verdict', 'UNKNOWN')
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        return {
            'total_experiences': count,
            'verdict_counts': verdict_counts
        }

    def add_rejection(
        self,
        query: str,
        response: str,
        reason: str,
        retrieved_ids: Optional[List[str]] = None
    ) -> str:
        """
        Syntactic sugar for adding a rejection experience.
        
        Parameters
        ----------
        query : str
            User query
        response : str
            The rejected response
        reason : str
            Reason for rejection
        retrieved_ids : List[str], optional
            IDs of docs used
        
        Returns
        -------
        str
            ID of the added experience log
        """
        return self.add_experience(
            query=query,
            response=response,
            grounding_score=0.0,
            attribution_check=False,
            critique_text=f"REJECTED: {reason}",
            verdict="WARNING_HALLUCINATION",
            retrieved_ids=retrieved_ids
        )

    def find_similar_failures(
        self,
        query: str,
        threshold: float = 0.8
    ) -> List[Dict]:
        """
        Find if this query has failed similarly in the past.
        Used for calculating Failure Repetition Rate.
        
        Parameters
        ----------
        query : str
            Current query
        threshold : float
            Similarity threshold (0-1)
            
        Returns
        -------
        List[Dict]
            List of similar past failures
        """
        # Get query embedding
        query_embedding = self.embedding_model.encode(
            query, 
            convert_to_tensor=False
        ).tolist()
        
        # Query specifically for failures
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where={"verdict": "WARNING_HALLUCINATION"},
            include=['metadatas', 'distances']
        )
        
        if not results['ids'] or not results['ids'][0]:
            return []
            
        failures = []
        for i, dist in enumerate(results['distances'][0]):
            # Convert cosine distance to similarity (approx)
            similarity = 1 - dist
            if similarity >= threshold:
                failures.append(results['metadatas'][0][i])
                
        return failures

