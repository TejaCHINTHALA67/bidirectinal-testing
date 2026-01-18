"""
Critique Generator for Bidirectional RAG 2.0
==============================================

Generates structured critiques from acceptance results, creating
meta-cognitive objects that explain why responses passed or failed.
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Critique:
    """Structured critique object."""
    verdict: str  # "VERIFIED_SYNTHESIS", "WARNING_HALLUCINATION", or "PARTIAL_SUCCESS"
    critique_text: str
    grounding_score: float
    attribution_check: bool
    novelty_score: float
    rejection_reason: Optional[str] = None


class CritiqueGenerator:
    """
    Generates structured critiques from acceptance layer results.
    
    Transforms AcceptanceResult into Critique objects that explain
    why responses were accepted or rejected, enabling the system to
    learn from past attempts.
    """
    
    def __init__(self):
        """Initialize the critique generator."""
        pass
    
    def generate_critique(
        self,
        query: str,
        response: str,
        acceptance_result,
        retrieved_ids: list
    ) -> Critique:
        """
        Generate a critique from acceptance result.
        
        Parameters
        ----------
        query : str
            Original user query
        response : str
            Generated response
        acceptance_result
            AcceptanceResult object from acceptance layer
        retrieved_ids : list
            IDs of retrieved documents
        
        Returns
        -------
        Critique
            Structured critique object
        """
        grounding_score = acceptance_result.grounding_score
        has_attribution = acceptance_result.has_attribution
        novelty_score = acceptance_result.novelty_score
        accepted = acceptance_result.accepted
        rejection_reason = acceptance_result.rejection_reason
        
        # Determine verdict
        if accepted:
            if grounding_score >= 0.9:
                verdict = "VERIFIED_SYNTHESIS"
                critique_text = (
                    f"SUCCESS: Successfully synthesized answer from retrieved documents. "
                    f"Grounding score: {grounding_score:.2f}. "
                    f"Attribution: {'Present' if has_attribution else 'Missing'}. "
                    f"Novelty: {novelty_score:.2f}."
                )
            else:
                verdict = "PARTIAL_SUCCESS"
                critique_text = (
                    f"PARTIAL: Answer accepted but with moderate grounding ({grounding_score:.2f}). "
                    f"Attribution: {'Present' if has_attribution else 'Missing'}. "
                    f"Use with caution."
                )
        else:
            verdict = "WARNING_HALLUCINATION"
            critique_text = (
                f"FAILED: Attempted to answer '{query[:50]}...' but validation failed. "
                f"Grounding score: {grounding_score:.2f} (threshold: 0.85). "
                f"Attribution: {'Present' if has_attribution else 'Missing'}. "
                f"Reason: {rejection_reason or 'Unknown'}. "
                f"Do not repeat this approach."
            )
        
        return Critique(
            verdict=verdict,
            critique_text=critique_text,
            grounding_score=grounding_score,
            attribution_check=has_attribution,
            novelty_score=novelty_score,
            rejection_reason=rejection_reason
        )
    
    def format_experiences_for_prompt(
        self,
        experiences: list
    ) -> str:
        """
        Format experience logs for inclusion in system prompt.
        
        Parameters
        ----------
        experiences : list
            List of experience log dictionaries from ExperienceStore
        
        Returns
        -------
        str
            Formatted string for prompt inclusion
        """
        if not experiences:
            return ""
        
        formatted = "\n\nPast Experiences (Self-Reflection):\n"
        formatted += "=" * 50 + "\n"
        
        for i, exp in enumerate(experiences, 1):
            meta = exp.get('metadata', {})
            verdict = meta.get('verdict', 'UNKNOWN')
            critique = meta.get('critique_text', '')
            
            if verdict == "WARNING_HALLUCINATION":
                formatted += f"Log {i} (Warning): {critique}\n"
            elif verdict == "VERIFIED_SYNTHESIS":
                formatted += f"Log {i} (Success): {critique}\n"
            else:
                formatted += f"Log {i} (Partial): {critique}\n"
        
        formatted += "=" * 50 + "\n"
        formatted += "If a 'Warning' log is present, strictly avoid that specific error.\n"
        
        return formatted

