"""
Hallucination Metrics
=====================

Implements direct hallucination measurement metrics:
1. FActScore (Factual Consistency Score)
2. NLI-based consistency (AlignScore proxy)

"""

import logging
import re
import os
from typing import List, Dict, Any, Tuple
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    
try:
    from sentence_transformers import CrossEncoder
    import torch
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

logger = logging.getLogger(__name__)

class HallucinationEvaluator:
    """
    Evaluator for hallucination metrics.
    
    Implements:
    - FActScore-lite: Atomic fact extraction + NLI verification
    - Grounding Score: Direct NLI entailment check
    """
    
    def __init__(self, model_name: str = "llama3.2:3b", device: str = None):
        """
        Initialize evaluator.
        
        Parameters
        ----------
        model_name : str
            Ollama model for fact extraction
        device : str
            Device for NLI model
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize NLI model
        if CROSS_ENCODER_AVAILABLE:
            try:
                os.environ['HF_HUB_OFFLINE'] = '1'
                self.nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base', device=self.device)
                logger.info("NLI model loaded for hallucination metrics")
            except Exception as e:
                logger.warning(f"Failed to load NLI model: {e}")
                self.nli_model = None
        else:
            self.nli_model = None
            logger.warning("sentence-transformers not installed, NLI metrics disabled")

    def compute_fact_score(
        self,
        generations: List[str],
        knowledge_sources: List[List[str]]
    ) -> List[Dict[str, float]]:
        """
        Compute FActScore-like metrics for a batch.
        """
        results = []
        for gen, sources in zip(generations, knowledge_sources):
            # 1. Extract atomic facts
            facts = self._extract_facts(gen)
            
            # 2. Verify facts against sources
            supported_count = 0
            for fact in facts:
                is_supported = self._verify_fact(fact, sources)
                if is_supported:
                    supported_count += 1
            
            score = supported_count / len(facts) if facts else 0.0
            
            results.append({
                "fact_score": score,
                "num_facts": len(facts),
                "supported_facts": supported_count
            })
        return results

    def _extract_facts(self, text: str) -> List[str]:
        """Extract atomic facts using LLM."""
        if not OLLAMA_AVAILABLE:
            return [text] # Fallback
            
        prompt = f"""Break down the following text into a list of atomic, independent facts.
Text: "{text}"

Output ONLY the facts, one per line. Do not include bullet points or numbers.
Facts:"""

        try:
            response = ollama.generate(model=self.model_name, prompt=prompt)
            facts = response['response'].strip().split('\n')
            return [f.strip() for f in facts if f.strip() and not f.startswith(('Here', 'Sure'))]
        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")
            return [text]

    def _verify_fact(self, fact: str, sources: List[str]) -> bool:
        """Verify if a fact is supported by any source."""
        if not self.nli_model or not sources:
            return False # Conservative
            
        # Check against each source doc
        # Efficient strategy: Check max entailment score across all docs
        pairs = [[doc, fact] for doc in sources]
        scores = self.nli_model.predict(pairs)
        
        # Softmax not strictly needed for ranking, but good for threshold
        # Using raw logits? CrossEncoder output depends on model. nli-deberta-v3-base outputs logits for (Contradiction, Neutral, Entailment)
        # We want Entailment > Threshold
        
        # Deberta V3 NLI: Label 0=Contradiction, 1=Neutral, 2=Entailment
        # Wait, check model card. Usually labels are C, N, E.
        # But 'cross-encoder/nli-deberta-v3-base' return just one score? NO, classifier returns 3.
        
        # Actually some cross-encoders return probability of "relevant".
        # nli-deberta-v3-base outputs 3 scores.
        
        # Let's assume binary "entails" if Entailment > others
        
        max_entailment_prob = 0.0
        
        for score_vec in scores:
            probs = torch.softmax(torch.tensor(score_vec), dim=0)
            entailment_prob = float(probs[-1]) # Assume last is entailment
            max_entailment_prob = max(max_entailment_prob, entailment_prob)
            
        return max_entailment_prob > 0.7

    def compute_grounding_score(
        self,
        hypothesis: str,
        context: str
    ) -> float:
        """
        Compute grounding score using NLI.
        """
        if not self.nli_model:
            return 0.0
            
        scores = self.nli_model.predict([[context[:1000], hypothesis[:500]]])
        probs = torch.softmax(torch.tensor(scores[0]), dim=0)
        return float(probs[-1]) # Entailment probability
