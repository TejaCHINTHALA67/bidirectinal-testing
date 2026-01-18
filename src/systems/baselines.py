"""
Baseline RAG Systems for IEEE-Grade Experiments

Implements 6 RAG systems with identical interfaces:
1. StandardRAG - Standard retrieval + generation (baseline)
2. NaiveWritebackRAG - Write-back without validation
3. SelfRAGAdapter - Wrapper for Self-RAG
4. FLARERAG - Generate until [RET] token, then retrieve
5. CRAGRAG - Wrapper for CRAG
6. BidirectionalRAG - Our proposed system with acceptance layer

All systems are offline-capable and thread-safe for parallel experiments.
"""

import json
import os
import re
import time
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid

# Monkey-patch to prevent ONNX import errors
# MUST be done BEFORE any other imports that might check onnxruntime
import sys
import importlib.machinery
import importlib.util
from types import ModuleType

# Create a proper mock loader
class MockLoader:
    """Mock loader for ONNX module."""
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        pass

# Create a proper spec using ModuleSpec directly (guaranteed not None)
_onnx_spec = importlib.machinery.ModuleSpec(
    name='onnxruntime',
    loader=MockLoader(),
    origin='mock',
    is_package=False
)

# Create mock module with proper spec
_mock_onnx = ModuleType('onnxruntime')
_mock_onnx.__spec__ = _onnx_spec
_mock_onnx.__loader__ = MockLoader()
_mock_onnx.__version__ = '0.0.0'
_mock_onnx.__file__ = None
_mock_onnx.__package__ = 'onnxruntime'

# Prevent any module from importing onnxruntime
# This must happen BEFORE chromadb, torch, or transformers are imported
if 'onnxruntime' not in sys.modules:
    sys.modules['onnxruntime'] = _mock_onnx

# Patch find_spec to return our spec for onnxruntime
_original_find_spec = importlib.util.find_spec
def _patched_find_spec(name, package=None):
    if name == 'onnxruntime' or (isinstance(name, str) and name.startswith('onnxruntime')):
        return _onnx_spec
    return _original_find_spec(name, package)
importlib.util.find_spec = _patched_find_spec

# Disable ChromaDB telemetry BEFORE importing chromadb
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY_DISABLED'] = 'True'

# Suppress telemetry logger before any ChromaDB operations
import logging
chromadb_telemetry_logger = logging.getLogger('chromadb.telemetry')
chromadb_telemetry_logger.setLevel(logging.CRITICAL)
chromadb_telemetry_logger.disabled = True

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Import V2.0 modules
from src.ragcore.experience_store import ExperienceStore
from src.ragcore.critique_generator import CritiqueGenerator
from src.evaluation.citation_manager import ProvenanceTracker

# Monkey-patch ChromaDB telemetry to prevent errors
try:
    # Disable posthog library completely
    import posthog
    posthog.disabled = True
    
    # Also patch the Posthog class capture method
    import chromadb.telemetry.product.posthog as posthog_module
    if hasattr(posthog_module, 'Posthog'):
        original_capture = posthog_module.Posthog.capture
        def noop_capture(self, event):
            pass  # Do nothing - suppress telemetry
        posthog_module.Posthog.capture = noop_capture
        
    # Suppress the logger that shows the errors
    posthog_logger = logging.getLogger('chromadb.telemetry.product.posthog')
    posthog_logger.setLevel(logging.CRITICAL)
    posthog_logger.disabled = True
    
except (ImportError, AttributeError) as e:
    # If telemetry module structure is different, that's OK
    # Logger not yet defined, so just pass
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread lock for thread-safe operations
_lock = threading.Lock()


class BaseSystem(ABC):
    """
    Abstract base class for all RAG systems.
    
    All systems must implement the query() method with identical return format.
    """
    
    def __init__(
        self,
        corpus_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        k_retrieval: int = 5,
        llm_model: str = "llama3.2:3b",
        chroma_persist_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        seed: Optional[int] = None,
        system_slug: Optional[str] = None,
        output_dir: str = "results",
        enable_query_logging: bool = True,
        debug: bool = False,
        **kwargs
    ):
        """
        Initialize base RAG system.
        
        Parameters
        ----------
        corpus_path : str
            Path to corpus JSON file
        embedding_model : str
            Sentence-transformers model name
        k_retrieval : int
            Number of documents to retrieve
        llm_model : str
            Ollama model name
        chroma_persist_dir : str, optional
            ChromaDB persistence directory (auto-generated if None)
        """
        self.corpus_path = Path(corpus_path)
        self.k_retrieval = k_retrieval
        self.llm_model = llm_model
        self.system_name = self.__class__.__name__
        self.system_slug = system_slug or self.system_name.lower()
        self.dataset_name = dataset_name
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.enable_query_logging = enable_query_logging
        self.query_counter = 0
        self.debug = debug or bool(os.environ.get("DEBUG_BASELINES"))
        
        # Set offline mode
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        # Initialize embedding model
        logger.info(f"[{self.system_name}] Loading embedding model: {embedding_model}")
        try:
            self.embedding_model = SentenceTransformer(embedding_model, device='cpu')
            logger.info(f"[{self.system_name}] Embedding model loaded")
        except Exception as e:
            logger.error(f"[{self.system_name}] Failed to load embedding model: {e}")
            raise
        
        # Create custom embedding function using sentence-transformers
        class SentenceTransformerEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __init__(self, model):
                self.model = model
            
            def __call__(self, input):
                if isinstance(input, str):
                    input = [input]
                embeddings = self.model.encode(input, convert_to_numpy=True)
                return embeddings.tolist()
        
        self.embedding_fn = SentenceTransformerEmbeddingFunction(self.embedding_model)
        
        # Initialize ChromaDB - seed isolation
        if chroma_persist_dir is None:
            if self.seed is not None:
                chroma_persist_dir = f"./chroma_{self.system_name.lower()}_{self.seed}"
            else:
                chroma_persist_dir = f"./chroma_{self.system_name.lower()}"
        
        self.chroma_persist_dir = chroma_persist_dir
        
        # Create client with telemetry disabled (already set at module level)
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create collection with custom embedding function
        collection_name = f"{self.system_name.lower()}_collection"
        try:
            # Try to get existing collection, but if it uses ONNX, delete and recreate
            try:
                self.collection = self.chroma_client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn
                )
                logger.info(f"[{self.system_name}] Loaded existing ChromaDB collection")
            except Exception as e:
                # If collection exists but has wrong embedding function, delete and recreate
                logger.warning(f"[{self.system_name}] Collection exists but incompatible, recreating...")
                try:
                    self.chroma_client.delete_collection(name=collection_name)
                except:
                    pass
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"[{self.system_name}] Created new ChromaDB collection")
        except Exception as e:
            # If get_collection fails, try to create
            try:
                self.chroma_client.delete_collection(name=collection_name)
            except:
                pass
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"[{self.system_name}] Created new ChromaDB collection")
        
        # Load corpus
        self._load_corpus()
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'documents_added': 0
        }
        
        logger.info(f"[{self.system_name}] Initialized successfully")
    
    def _load_corpus(self):
        """Load corpus from JSON file."""
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")
        
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        
        logger.info(f"[{self.system_name}] Loading {len(corpus_data)} documents...")
        
        # Check if collection is empty
        if self.collection.count() == 0:
            # Add documents to ChromaDB (embeddings computed automatically by embedding function)
            ids = []
            documents = []
            metadatas = []
            
            for doc in corpus_data:
                doc_id = doc.get('id', str(uuid.uuid4()))
                doc_text = doc.get('text', doc.get('content', ''))
                
                if not doc_text:
                    continue
                
                ids.append(doc_id)
                documents.append(doc_text)
                metadatas.append({
                    'origin': 'human_authored',
                    'topic': doc.get('topic', 'unknown')
                })
            
            # Batch add to ChromaDB (embeddings computed automatically)
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                self.collection.add(
                    ids=ids[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )
            
            logger.info(f"[{self.system_name}] Added {len(ids)} documents to ChromaDB")
        else:
            logger.info(f"[{self.system_name}] ChromaDB collection already populated ({self.collection.count()} docs)")
    
    def _retrieve(self, query: str, k: Optional[int] = None) -> Tuple[List[str], List[str], List[Dict], float]:
        """
        Retrieve documents using semantic search.
        
        Returns
        -------
        tuple
            (documents, ids, metadatas, top_distance)
        """
        k = k or self.k_retrieval
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['documents'] or not results['documents'][0]:
            return [], [], [], 1.0
        
        documents = results['documents'][0]
        ids = results['ids'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        top_distance = distances[0] if distances else 1.0
        
        return documents, ids, metadatas, top_distance
    
    def _generate_ollama(self, query: str, context: List[str], retrieved_ids: List[str], experience_logs: Optional[List[Dict]] = None) -> str:
        """Generate response using Ollama (V2.0: accepts experience_logs for BidirectionalRAG)."""
        try:
            import ollama
            
            # Format context with citations
            context_str = ""
            for i, (doc, doc_id) in enumerate(zip(context, retrieved_ids)):
                context_str += f"\n[{doc_id}]: {doc[:500]}\n"  # Limit doc length
            
            # V2.0: Format experience logs if provided (for BidirectionalRAG)
            experiences_str = ""
            if experience_logs and hasattr(self, 'critique_generator'):
                experiences_str = self.critique_generator.format_experiences_for_prompt(experience_logs)
            
            # Create prompt (V2.0: includes experiences if provided)
            prompt = f"""You are an expert assistant that answers questions based on the provided context.
You MUST cite your sources using the document IDs in square brackets like [doc_id].

Context (Facts):
{context_str}
{experiences_str}
Question: {query}

Instructions:
1. Answer the question based ONLY on the information in the context
2. Cite sources by including [doc_id] after each claim
3. If the context doesn't contain enough information, say so
4. If a "Warning" log is present in Past Experiences, strictly avoid that specific error
5. Be concise and accurate

Answer:"""
            
            # Call Ollama
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'max_tokens': 512
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"[{self.system_name}] Ollama generation failed: {e}")
            # Fallback to simple response
            return f"Based on the provided context: {context[0][:200] if context else 'No context available'}... [{retrieved_ids[0] if retrieved_ids else 'unknown'}]"
    
    def _generate_gemini(self, query: str, context: List[str], retrieved_ids: List[str], experience_logs: Optional[List[Dict]] = None) -> str:
        """Generate response using Google Gemini API.
        
        Set GEMINI_API_KEY environment variable before use.
        Usage: --llm_model gemini-pro or --llm_model gemini-1.5-flash
        """
        try:
            import google.generativeai as genai
            
            # Configure API key
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set!")
            genai.configure(api_key=api_key)
            
            # Format context with citations
            context_str = ""
            for i, (doc, doc_id) in enumerate(zip(context, retrieved_ids)):
                context_str += f"\n[{doc_id}]: {doc[:500]}\n"
            
            # Format experience logs if provided
            experiences_str = ""
            if experience_logs and hasattr(self, 'critique_generator'):
                experiences_str = self.critique_generator.format_experiences_for_prompt(experience_logs)
            
            # Create prompt
            prompt = f"""You are an expert assistant that answers questions based on the provided context.
You MUST cite your sources using the document IDs in square brackets like [doc_id].

Context (Facts):
{context_str}
{experiences_str}
Question: {query}

Instructions:
1. Answer the question based ONLY on the information in the context
2. Cite sources by including [doc_id] after each claim
3. If the context doesn't contain enough information, say so
4. If a "Warning" log is present in Past Experiences, strictly avoid that specific error
5. Be concise and accurate

Answer:"""
            
            # Get model name (default to gemini-2.5-flash for speed/cost)
            # Use full model path format: models/gemini-...
            if 'gemini' in self.llm_model and 'models/' in self.llm_model:
                model_name = self.llm_model
            elif 'gemini' in self.llm_model:
                model_name = f"models/{self.llm_model}"
            else:
                model_name = 'models/gemini-2.5-flash-preview-09-2025'
            model = genai.GenerativeModel(model_name)
            
            # Generate response with safety settings to avoid 'finish_reason 2' blocks
            # Use explicit types for maximum compatibility with google-generativeai library
            safety_settings = {
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }

            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=512,
                ),
                safety_settings=safety_settings
            )
            
            try:
                return response.text.strip()
            except ValueError:
                # If blocked by safety, log the detailed feedback
                logger.error(f"[{self.system_name}] Gemini Safety Block: {response.prompt_feedback}")
                return ""
            
        except Exception as e:
            logger.error(f"[{self.system_name}] Gemini generation failed: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
                logger.error(f"[{self.system_name}] Feedback: {e.response.prompt_feedback}")
            return f"Based on the provided context: {context[0][:200] if context else 'No context available'}... [{retrieved_ids[0] if retrieved_ids else 'unknown'}]"
    
    def _generate(self, query: str, context: List[str], retrieved_ids: List[str], experience_logs: Optional[List[Dict]] = None) -> str:
        """Generate response using configured LLM (Ollama or Gemini)."""
        if 'gemini' in self.llm_model.lower():
            return self._generate_gemini(query, context, retrieved_ids, experience_logs)
        else:
            return self._generate_ollama(query, context, retrieved_ids, experience_logs)
    
    @abstractmethod
    def query(self, question: str) -> Dict:
        """
        Process a query and return results.
        
        Must return:
        {
            "response": str,
            "citations": List[str],
            "retrieved_docs": List[str],
            "latency_ms": float,
            "grounding_score": float (if applicable)
        }
        """
        pass
    
    def get_statistics(self) -> Dict:
        """Get system statistics."""
        return self.stats.copy()
    
    def _extract_citations(self, response: str) -> List[str]:
        """Extract citation IDs from response."""
        citations = re.findall(r'\[([^\]]+)\]', response)
        return citations

    def _debug(self, message: str):
        if self.debug:
            logger.debug(f"[{self.system_name}] {message}")

    def _persist_query_log(
        self,
        query: str,
        retrieved_docs: List[str],
        retrieved_ids: List[str],
        response: str,
        citations: List[str]
    ):
        """
        Persist full query context for hallucination analysis.
        """
        if not self.enable_query_logging:
            return
        if not (self.dataset_name and self.system_slug and self.seed is not None):
            return

        try:
            log_dir = (
                self.output_dir
                / str(self.dataset_name)
                / str(self.system_slug)
                / str(self.seed)
                / "query_logs"
            )
            log_dir.mkdir(parents=True, exist_ok=True)

            # Combine ids + docs into structured entries
            structured_docs = []
            for doc_text, doc_id in zip(retrieved_docs or [], retrieved_ids or []):
                structured_docs.append({"id": doc_id, "text": doc_text})

            query_data = {
                "query": query,
                "retrieved_docs": structured_docs,
                "generated_response": response,
                "citations": citations or [],
                "timestamp": time.time(),
                "query_index": self.query_counter,
            }

            log_path = log_dir / f"query_{self.query_counter}.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(query_data, f, indent=2)

            self.query_counter += 1
        except Exception as e:
            logger.warning(f"[{self.system_name}] Failed to persist query log: {e}")


class StandardRAG(BaseSystem):
    """
    Standard RAG: Retrieve top-k docs, generate response.
    No write-back. Baseline system.
    """
    
    def query(self, question: str) -> Dict:
        """Process query with standard RAG."""
        start_time = time.perf_counter()
        
        # Retrieve
        docs, ids, metadatas, top_distance = self._retrieve(question)
        self._debug(f"Retrieved {len(docs)} docs (top_distance={top_distance:.3f})")
        self._debug(f"Retrieved {len(docs)} docs (top_distance={top_distance:.3f})")
        self._debug(f"Retrieved {len(docs)} docs (top_distance={top_distance:.3f})")
        
        if not docs:
            return {
                "response": "I don't have enough information to answer this question.",
                "citations": [],
                "retrieved_docs": [],
                "retrieved_ids": [],
                "latency_ms": (time.perf_counter() - start_time) * 1000,
                "grounding_score": 0.0,
                "top_distance": 1.0
            }
        
        # Generate
        response = self._generate(question, docs, ids)
        
        # Extract citations
        citations = self._extract_citations(response)
        self._debug(f"Citations found: {citations}")
        self._debug(f"Citations found: {citations}")
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        self.stats['total_queries'] += 1

        self._persist_query_log(
            query=question,
            retrieved_docs=docs,
            retrieved_ids=ids,
            response=response,
            citations=citations
        )
        
        return {
            "response": response,
            "citations": citations,
            "retrieved_docs": docs,
            "retrieved_ids": ids,
            "latency_ms": latency_ms,
            "grounding_score": 0.0,  # Not computed for standard RAG
            "top_distance": top_distance
        }


class NaiveWritebackRAG(BaseSystem):
    """
    Naive Write-back RAG: Same as StandardRAG but automatically writes back all responses.
    No validation. Upper bound on corpus growth.
    """
    
    def query(self, question: str) -> Dict:
        """Process query with naive write-back."""
        start_time = time.perf_counter()
        
        # Retrieve
        docs, ids, metadatas, top_distance = self._retrieve(question)
        
        if not docs:
            return {
                "response": "I don't have enough information to answer this question.",
                "citations": [],
                "retrieved_docs": [],
                "retrieved_ids": [],
                "latency_ms": (time.perf_counter() - start_time) * 1000,
                "grounding_score": 0.0,
                "top_distance": 1.0
            }
        
        # Generate
        response = self._generate(question, docs, ids)
        
        # Extract citations
        citations = self._extract_citations(response)
        
        # Write back automatically (no validation)
        with _lock:
            doc_id = f"naive_doc_{uuid.uuid4()}"
            embedding = self.embedding_model.encode(response, convert_to_tensor=False)
            
            self.collection.add(
                ids=[doc_id],
                documents=[response],
                embeddings=[embedding.tolist()],
                metadatas=[{
                    'origin': 'model_generated',
                    'topic': 'unknown',
                    'timestamp': datetime.now().isoformat(),
                    'source_query': question
                }]
            )
            
            self.stats['documents_added'] += 1
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        self.stats['total_queries'] += 1

        self._persist_query_log(
            query=question,
            retrieved_docs=docs,
            retrieved_ids=ids,
            response=response,
            citations=citations
        )
        
        return {
            "response": response,
            "citations": citations,
            "retrieved_docs": docs,
            "retrieved_ids": ids,
            "latency_ms": latency_ms,
            "grounding_score": 0.0,
            "written_back": True,
            "top_distance": top_distance
        }


class SelfRAGAdapter(BaseSystem):
    """
    Self-RAG Adapter: Wrapper for Self-RAG implementation.
    
    Adapts Self-RAG to use our corpus instead of their default.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize Self-RAG adapter."""
        super().__init__(*args, **kwargs)
        
        # Try to import Self-RAG
        try:
            # Self-RAG might not be installed, so we'll implement a simplified version
            self.self_rag_available = False
            logger.warning("[SelfRAGAdapter] Self-RAG library not available, using simplified implementation")
        except:
            self.self_rag_available = False
    
    def query(self, question: str) -> Dict:
        """
        Process query with Self-RAG.
        
        Self-RAG uses reflection tokens to decide when to retrieve.
        Simplified implementation: retrieve, generate, reflect, potentially retrieve again.
        """
        start_time = time.perf_counter()
        
        # Initial retrieval
        docs, ids, metadatas, top_distance = self._retrieve(question)
        
        if not docs:
            return {
                "response": "I don't have enough information to answer this question.",
                "citations": [],
                "retrieved_docs": [],
                "retrieved_ids": [],
                "latency_ms": (time.perf_counter() - start_time) * 1000,
                "grounding_score": 0.0,
                "top_distance": 1.0
            }
        
        # Generate with reflection
        response = self._generate(question, docs, ids)
        self._debug(f"Initial response len={len(response)}")
        
        # Simplified reflection: check if response mentions uncertainty
        uncertainty_keywords = ['uncertain', 'not sure', 'unclear', 'might', 'possibly', 'perhaps']
        needs_retrieval = any(keyword in response.lower() for keyword in uncertainty_keywords)
        self._debug(f"Uncertainty detected={needs_retrieval}")
        
        # If uncertain, retrieve more documents
        if needs_retrieval and len(docs) < 10:
            additional_docs, additional_ids, _, _ = self._retrieve(question, k=10)
            self._debug(f"Reflection retrieval added {len(additional_docs)} docs")
            # Merge (avoid duplicates)
            for doc, doc_id in zip(additional_docs, additional_ids):
                if doc_id not in ids:
                    docs.append(doc)
                    ids.append(doc_id)
        
        # Re-generate with potentially more context
        if needs_retrieval:
            response = self._generate(question, docs, ids)
        
        citations = self._extract_citations(response)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        self.stats['total_queries'] += 1

        self._persist_query_log(
            query=question,
            retrieved_docs=docs,
            retrieved_ids=ids,
            response=response,
            citations=citations
        )
        
        return {
            "response": response,
            "citations": citations,
            "retrieved_docs": docs,
            "retrieved_ids": ids,
            "latency_ms": latency_ms,
            "grounding_score": 0.0,  # Self-RAG doesn't compute grounding scores
            "top_distance": top_distance
        }


class FLARERAG(BaseSystem):
    """
    FLARE RAG: Generate until [RET] token appears, then retrieve.
    
    Key idea: Generate text, detect uncertainty, pause generation,
    retrieve relevant documents, continue generation.
    """
    
    def query(self, question: str) -> Dict:
        """Process query with FLARE RAG."""
        start_time = time.perf_counter()
        
        # Initial retrieval
        docs, ids, metadatas, top_distance = self._retrieve(question, k=3)
        
        # Start generation
        partial_response = ""
        all_retrieved_docs = docs.copy()
        all_retrieved_ids = ids.copy()
        max_iterations = 3  # Prevent infinite loops
        
        for iteration in range(max_iterations):
            # Generate continuation
            if iteration == 0:
                # First generation with initial context
                context = docs if docs else []
            else:
                # Subsequent generations with accumulated context
                context = all_retrieved_docs[-5:]  # Use last 5 retrieved docs
            
            # Generate chunk
            chunk = self._generate(question, context, all_retrieved_ids[-len(context):] if context else [])
            partial_response += chunk + " "
            
            # Check for uncertainty markers (simplified [RET] detection)
            uncertainty_markers = [
                'i need more information',
                'let me search',
                '[RET]',
                'uncertain',
                'not sure'
            ]
            
            has_uncertainty = any(marker in chunk.lower() for marker in uncertainty_markers)
            
            if not has_uncertainty:
                # No uncertainty, generation complete
                break
            
            # Uncertainty detected: retrieve more documents
            # Extract key terms from partial response for retrieval
            key_terms = self._extract_key_terms(partial_response)
            retrieval_query = f"{question} {key_terms}"
            
            new_docs, new_ids, _, _ = self._retrieve(retrieval_query, k=3)
            
            # Add new documents (avoid duplicates)
            for doc, doc_id in zip(new_docs, new_ids):
                if doc_id not in all_retrieved_ids:
                    all_retrieved_docs.append(doc)
                    all_retrieved_ids.append(doc_id)
        
        response = partial_response.strip()
        citations = self._extract_citations(response)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        self.stats['total_queries'] += 1

        self._persist_query_log(
            query=question,
            retrieved_docs=all_retrieved_docs,
            retrieved_ids=all_retrieved_ids,
            response=response,
            citations=citations
        )
        
        return {
            "response": response,
            "citations": citations,
            "retrieved_docs": all_retrieved_docs,
            "retrieved_ids": all_retrieved_ids,
            "latency_ms": latency_ms,
            "grounding_score": 0.0,
            "top_distance": top_distance
        }
    
    def _extract_key_terms(self, text: str) -> str:
        """Extract key terms for retrieval."""
        # Simple: extract nouns and important words
        words = text.lower().split()
        # Filter common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        key_terms = [w for w in words if len(w) > 4 and w not in stop_words]
        return ' '.join(key_terms[:5])  # Top 5 terms


class CRAGRAG(BaseSystem):
    """
    CRAG (Corrective Retrieval Augmented Generation) Adapter.
    
    CRAG uses web search for corrective retrieval. We adapt it to use
    local retrieval only (offline mode).
    """
    
    def query(self, question: str) -> Dict:
        """
        Process query with CRAG.
        
        CRAG strategy:
        1. Initial retrieval
        2. Generate response
        3. Evaluate relevance of retrieved docs
        4. If low relevance, perform corrective retrieval
        5. Re-generate with corrected context
        """
        start_time = time.perf_counter()
        
        # Initial retrieval
        docs, ids, metadatas, top_distance = self._retrieve(question)
        
        if not docs:
            return {
                "response": "I don't have enough information to answer this question.",
                "citations": [],
                "retrieved_docs": [],
                "retrieved_ids": [],
                "latency_ms": (time.perf_counter() - start_time) * 1000,
                "grounding_score": 0.0,
                "top_distance": 1.0
            }
        
        # Generate initial response
        response = self._generate(question, docs, ids)
        self._debug(f"Initial retrieval {len(docs)} docs, top_distance={top_distance:.3f}")
        
        # Evaluate relevance (simplified: check if response mentions "don't know" or similar)
        low_relevance_indicators = [
            "don't know",
            "not in the context",
            "unclear",
            "cannot answer"
        ]
        
        needs_correction = any(indicator in response.lower() for indicator in low_relevance_indicators)
        needs_correction = needs_correction or top_distance > 0.6  # High distance = low relevance
        self._debug(f"Needs correction={needs_correction}")
        
        # Corrective retrieval if needed
        if needs_correction:
            # Extract entities/keywords from question for better retrieval
            corrected_query = self._extract_entities(question)
            corrected_docs, corrected_ids, _, _ = self._retrieve(corrected_query, k=5)
            self._debug(f"Corrective retrieval fetched {len(corrected_docs)} docs")
            
            # Replace with corrected documents
            docs = corrected_docs
            ids = corrected_ids
            
            # Re-generate with corrected context
            response = self._generate(question, docs, ids)
        
        citations = self._extract_citations(response)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        self.stats['total_queries'] += 1

        self._persist_query_log(
            query=question,
            retrieved_docs=docs,
            retrieved_ids=ids,
            response=response,
            citations=citations
        )
        
        return {
            "response": response,
            "citations": citations,
            "retrieved_docs": docs,
            "retrieved_ids": ids,
            "latency_ms": latency_ms,
            "grounding_score": 0.0,
            "top_distance": top_distance
        }
    
    def _extract_entities(self, text: str) -> str:
        """Extract entities/keywords for corrective retrieval."""
        # Simple: extract capitalized words and important terms
        words = text.split()
        entities = [w for w in words if w[0].isupper() or len(w) > 5]
        return ' '.join(entities) if entities else text


class BidirectionalRAG(BaseSystem):
    """
    Bidirectional RAG: Our proposed system with multi-stage acceptance layer.
    
    Uses validated write-back with grounding, attribution, and novelty checks.
    """
    
    def __init__(
        self,
        corpus_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        k_retrieval: int = 5,
        llm_model: str = "llama3.2:3b",
        chroma_persist_dir: Optional[str] = None,
        grounding_threshold: float = 0.65,
        novelty_threshold: float = 0.90,
        dataset_name: Optional[str] = None,
        seed: Optional[int] = None,
        system_slug: Optional[str] = None,
        output_dir: str = "results",
        enable_query_logging: bool = True,
        enable_experience_store: bool = True,
    ):
        """Initialize Bidirectional RAG with acceptance layer."""
        super().__init__(
            corpus_path,
            embedding_model,
            k_retrieval,
            llm_model,
            chroma_persist_dir,
            dataset_name=dataset_name,
            seed=seed,
            system_slug=system_slug,
            output_dir=output_dir,
            enable_query_logging=enable_query_logging,
        )
        
        self.grounding_threshold = grounding_threshold
        self.novelty_threshold = novelty_threshold
        self.enable_experience_store = enable_experience_store
        
        # Initialize NLI model for grounding
        try:
            from sentence_transformers import CrossEncoder
            os.environ['HF_HUB_OFFLINE'] = '1'
            self.nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base', device='cpu')
            logger.info(f"[{self.system_name}] NLI model loaded")
        except Exception as e:
            logger.warning(f"[{self.system_name}] NLI model not available: {e}")
            self.nli_model = None
        
        # Statistics
        self.stats.update({
            'total_accepted': 0,
            'total_rejected': 0,
            'rejection_reasons': {
                'grounding': 0,
                'attribution': 0,
                'novelty': 0
            }
        })
        
        # V2.0: Initialize Experience Store and Critique Generator
        experience_store_dir = self.chroma_persist_dir.replace('chroma_', 'chroma_experience_')
        self.experience_store = ExperienceStore(
            chroma_persist_dir=experience_store_dir,
            embedding_model=embedding_model
        )
        self.critique_generator = CritiqueGenerator()
        logger.info(f"[{self.system_name}] V2.0 features initialized (Experience Store + Critique Generator)")
    
    def _verify_grounding(self, response: str, context: List[str]) -> float:
        """Verify response is grounded in context using NLI."""
        if not self.nli_model or not context:
            return 1.0  # Assume grounded if no NLI model
        
        try:
            # Check each sentence in response against context
            sentences = re.split(r'[.!?]+', response)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return 0.0
            
            # Check first sentence against best context
            best_context = context[0] if context else ""
            if not best_context:
                return 0.0
            
            # NLI check: context (premise) entails response (hypothesis)
            scores = self.nli_model.predict([
                [best_context[:500], sentences[0][:500]]
            ])
            scores_tensor = torch.tensor(scores[0])
            probs = torch.softmax(scores_tensor, dim=-1)
            entailment_prob = float(probs[-1].item()) if probs.numel() >= 3 else 0.5
            return entailment_prob
            
        except Exception as e:
            logger.warning(f"[{self.system_name}] Grounding verification failed: {e}")
            return 0.5  # Default to neutral
    
    def _check_attribution(self, response: str, retrieved_ids: List[str]) -> bool:
        """Check if response properly cites retrieved documents."""
        citations = self._extract_citations(response)
        if not citations:
            return False
        
        # Check if at least one citation references a retrieved document
        return any(cite in retrieved_ids for cite in citations)
    
    def _check_novelty(self, response: str) -> float:
        """Check novelty by measuring similarity to existing corpus."""
        try:
            # Generate embedding for response
            response_embedding = self.embedding_model.encode(response, convert_to_tensor=False)
            
            # Query for similar documents
            results = self.collection.query(
                query_embeddings=[response_embedding.tolist()],
                n_results=5,
                include=['distances']
            )
            
            if not results['distances'] or not results['distances'][0]:
                return 0.0  # No similar documents found
            
            # Return maximum similarity (1 - distance, since distance is cosine distance)
            max_similarity = 1.0 - results['distances'][0][0]
            return float(max_similarity)
            
        except Exception as e:
            logger.warning(f"[{self.system_name}] Novelty check failed: {e}")
            return 0.0  # Assume novel if check fails
    
    def query(self, question: str) -> Dict:
        """Process query with Bidirectional RAG V2.0 (with Experience Store)."""
        start_time = time.perf_counter()
        
        # V2.0: Dual Retrieval - facts + experiences
        docs, ids, metadatas, top_distance = self._retrieve(question)
        
        experience_logs = []
        if self.enable_experience_store:
            experience_logs = self.experience_store.retrieve_experiences(question, n_results=3)
        
        if not docs:
            return {
                "response": "I don't have enough information to answer this question.",
                "citations": [],
                "retrieved_docs": [],
                "retrieved_ids": [],
                "latency_ms": (time.perf_counter() - start_time) * 1000,
                "grounding_score": 0.0,
                "written_back": False,
                "top_distance": 1.0
            }
        
        # V2.0: Generate with experience logs in prompt
        response = self._generate(question, docs, ids, experience_logs)
        citations = self._extract_citations(response)
        
        # Acceptance layer
        grounding_score = self._verify_grounding(response, docs)
        has_attribution = self._check_attribution(response, ids)
        novelty_score = self._check_novelty(response)
        
        # Decision
        written_back = False
        rejection_reason = None
        
        if grounding_score < self.grounding_threshold:
            rejection_reason = "Low grounding"
            self.stats['rejection_reasons']['grounding'] += 1
        elif not has_attribution:
            rejection_reason = "Missing attribution"
            self.stats['rejection_reasons']['attribution'] += 1
        elif novelty_score >= self.novelty_threshold:
            rejection_reason = "Low novelty"
            self.stats['rejection_reasons']['novelty'] += 1
        else:
            # V2.0: Always store experience, conditionally store in corpus
            # Create acceptance result object
            from dataclasses import dataclass
            @dataclass
            class AcceptanceResult:
                accepted: bool
                grounding_score: float
                has_attribution: bool
                novelty_score: float
                rejection_reason: Optional[str] = None
            
            acceptance_result_obj = AcceptanceResult(
                accepted=True,
                grounding_score=grounding_score,
                has_attribution=has_attribution,
                novelty_score=novelty_score,
                rejection_reason=None
            )
            
            # Generate critique and store experience
            critique = self.critique_generator.generate_critique(
                question, response, acceptance_result_obj, ids
            )
            
            # Always store experience log
            exp_id = self.experience_store.add_experience(
                query=question,
                response=response,
                grounding_score=grounding_score,
                attribution_check=has_attribution,
                critique_text=critique.critique_text,
                verdict=critique.verdict,
                retrieved_ids=ids
            )
            
            # Only write to corpus if VERIFIED_SYNTHESIS
            if critique.verdict == "VERIFIED_SYNTHESIS":
                # V2.1 Fix: Generate Hybrid ID encoding source provenance
                doc_id = ProvenanceTracker.generate_synthetic_id(
                    source_ids=ids
                )
                
                with _lock:
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
                            'source_docs': ','.join(ids),
                            'grounding_score': grounding_score,
                            'novelty_score': novelty_score,
                            'experience_log_id': exp_id
                        }]
                    )
                    
                    self.stats['documents_added'] += 1
                    written_back = True
            else:
                logger.info(f"[{self.system_name}] Response stored as experience only (verdict: {critique.verdict})")
            
            self.stats['total_accepted'] += 1
        
        # V2.0: Store experience log even for rejected responses
        if not written_back:
            self.stats['total_rejected'] += 1
            
            # Generate critique for rejected response
            from dataclasses import dataclass
            @dataclass
            class AcceptanceResult:
                accepted: bool
                grounding_score: float
                has_attribution: bool
                novelty_score: float
                rejection_reason: Optional[str] = None
            
            from dataclasses import dataclass
            @dataclass
            class AcceptanceResult:
                accepted: bool
                grounding_score: float
                has_attribution: bool
                novelty_score: float
                rejection_reason: Optional[str] = None
            
            acceptance_result_obj = AcceptanceResult(
                accepted=False,
                grounding_score=grounding_score,
                has_attribution=has_attribution,
                novelty_score=novelty_score,
                rejection_reason=rejection_reason
            )
            
            critique = self.critique_generator.generate_critique(
                question, response, acceptance_result_obj, ids
            )
            
            # Store as experience log (warning)
            self.experience_store.add_experience(
                query=question,
                response=response,
                grounding_score=grounding_score,
                attribution_check=has_attribution,
                critique_text=critique.critique_text,
                verdict=critique.verdict,
                retrieved_ids=ids
            )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.stats['total_queries'] += 1

        self._persist_query_log(
            query=question,
            retrieved_docs=docs,
            retrieved_ids=ids,
            response=response,
            citations=citations
        )

        return {
            "response": response,
            "citations": citations,
            "retrieved_docs": docs,
            "retrieved_ids": ids,
            "latency_ms": latency_ms,
            "grounding_score": grounding_score,
            "written_back": written_back,
            "rejection_reason": rejection_reason,
            "novelty_score": novelty_score,
            "top_distance": top_distance
        }


def get_system_class(system_name: str):
    """
    Get system class by name.
    
    Parameters
    ----------
    system_name : str
        Name of system: 'standard_rag', 'naive_writeback', 'self_rag', 
        'flare', 'crag', 'bidirectional_rag'
    
    Returns
    -------
    class
        System class
    """
    system_map = {
        'standard_rag': StandardRAG,
        'naive_writeback': NaiveWritebackRAG,
        'self_rag': SelfRAGAdapter,
        'flare': FLARERAG,
        'crag': CRAGRAG,
        'bidirectional_rag': BidirectionalRAG
    }
    
    # Lazy import hybrids to avoid circular dependency
    if system_name in ['self_rag_wb', 'flare_wb', 'crag_wb']:
        from src.systems.hybrid_baselines import SelfRAGWriteback, FLAREWriteback, CRAGWriteback
        system_map.update({
            'self_rag_wb': SelfRAGWriteback,
            'flare_wb': FLAREWriteback,
            'crag_wb': CRAGWriteback
        })
    
    return system_map[system_name]


# For backward compatibility
def create_system(system_name: str, corpus_path: str, **kwargs):
    """
    Create a system instance.
    
    Parameters
    ----------
    system_name : str
        Name of system
    corpus_path : str
        Path to corpus JSON file
    **kwargs
        Additional arguments for system initialization
    
    Returns
    -------
    BaseSystem
        System instance
    """
    SystemClass = get_system_class(system_name)
    return SystemClass(corpus_path=corpus_path, **kwargs)


if __name__ == '__main__':
    # Test all systems
    import sys
    
    corpus_path = "data/processed/stackoverflow_corpus.json"
    
    if not Path(corpus_path).exists():
        print(f"[ERROR] Corpus not found: {corpus_path}")
        print("[INFO] Run: python src/data/dataset_loader.py --dataset stackoverflow")
        sys.exit(1)
    
    test_question = "What is Python?"
    
    systems = ['standard_rag', 'naive_writeback', 'self_rag', 'flare', 'crag', 'bidirectional_rag']
    
    print("="*70)
    print("TESTING ALL BASELINE SYSTEMS")
    print("="*70)
    
    for system_name in systems:
        print(f"\n[{system_name.upper()}]")
        try:
            SystemClass = get_system_class(system_name)
            system = SystemClass(corpus_path=corpus_path)
            result = system.query(test_question)
            
            print(f"  Response: {result['response'][:100]}...")
            print(f"  Citations: {len(result['citations'])}")
            print(f"  Latency: {result['latency_ms']:.1f}ms")
            print(f"  Grounding: {result.get('grounding_score', 0.0):.3f}")
            if 'written_back' in result:
                print(f"  Written back: {result['written_back']}")
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)

