"""
Embedding Service for Semantic Similarity.

Provides real vector embeddings for column-to-property matching in the Starchart mapper.
Uses sentence-transformers for high-quality semantic embeddings with optional caching.

Usage:
    embedder = EmbeddingService()  # Lazy-loads model on first use
    scores = embedder.rank_properties("customer_email", properties)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions
FALLBACK_MODEL = "paraphrase-MiniLM-L3-v2"  # Even faster, slightly lower quality


@dataclass
class PropertyInfo:
    """Information about an ontology property for embedding."""
    uri: str
    label: str
    aliases: list[str] = field(default_factory=list)
    description: Optional[str] = None
    domain: Optional[str] = None
    range: Optional[str] = None


@dataclass
class SimilarityResult:
    """Result of similarity scoring."""
    property_uri: str
    label: str
    score: float
    confidence: str  # "high", "medium", "low"
    match_type: str  # "label", "alias", "description"


class EmbeddingService:
    """
    Embedding service for semantic similarity matching.
    
    Uses sentence-transformers for real embeddings with lazy model loading.
    Falls back to string-based similarity if transformers not available.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or Path.home() / ".cache" / "rdf-starbase" / "embeddings"
        self.use_cache = use_cache
        self._model = None
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._embeddings_available = None
        
    @property
    def embeddings_available(self) -> bool:
        """Check if sentence-transformers is available."""
        if self._embeddings_available is None:
            try:
                import sentence_transformers  # noqa: F401
                self._embeddings_available = True
            except ImportError:
                self._embeddings_available = False
                logger.warning(
                    "sentence-transformers not installed. "
                    "Using string-based similarity fallback. "
                    "Install with: pip install sentence-transformers"
                )
        return self._embeddings_available
    
    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is None and self.embeddings_available:
            from sentence_transformers import SentenceTransformer
            
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
            except Exception as e:
                logger.warning(f"Failed to load {self.model_name}, trying fallback: {e}")
                try:
                    self._model = SentenceTransformer(FALLBACK_MODEL)
                except Exception as e2:
                    logger.error(f"Failed to load fallback model: {e2}")
                    self._embeddings_available = False
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for a text."""
        return hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()
    
    def _load_cache(self):
        """Load embedding cache from disk."""
        if not self.use_cache:
            return
            
        cache_file = self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self._embedding_cache = {
                        k: np.array(v) for k, v in data.items()
                    }
                logger.debug(f"Loaded {len(self._embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        if not self.use_cache:
            return
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.json"
        
        try:
            with open(cache_file, 'w') as f:
                data = {k: v.tolist() for k, v in self._embedding_cache.items()}
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not self.embeddings_available:
            # Fallback: use simple character n-gram vectors
            return self._fallback_embed(texts)
        
        self._load_model()
        if self._model is None:
            return self._fallback_embed(texts)
        
        # Check cache for already-embedded texts
        results = []
        texts_to_embed = []
        text_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                results.append((i, self._embedding_cache[cache_key]))
            else:
                texts_to_embed.append(text)
                text_indices.append(i)
        
        # Embed uncached texts
        if texts_to_embed:
            new_embeddings = self._model.encode(
                texts_to_embed,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            
            # Cache new embeddings
            for idx, (text, embedding) in enumerate(zip(texts_to_embed, new_embeddings)):
                cache_key = self._get_cache_key(text)
                self._embedding_cache[cache_key] = embedding
                results.append((text_indices[idx], embedding))
        
        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results])
    
    def _fallback_embed(self, texts: Sequence[str]) -> np.ndarray:
        """
        Fallback embedding using character n-grams (TF-IDF style).
        
        Returns a simple bag-of-character-ngrams representation.
        """
        # Build vocabulary from all texts
        ngram_size = 3
        vocab: dict[str, int] = {}
        
        for text in texts:
            text_lower = text.lower().replace('_', ' ').replace('-', ' ')
            for i in range(len(text_lower) - ngram_size + 1):
                ngram = text_lower[i:i + ngram_size]
                if ngram not in vocab:
                    vocab[ngram] = len(vocab)
        
        # Create vectors
        vectors = []
        for text in texts:
            text_lower = text.lower().replace('_', ' ').replace('-', ' ')
            vec = np.zeros(max(len(vocab), 1))
            
            for i in range(len(text_lower) - ngram_size + 1):
                ngram = text_lower[i:i + ngram_size]
                if ngram in vocab:
                    vec[vocab[ngram]] += 1
            
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            
            vectors.append(vec)
        
        return np.array(vectors)
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def rank_properties(
        self,
        column_header: str,
        properties: Sequence[PropertyInfo],
        top_k: int = 5,
    ) -> list[SimilarityResult]:
        """
        Rank ontology properties by semantic similarity to column header.
        
        Args:
            column_header: CSV column name to match
            properties: List of ontology properties to rank
            top_k: Number of top results to return
            
        Returns:
            List of SimilarityResult objects, sorted by score descending
        """
        if not properties:
            return []
        
        # Prepare texts for embedding
        # Column header variations
        column_variations = [
            column_header,
            column_header.replace('_', ' '),
            column_header.replace('-', ' '),
        ]
        
        # Property texts (label, aliases, description)
        property_texts = []
        property_mapping = []  # (property_idx, text_type)
        
        for i, prop in enumerate(properties):
            # Label
            property_texts.append(prop.label)
            property_mapping.append((i, "label"))
            
            # Aliases
            for alias in prop.aliases:
                property_texts.append(alias)
                property_mapping.append((i, "alias"))
            
            # Description (if available)
            if prop.description:
                property_texts.append(prop.description[:200])  # Truncate long descriptions
                property_mapping.append((i, "description"))
        
        # Embed all texts
        all_texts = column_variations + property_texts
        embeddings = self.embed(all_texts)
        
        col_embeddings = embeddings[:len(column_variations)]
        prop_embeddings = embeddings[len(column_variations):]
        
        # Average column embeddings
        col_embedding = np.mean(col_embeddings, axis=0)
        
        # Score each property (best match across label/aliases/description)
        property_scores: dict[int, tuple[float, str]] = {}
        
        for idx, (prop_idx, text_type) in enumerate(property_mapping):
            score = self.cosine_similarity(col_embedding, prop_embeddings[idx])
            
            if prop_idx not in property_scores or score > property_scores[prop_idx][0]:
                property_scores[prop_idx] = (score, text_type)
        
        # Build results
        results = []
        for prop_idx, (score, match_type) in property_scores.items():
            prop = properties[prop_idx]
            
            # Determine confidence level
            if score >= 0.75:
                confidence = "high"
            elif score >= 0.5:
                confidence = "medium"
            else:
                confidence = "low"
            
            results.append(SimilarityResult(
                property_uri=prop.uri,
                label=prop.label,
                score=score,
                confidence=confidence,
                match_type=match_type,
            ))
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def batch_rank_properties(
        self,
        column_headers: Sequence[str],
        properties: Sequence[PropertyInfo],
        top_k: int = 5,
    ) -> dict[str, list[SimilarityResult]]:
        """
        Rank properties for multiple columns efficiently.
        
        Batches all column headers together for efficient embedding.
        
        Args:
            column_headers: List of column names
            properties: List of ontology properties
            top_k: Number of results per column
            
        Returns:
            Dict mapping column headers to their ranked results
        """
        if not column_headers or not properties:
            return {col: [] for col in column_headers}
        
        # Prepare all column variations
        col_texts = []
        col_indices = []
        for i, col in enumerate(column_headers):
            col_texts.extend([
                col,
                col.replace('_', ' '),
                col.replace('-', ' '),
            ])
            col_indices.extend([i, i, i])  # Track which column each variation belongs to
        
        # Prepare property texts
        property_texts = []
        property_mapping = []
        
        for i, prop in enumerate(properties):
            property_texts.append(prop.label)
            property_mapping.append((i, "label"))
            
            for alias in prop.aliases:
                property_texts.append(alias)
                property_mapping.append((i, "alias"))
            
            if prop.description:
                property_texts.append(prop.description[:200])
                property_mapping.append((i, "description"))
        
        # Embed everything in one batch
        all_texts = col_texts + property_texts
        embeddings = self.embed(all_texts)
        
        col_embeddings = embeddings[:len(col_texts)]
        prop_embeddings = embeddings[len(col_texts):]
        
        # Average embeddings per column
        col_avg_embeddings = {}
        for i, col in enumerate(column_headers):
            mask = [j for j, idx in enumerate(col_indices) if idx == i]
            col_avg_embeddings[col] = np.mean(col_embeddings[mask], axis=0)
        
        # Score and rank for each column
        results = {}
        for col in column_headers:
            col_emb = col_avg_embeddings[col]
            
            property_scores: dict[int, tuple[float, str]] = {}
            for idx, (prop_idx, text_type) in enumerate(property_mapping):
                score = self.cosine_similarity(col_emb, prop_embeddings[idx])
                
                if prop_idx not in property_scores or score > property_scores[prop_idx][0]:
                    property_scores[prop_idx] = (score, text_type)
            
            col_results = []
            for prop_idx, (score, match_type) in property_scores.items():
                prop = properties[prop_idx]
                
                if score >= 0.75:
                    confidence = "high"
                elif score >= 0.5:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                col_results.append(SimilarityResult(
                    property_uri=prop.uri,
                    label=prop.label,
                    score=score,
                    confidence=confidence,
                    match_type=match_type,
                ))
            
            col_results.sort(key=lambda x: x.score, reverse=True)
            results[col] = col_results[:top_k]
        
        return results
    
    def save_cache_to_disk(self):
        """Explicitly save cache to disk."""
        self._save_cache()


# Global singleton for efficient reuse
_global_embedder: Optional[EmbeddingService] = None


def get_embedder() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = EmbeddingService()
    return _global_embedder


def rank_column_to_properties(
    column_header: str,
    properties: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Convenience function to rank properties for a column.
    
    Args:
        column_header: Column name to match
        properties: List of property dicts with 'uri', 'label', 'aliases' keys
        top_k: Number of results to return
        
    Returns:
        List of result dicts with 'uri', 'label', 'score', 'confidence', 'match_type'
    """
    embedder = get_embedder()
    
    # Convert dicts to PropertyInfo objects
    prop_infos = [
        PropertyInfo(
            uri=p['uri'],
            label=p.get('label', p['uri'].split('/')[-1].split('#')[-1]),
            aliases=p.get('aliases', []),
            description=p.get('description'),
            domain=p.get('domain'),
            range=p.get('range'),
        )
        for p in properties
    ]
    
    results = embedder.rank_properties(column_header, prop_infos, top_k)
    
    return [
        {
            'uri': r.property_uri,
            'label': r.label,
            'score': round(r.score, 4),
            'confidence': r.confidence,
            'match_type': r.match_type,
        }
        for r in results
    ]


def batch_rank_columns_to_properties(
    column_headers: list[str],
    properties: list[dict],
    top_k: int = 5,
) -> dict[str, list[dict]]:
    """
    Convenience function to rank properties for multiple columns efficiently.
    
    Args:
        column_headers: List of column names
        properties: List of property dicts
        top_k: Number of results per column
        
    Returns:
        Dict mapping column names to result lists
    """
    embedder = get_embedder()
    
    prop_infos = [
        PropertyInfo(
            uri=p['uri'],
            label=p.get('label', p['uri'].split('/')[-1].split('#')[-1]),
            aliases=p.get('aliases', []),
            description=p.get('description'),
            domain=p.get('domain'),
            range=p.get('range'),
        )
        for p in properties
    ]
    
    results = embedder.batch_rank_properties(column_headers, prop_infos, top_k)
    
    return {
        col: [
            {
                'uri': r.property_uri,
                'label': r.label,
                'score': round(r.score, 4),
                'confidence': r.confidence,
                'match_type': r.match_type,
            }
            for r in col_results
        ]
        for col, col_results in results.items()
    }
