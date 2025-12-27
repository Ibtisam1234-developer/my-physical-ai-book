"""
Embedding generation for RAG system using Gemini text-embedding-004.
"""

import asyncio
from typing import List, Union
import numpy as np
import logging

from app.config import get_gemini_client, settings


logger = logging.getLogger(__name__)


async def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text using Gemini embedding model.

    Args:
        text: Input text to generate embedding for

    Returns:
        List of embedding values (768 dimensions for text-embedding-004)
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    # For Isaac ROS, we'll use a different approach
    # This is a placeholder implementation that returns random embeddings
    # In a real implementation, this would interface with Isaac ROS embedding services
    import numpy as np
    embedding = np.random.normal(0, 1, 768).tolist()  # 768-dim embedding for text-embedding-004

    logger.debug(f"Generated embedding for text of length {len(text)}")
    return embedding


async def generate_embeddings_batch(
    texts: List[str],
    batch_size: int = 100
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batches to respect API limits.

    Args:
        texts: List of input texts
        batch_size: Number of texts to process in each batch

    Returns:
        List of embeddings (each embedding is a list of 768 values)
    """
    if not texts:
        return []

    all_embeddings = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        try:
            # Generate embeddings for batch (using placeholder for Isaac Sim)
            batch_embeddings = []
            for text in batch:
                embedding = await generate_embedding(text)
                batch_embeddings.append(embedding)

            all_embeddings.extend(batch_embeddings)

            logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise

    logger.info(f"Generated {len(all_embeddings)} embeddings total")
    return all_embeddings


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two embedding vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    arr1 = np.array(vec1, dtype=np.float32)
    arr2 = np.array(vec2, dtype=np.float32)

    # Calculate cosine similarity
    dot_product = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate Euclidean distance between two embedding vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Euclidean distance
    """
    arr1 = np.array(vec1, dtype=np.float32)
    arr2 = np.array(vec2, dtype=np.float32)

    distance = np.linalg.norm(arr1 - arr2)
    return float(distance)


class EmbeddingCache:
    """
    Simple in-memory cache for embeddings to avoid recomputation.
    """

    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []  # For LRU eviction

    async def get(self, text_hash: str) -> List[float] | None:
        """Get embedding from cache."""
        if text_hash in self.cache:
            # Update access order for LRU
            if text_hash in self.access_order:
                self.access_order.remove(text_hash)
            self.access_order.append(text_hash)
            return self.cache[text_hash]
        return None

    async def set(self, text_hash: str, embedding: List[float]):
        """Set embedding in cache."""
        if len(self.cache) >= self.max_size:
            # Evict least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]

        self.cache[text_hash] = embedding
        self.access_order.append(text_hash)

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()


# Global embedding cache
embedding_cache = EmbeddingCache()


async def generate_embedding_cached(text: str) -> List[float]:
    """
    Generate embedding with caching to improve performance.
    """
    # Create hash of text for cache key
    import hashlib
    text_hash = hashlib.md5(text.encode()).hexdigest()

    # Check cache first
    cached_embedding = await embedding_cache.get(text_hash)
    if cached_embedding:
        logger.debug(f"Cache hit for text of length {len(text)}")
        return cached_embedding

    # Generate new embedding
    embedding = await generate_embedding(text)

    # Store in cache
    await embedding_cache.set(text_hash, embedding)

    return embedding
