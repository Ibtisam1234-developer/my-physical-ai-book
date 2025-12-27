"""
Semantic retrieval system for RAG using Qdrant vector database.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from app.config import settings
from app.rag.embeddings import generate_embedding
from app.config import get_qdrant_client
import io


logger = logging.getLogger(__name__)


class VLARecommender:
    """
    Recommender system for VLA (Vision-Language-Action) based on similarity.
    """

    def __init__(self):
        self.qdrant_client = get_qdrant_client()

    async def find_related_content(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find related documentation content based on semantic similarity.

        Args:
            query: Query text to find similar content
            top_k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of similar content with metadata
        """
        # Generate query embedding
        query_embedding = await generate_embedding(query)

        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True
        )

        # Format results
        related_content = []
        for hit in search_results:
            content = {
                "text": hit.payload.get("text", ""),
                "source_file": hit.payload.get("source_file", ""),
                "section": hit.payload.get("section", ""),
                "file_name": hit.payload.get("file_name", ""),
                "score": hit.score,
                "metadata": hit.payload.get("metadata", {})
            }
            related_content.append(content)

        logger.info(f"Found {len(related_content)} related content items for query")
        return related_content


class ContextAugmenter:
    """
    Augments queries with relevant context from documentation.
    """

    def __init__(self):
        self.recommender = VLARecommender()

    async def augment_with_context(
        self,
        query: str,
        max_context_length: int = 3000
    ) -> Dict[str, Any]:
        """
        Augment query with relevant context from documentation.

        Args:
            query: Original query
            max_context_length: Maximum length of context to include

        Returns:
            Dictionary with augmented query and retrieved context
        """
        # Find related content
        related_content = await self.recommender.find_related_content(
            query, top_k=7, score_threshold=0.6
        )

        # Build context string
        context_parts = []
        total_length = 0

        for content in related_content:
            content_text = content["text"]
            if total_length + len(content_text) <= max_context_length:
                context_parts.append({
                    "text": content_text,
                    "source": content["source_file"],
                    "section": content["section"],
                    "score": content["score"]
                })
                total_length += len(content_text)
            else:
                # Add partial content if there's remaining space
                remaining_chars = max_context_length - total_length
                if remaining_chars > 0:
                    partial_text = content_text[:remaining_chars]
                    context_parts.append({
                        "text": partial_text,
                        "source": content["source_file"],
                        "section": content["section"],
                        "score": content["score"],
                        "partial": True
                    })
                break

        # Build augmented query
        if context_parts:
            context_text = "\n\n".join([part["text"] for part in context_parts])
            augmented_query = f"Context: {context_text}\n\nQuestion: {query}"
        else:
            augmented_query = query

        return {
            "augmented_query": augmented_query,
            "original_query": query,
            "context": context_parts,
            "context_length": total_length
        }


class MultiModalRetriever:
    """
    Retrieves information using multiple modalities (text, vision, robot state).
    """

    def __init__(self):
        self.text_retriever = TextRetriever()
        self.vision_retriever = VisionRetriever()
        self.fusion_retriever = FusionRetriever()

    async def retrieve_multimodal_context(
        self,
        text_query: str,
        visual_context: Optional[np.ndarray] = None,
        robot_state: Optional[Dict[str, Any]] = None,
        top_k: int = 7
    ) -> Dict[str, Any]:
        """
        Retrieve context using multiple modalities.

        Args:
            text_query: Natural language query
            visual_context: Optional visual context from robot cameras
            robot_state: Optional current robot state
            top_k: Number of results to return

        Returns:
            Dictionary with multimodal context
        """
        # Retrieve text-based context
        text_context = await self.text_retriever.retrieve(
            text_query, top_k=top_k
        )

        # Retrieve vision-based context if available
        vision_context = []
        if visual_context is not None:
            vision_context = await self.vision_retriever.retrieve(
                visual_context, top_k=top_k // 2
            )

        # Retrieve robot-state-based context if available
        robot_context = []
        if robot_state is not None:
            robot_context = await self._retrieve_robot_context(
                robot_state, top_k=top_k // 3
            )

        # Fuse contexts with relevance scoring
        fused_context = await self.fusion_retriever.fuse_contexts(
            text_context=text_context,
            vision_context=vision_context,
            robot_context=robot_context,
            query=text_query
        )

        return {
            "text_context": text_context,
            "vision_context": vision_context,
            "robot_context": robot_context,
            "fused_context": fused_context,
            "relevance_scores": self.calculate_relevance_scores(fused_context, text_query)
        }

    async def _retrieve_robot_context(
        self,
        robot_state: Dict[str, Any],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve context based on current robot state.
        """
        # Example: find documentation related to current robot capabilities
        capabilities = robot_state.get("capabilities", [])
        location = robot_state.get("location", "unknown")

        # Query documentation about current capabilities
        capability_query = f"Documentation about capabilities: {', '.join(capabilities)}"
        location_query = f"Documentation for location: {location}"

        # Retrieve relevant content
        capability_context = await self.text_retriever.retrieve(
            capability_query, top_k=top_k // 2
        )
        location_context = await self.text_retriever.retrieve(
            location_query, top_k=top_k // 2
        )

        return capability_context + location_context

    def calculate_relevance_scores(
        self,
        context: List[Dict[str, Any]],
        query: str
    ) -> List[float]:
        """
        Calculate relevance scores for retrieved context.
        """
        scores = []
        query_lower = query.lower()

        for ctx in context:
            text_lower = ctx["text"].lower()
            # Simple relevance calculation (in practice, use more sophisticated methods)
            relevance = self.calculate_text_relevance(text_lower, query_lower)
            scores.append(relevance)

        return scores

    def calculate_text_relevance(self, text: str, query: str) -> float:
        """
        Calculate relevance score between text and query.
        """
        query_words = set(query.split())
        text_words = set(text.split())

        if not query_words:
            return 0.0

        # Calculate overlap
        common_words = query_words.intersection(text_words)
        overlap_score = len(common_words) / len(query_words)

        # Calculate keyword presence
        keyword_score = sum(1 for word in query_words if word in text) / len(query_words)

        # Combine scores
        relevance = (overlap_score * 0.6 + keyword_score * 0.4)
        return min(relevance, 1.0)  # Cap at 1.0


class TextRetriever:
    """
    Text-based retrieval using semantic search.
    """

    def __init__(self):
        self.qdrant_client = get_qdrant_client()

    async def retrieve(
        self,
        query: str,
        top_k: int = 7,
        score_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant text chunks for query.

        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of relevant text chunks with metadata
        """
        # Generate query embedding
        query_embedding = await generate_embedding(query)

        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True
        )

        # Format results
        retrieved_chunks = []
        for hit in search_results:
            chunk = {
                "text": hit.payload.get("text", ""),
                "source_file": hit.payload.get("source_file", ""),
                "section": hit.payload.get("section", ""),
                "score": hit.score,
                "metadata": hit.payload.get("metadata", {}),
                "file_path": hit.payload.get("file_path", ""),
                "file_name": hit.payload.get("file_name", "")
            }
            retrieved_chunks.append(chunk)

        logger.info(f"Retrieved {len(retrieved_chunks)} text chunks for query")
        return retrieved_chunks


class VisionRetriever:
    """
    Vision-based retrieval using visual embeddings.
    """

    def __init__(self):
        self.qdrant_client = get_qdrant_client()
        self.vision_encoder = self.load_vision_encoder()

    def load_vision_encoder(self):
        """
        Load vision encoder for generating visual embeddings.
        """
        # This would load a vision model (e.g., CLIP visual encoder)
        # For now, using a placeholder
        class PlaceholderEncoder:
            def __call__(self, image):
                # Return a dummy embedding
                return [0.0] * 512  # Placeholder embedding size

        return PlaceholderEncoder()

    async def retrieve(
        self,
        visual_input: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve visual context based on image input.

        Args:
            visual_input: Image array
            top_k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of relevant visual context
        """
        try:
            # Generate visual embedding using the encoder
            # This would use the vision encoder to create embeddings
            if self.vision_encoder:
                visual_embedding = self.vision_encoder(visual_input)
            else:
                # Fallback to a dummy embedding
                visual_embedding = [0.0] * 512

            # Search in Qdrant for visual embeddings
            # For now, we'll search in the same collection as text
            # In a real implementation, you'd have a separate visual collection
            search_results = self.qdrant_client.search(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query_vector=visual_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True
            )

            # Format results
            retrieved_chunks = []
            for hit in search_results:
                chunk = {
                    "text": hit.payload.get("text", ""),
                    "source_file": hit.payload.get("source_file", ""),
                    "section": hit.payload.get("section", ""),
                    "score": hit.score,
                    "metadata": hit.payload.get("metadata", {}),
                    "file_path": hit.payload.get("file_path", ""),
                    "file_name": hit.payload.get("file_name", "")
                }
                retrieved_chunks.append(chunk)

            logger.info(f"Retrieved {len(retrieved_chunks)} visual context items")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Vision retrieval failed: {e}")
            # Return empty list as fallback
            return []


class FusionRetriever:
    """
    Fuses multiple retrieval modalities.
    """

    def __init__(self):
        self.qdrant_client = get_qdrant_client()

    async def fuse_contexts(
        self,
        text_context: List[Dict[str, Any]],
        vision_context: List[Dict[str, Any]],
        robot_context: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Fuse contexts from different modalities.

        Args:
            text_context: Retrieved text context
            vision_context: Retrieved vision context
            robot_context: Retrieved robot context
            query: Original query

        Returns:
            Fused context list sorted by relevance
        """
        # Combine all contexts
        all_contexts = []

        # Add text contexts with source identifier
        for ctx in text_context:
            ctx["source_modality"] = "text"
            all_contexts.append(ctx)

        # Add vision contexts with source identifier
        for ctx in vision_context:
            ctx["source_modality"] = "vision"
            all_contexts.append(ctx)

        # Add robot contexts with source identifier
        for ctx in robot_context:
            ctx["source_modality"] = "robot"
            all_contexts.append(ctx)

        # Re-rank combined contexts based on query relevance
        reranked_contexts = await self.rerank_contexts(all_contexts, query)

        # Return top results
        return reranked_contexts[:10]  # Return top 10 fused contexts

    async def rerank_contexts(
        self,
        contexts: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Re-rank contexts based on query relevance using cross-encoder.
        """
        # In practice, this would use a cross-encoder model
        # For now, using similarity-based re-ranking

        # Calculate relevance scores
        scored_contexts = []
        query_embedding = await generate_embedding(query)

        for ctx in contexts:
            # Generate embedding for context text
            ctx_embedding = await generate_embedding(ctx["text"])

            # Calculate similarity using the cosine_similarity function from embeddings
            from app.rag.embeddings import cosine_similarity
            similarity = cosine_similarity(query_embedding, ctx_embedding)
            ctx["reranked_score"] = similarity
            scored_contexts.append(ctx)

        # Sort by reranked score (descending)
        scored_contexts.sort(key=lambda x: x["reranked_score"], reverse=True)

        return scored_contexts



async def search_similar_chunks(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Search for similar text chunks based on semantic similarity.

    Args:
        query: Query text to find similar chunks
        top_k: Number of similar chunks to return
        score_threshold: Minimum similarity score threshold

    Returns:
        List of similar chunks with metadata
    """
    text_retriever = TextRetriever()
    return await text_retriever.retrieve(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold
    )


# Global instances for singleton pattern
text_retriever = TextRetriever()
vision_retriever = VisionRetriever()
fusion_retriever = FusionRetriever()
context_augmenter = ContextAugmenter()
vla_recommender = VLARecommender()
multimodal_retriever = MultiModalRetriever()


# Example usage function
async def example_retrieval():
    """
    Example of using the retrieval system.
    """
    query = "How do I implement PID control for humanoid balance?"

    # Retrieve text context
    text_results = await text_retriever.retrieve(query, top_k=5)
    print(f"Text retrieval found {len(text_results)} results")

    # Augment query with context
    augmented = await context_augmenter.augment_with_context(query)
    print(f"Augmented query length: {len(augmented['augmented_query'])}")

    # Find related content
    related = await vla_recommender.find_related_content(query)
    print(f"Found {len(related)} related content items")

    return text_results, augmented, related


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_retrieval())
