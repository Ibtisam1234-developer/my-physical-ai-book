"""
RAG (Retrieval-Augmented Generation) service for Physical AI platform.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import time
import uuid
from pathlib import Path

from app.config import settings
from app.config import get_gemini_client
from app.config import get_qdrant_client
from app.rag.embeddings import generate_embedding, generate_embeddings_batch
from app.rag.prompts import build_rag_prompt, SYSTEM_PROMPT
from app.rag.retrieval import search_similar_chunks


@dataclass
class RAGResult:
    """Result of RAG pipeline execution."""
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    tokens_used: int
    response_time: float


class RAGService:
    """
    Complete RAG service integrating retrieval and generation.
    """

    def __init__(self):
        self.gemini_model = get_gemini_client()  # Google Generative AI model
        self.qdrant_client = get_qdrant_client()
        self.logger = logging.getLogger(__name__)

    async def process_query(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_k: int = 7,
        score_threshold: float = 0.7
    ) -> RAGResult:
        """
        Process a query through the RAG pipeline.

        Args:
            query: User's natural language query
            user_id: User identifier for session management
            session_id: Session identifier (creates new if not provided)
            temperature: Generation temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            top_k: Number of context chunks to retrieve
            score_threshold: Minimum relevance score for retrieval

        Returns:
            RAGResult with answer, sources, and metadata
        """
        start_time = time.time()

        # Create or validate session
        if not session_id:
            session_id = f"session_{uuid.uuid4()}"

        # Retrieve relevant context
        context_chunks = await self.retrieve_context(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )

        # Build RAG prompt
        rag_prompt = build_rag_prompt(query, context_chunks)

        # Generate response using the Google Generative AI model
        import google.generativeai as genai
        from app.config import get_gemini_client

        # Prepare the content for the model
        full_prompt = f"{SYSTEM_PROMPT}\n\nContext: {rag_prompt}"

        # Generate content using the model (sync call wrapped in async)
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.gemini_model.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
        )

        answer = response.text if response and response.text else ""
        tokens_used = len(answer.split()) if answer else 0

        # Create result
        result = RAGResult(
            answer=answer,
            sources=[
                {
                    'source': chunk.get('source_file', ''),
                    'section': chunk.get('section', ''),
                    'filename': chunk.get('file_name', ''),
                    'relevance_score': chunk.get('score', 0.0),
                    'text_snippet': chunk.get('text', '')[:200]  # Add required field for schema validation
                }
                for chunk in context_chunks
            ],
            session_id=session_id,
            tokens_used=tokens_used,
            response_time=time.time() - start_time
        )

        # Log interaction
        self.logger.info(
            "RAG query processed",
            extra={
                "user_id": user_id,
                "session_id": session_id,
                "query_length": len(query),
                "response_time": result.response_time,
                "tokens_used": tokens_used,
                "sources_count": len(context_chunks)
            }
        )

        return result

    async def process_query_stream(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_k: int = 7,
        score_threshold: float = 0.7
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process query with streaming response using Server-Sent Events.

        Args:
            query: User's natural language query
            user_id: User identifier
            session_id: Session identifier
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
            top_k: Number of context chunks to retrieve
            score_threshold: Minimum relevance score

        Yields:
            Stream chunks with content, sources, and completion status
        """
        start_time = time.time()

        if not session_id:
            session_id = f"session_{uuid.uuid4()}"

        # Retrieve context
        context_chunks = await self.retrieve_context(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )

        # Send sources first
        if context_chunks:
            yield {
                "type": "sources",
                "sources": [
                    {
                        "source": chunk.get("source_file", ""),
                        "section": chunk.get("section", ""),
                        "filename": chunk.get("file_name", ""),
                        "relevance_score": chunk.get("score", 0.0),
                        "text_snippet": chunk.get("text", "")[:200]  # Required for schema validation
                    }
                    for chunk in context_chunks
                ],
                "timestamp": time.time()
            }

        # Build RAG prompt
        rag_prompt = build_rag_prompt(query, context_chunks)

        # Stream response using Google Generative AI streaming
        try:
            import google.generativeai as genai

            # Prepare the content for the model
            full_prompt = f"{SYSTEM_PROMPT}\n\nContext: {rag_prompt}"

            # Create streaming response using the model
            response_generator = self.gemini_model.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
                stream=True
            )

            full_response = ""

            # Stream content as it arrives
            for chunk in response_generator:
                if chunk.text:
                    content = chunk.text
                    full_response += content

                    yield {
                        "type": "content",
                        "content": content,
                        "timestamp": time.time()
                    }

            # Send completion signal
            yield {
                "type": "done",
                "session_id": session_id,
                "response_time": time.time() - start_time,
                "tokens_used": len(full_response.split()),
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "timestamp": time.time()
            }

    async def retrieve_context(
        self,
        query: str,
        top_k: int = 7,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from vector database.

        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score

        Returns:
            List of relevant document chunks
        """
        # Use the search_similar_chunks function from retrieval module
        context_chunks = await search_similar_chunks(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )

        self.logger.info(
            f"Retrieved {len(context_chunks)} context chunks for query",
            extra={"query_length": len(query), "chunks_retrieved": len(context_chunks)}
        )

        return context_chunks

    async def store_document(
        self,
        content: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a document in the vector database.

        Args:
            content: Document content to store
            source: Source identifier
            metadata: Additional metadata

        Returns:
            Document ID
        """
        doc_id = f"doc_{uuid.uuid4()}"

        # Chunk document
        from app.rag.ingestion import chunk_text
        chunks = chunk_text(content)

        # Generate embeddings for chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = await generate_embeddings_batch(chunk_texts)

        # Prepare points for Qdrant
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = f"{doc_id}_chunk_{i}"
            point = {
                "id": point_id,
                "vector": embedding,
                "payload": {
                    "text": chunk['text'],
                    "source": source,
                    "section": chunk.get('section', ''),
                    "filename": Path(source).name,
                    "chunk_index": i,
                    "metadata": metadata or {}
                }
            }
            points.append(point)

        # Upsert to Qdrant
        self.qdrant_client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points
        )

        self.logger.info(
            f"Stored document {doc_id} with {len(points)} chunks",
            extra={"source": source, "chunks": len(points)}
        )

        return doc_id

    async def get_session_history(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """
        Retrieve chat session history.

        Args:
            session_id: Session identifier
            user_id: User identifier for validation

        Returns:
            Dictionary with session history and metadata
        """
        # This would typically query a database
        # For now, returning a basic structure as placeholder
        return {
            "session_id": session_id,
            "user_id": user_id,
            "messages": [],
            "created_at": time.time(),
            "updated_at": time.time()
        }

    async def list_user_sessions(self, user_id: str, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """
        List all sessions for a user.

        Args:
            user_id: User identifier
            page: Page number
            page_size: Items per page

        Returns:
            Dictionary with sessions and pagination info
        """
        # This would typically query a database
        # For now, returning empty list as placeholder
        return {
            "sessions": [],
            "total_count": 0,
            "page": page,
            "page_size": page_size
        }

    async def delete_session(self, session_id: str, user_id: str) -> bool:
        """
        Delete a chat session.

        Args:
            session_id: Session identifier
            user_id: User identifier for validation

        Returns:
            True if deletion successful
        """
        # This would typically delete from database
        # For now, returning True as placeholder to indicate success
        # In a real implementation, this would connect to a database
        try:
            # In a real implementation, this would delete from database
            # For now we just simulate success
            return True
        except Exception:
            return False

    async def rename_session(self, session_id: str, user_id: str, new_title: str) -> bool:
        """
        Rename a chat session.

        Args:
            session_id: Session identifier
            user_id: User identifier for validation
            new_title: New session title

        Returns:
            True if rename successful
        """
        # This would typically update database
        # For now, returning True as placeholder to indicate success
        try:
            # In a real implementation, this would update the database
            # For now we just simulate success
            return True
        except Exception:
            return False


# Singleton instance
rag_service = RAGService()


# Example usage function
async def example_usage():
    """
    Example of using the RAG service.
    """
    # Example query
    query = "Explain the difference between ROS 2 and ROS 1 in the context of humanoid robotics"

    # Process query
    result = await rag_service.process_query(
        query=query,
        user_id="user_123",
        session_id="session_456",
        temperature=0.7,
        max_tokens=512,
        top_k=5,
        score_threshold=0.6
    )

    print(f"Answer: {result.answer}")
    print(f"Sources: {result.sources}")
    print(f"Response time: {result.response_time:.2f}s")

    # Example streaming
    print("\nStreaming response:")
    async for chunk in rag_service.process_query_stream(
        query="What is Physical AI and why is it important?",
        user_id="user_123"
    ):
        if chunk["type"] == "content":
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "sources":
            print(f"\n\nSources: {chunk['sources']}")
        elif chunk["type"] == "done":
            print(f"\n\nResponse completed in {chunk['response_time']:.2f}s")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
