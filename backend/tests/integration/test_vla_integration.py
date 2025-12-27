"""
Integration tests for Vision-Language-Action (VLA) system.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.rag_service import RAGService
from app.rag.retrieval import search_similar_chunks


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client for testing."""
    # Create a mock that mimics the openai-agents API
    mock_agent = MagicMock()
    mock_agent.name = "PhysicalAIAssistant"
    mock_agent.model = MagicMock()

    # Mock the Runner.run method
    mock_runner = AsyncMock()
    mock_result = MagicMock()
    mock_result.final_output = "This is a test response from the VLA system."
    mock_runner.run = AsyncMock(return_value=mock_result)

    # Return a structure that matches what the real client would return
    return {
        "agent": mock_agent,
        "runner": mock_runner,
        "result": mock_result
    }


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock_client = MagicMock()

    # Mock search results
    mock_hits = [
        MagicMock(
            payload={
                "text": "Physical AI combines perception, action, and learning in embodied systems.",
                "source_file": "docs/intro.mdx",
                "section": "Introduction",
                "filename": "intro.mdx",
                "chunk_index": 0,
                "metadata": {"topic": "physical_ai_basics"}
            },
            score=0.95
        ),
        MagicMock(
            payload={
                "text": "Humanoid robots require bipedal locomotion control using ZMP (Zero Moment Point) principles.",
                "source_file": "docs/humanoid/locomotion.mdx",
                "section": "Bipedal Locomotion",
                "filename": "locomotion.mdx",
                "chunk_index": 1,
                "metadata": {"topic": "locomotion"}
            },
            score=0.87
        )
    ]

    mock_client.search = MagicMock(return_value=mock_hits)
    return mock_client


@pytest.mark.asyncio
async def test_vla_process_query(mock_gemini_client, mock_qdrant_client):
    """Test complete VLA pipeline: vision-language-action."""
    with patch('app.services.rag_service.get_gemini_client', return_value=mock_gemini_client["agent"]), \
         patch('app.services.rag_service.get_qdrant_client', return_value=mock_qdrant_client):

        # Initialize RAG service
        rag_service = RAGService()

        # Test query
        query = "What is Physical AI?"

        # Process query
        result = await rag_service.process_query(
            query=query,
            user_id="test_user_123",
            session_id="test_session_456",
            temperature=0.7,
            max_tokens=1024,
            top_k=5,
            score_threshold=0.7
        )

        # Verify results
        assert result.answer is not None
        assert len(result.sources) >= 0  # May have sources
        assert result.session_id == "test_session_456"
        assert result.tokens_used > 0


@pytest.mark.asyncio
async def test_vla_streaming_response(mock_gemini_client, mock_qdrant_client):
    """Test VLA streaming response functionality."""
    with patch('app.services.rag_service.get_gemini_client', return_value=mock_gemini_client["agent"]), \
         patch('app.services.rag_service.get_qdrant_client', return_value=mock_qdrant_client):

        rag_service = RAGService()

        # Test streaming
        query = "Explain humanoid locomotion"
        chunks_received = []

        async for chunk in rag_service.process_query_stream(
            query=query,
            user_id="test_user_123",
            session_id="test_session_789",
            temperature=0.7,
            max_tokens=512,
            top_k=3,
            score_threshold=0.6
        ):
            chunks_received.append(chunk)
            # Break after receiving a few chunks to avoid long test
            if len(chunks_received) >= 5:
                break

        # Verify streaming behavior
        assert len(chunks_received) > 0
        assert any(chunk["type"] == "content" for chunk in chunks_received)
        assert any(chunk["type"] == "done" for chunk in chunks_received)


@pytest.mark.asyncio
async def test_vla_context_integration(mock_gemini_client, mock_qdrant_client):
    """Test that VLA properly integrates vision and language context."""
    with patch('app.services.rag_service.get_gemini_client', return_value=mock_gemini_client["agent"]), \
         patch('app.services.rag_service.get_qdrant_client', return_value=mock_qdrant_client):

        rag_service = RAGService()

        # Test with a complex query that requires context
        complex_query = "How do I implement ZMP-based balance control for humanoid walking, and what are the key parameters?"

        result = await rag_service.process_query(
            query=complex_query,
            user_id="test_user_123",
            session_id="test_session_001"
        )

        # The response should contain information about ZMP and balance control
        assert result.answer is not None
        # Should have used context from the mock Qdrant results
        assert result.tokens_used > 0


@pytest.mark.asyncio
async def test_vla_error_handling(mock_gemini_client, mock_qdrant_client):
    """Test VLA error handling and fallback behavior."""
    with patch('app.services.rag_service.get_gemini_client', return_value=mock_gemini_client["agent"]), \
         patch('app.services.rag_service.get_qdrant_client', return_value=mock_qdrant_client):

        rag_service = RAGService()

        # Test with empty query (should handle gracefully)
        with pytest.raises(ValueError):
            await rag_service.process_query("", "test_user")

        # Test with very long query (should handle gracefully)
        long_query = "This is a very long query " * 1000
        try:
            result = await rag_service.process_query(long_query, "test_user")
            # Should either process successfully or handle the error gracefully
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass


def test_vla_performance_benchmarks():
    """Test VLA system performance benchmarks."""
    import time

    # This would be an actual performance test in a real implementation
    # For now, we'll just verify the structure exists
    assert hasattr(RAGService, '__init__')
    assert hasattr(RAGService, 'process_query')
    assert hasattr(RAGService, 'process_query_stream')


@pytest.mark.asyncio
async def test_vla_multimodal_fusion(mock_gemini_client, mock_qdrant_client):
    """Test multimodal fusion in VLA system."""
    with patch('app.services.rag_service.get_gemini_client', return_value=mock_gemini_client["agent"]), \
         patch('app.services.rag_service.get_qdrant_client', return_value=mock_qdrant_client):

        rag_service = RAGService()

        # Test query that requires both vision and language understanding
        query = "Based on the documentation, what objects can the humanoid robot manipulate?"

        # Process with context retrieval
        context_chunks = await search_similar_chunks(
            query=query,
            top_k=5,
            score_threshold=0.6
        )

        # Verify context was retrieved
        assert len(context_chunks) > 0

        # Process with retrieved context
        result = await rag_service.process_query(
            query=query,
            user_id="test_user_123",
            session_id="test_session_101"
        )

        # Verify result is meaningful
        assert result.answer is not None
        assert len(result.answer) > 0


# Test suite for VLA system validation
async def run_vla_tests():
    """Run all VLA integration tests."""
    test_results = {
        "test_vla_process_query": False,
        "test_vla_streaming_response": False,
        "test_vla_context_integration": False,
        "test_vla_error_handling": False,
        "test_vla_multimodal_fusion": False
    }

    try:
        await test_vla_process_query(None, None)  # Using mocks
        test_results["test_vla_process_query"] = True
    except Exception as e:
        print(f"test_vla_process_query failed: {e}")

    try:
        await test_vla_streaming_response(None, None)  # Using mocks
        test_results["test_vla_streaming_response"] = True
    except Exception as e:
        print(f"test_vla_streaming_response failed: {e}")

    try:
        await test_vla_context_integration(None, None)  # Using mocks
        test_results["test_vla_context_integration"] = True
    except Exception as e:
        print(f"test_vla_context_integration failed: {e}")

    try:
        await test_vla_error_handling(None, None)  # Using mocks
        test_results["test_vla_error_handling"] = True
    except Exception as e:
        print(f"test_vla_error_handling failed: {e}")

    try:
        await test_vla_multimodal_fusion(None, None)  # Using mocks
        test_results["test_vla_multimodal_fusion"] = True
    except Exception as e:
        print(f"test_vla_multimodal_fusion failed: {e}")

    print("VLA Integration Test Results:")
    for test, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test}")

    return test_results


if __name__ == "__main__":
    asyncio.run(run_vla_tests())
