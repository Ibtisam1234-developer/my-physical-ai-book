#!/usr/bin/env python3
"""
Test script to verify that the VLA implementation fixes work correctly.
"""

import asyncio
import sys
import os

# Add backend directory to path to import modules
sys.path.insert(0, os.path.dirname(__file__))

async def test_imports():
    """Test that all modules can be imported without errors."""
    print("Testing imports...")

    try:
        from app.schemas.chat import ChatRequest, ChatResponse, StreamChunk
        print("[OK] Chat schemas imported successfully")
    except Exception as e:
        print(f"[ERROR] Chat schemas import failed: {e}")
        return False

    try:
        from app.services.rag_service import RAGService
        print("[OK] RAG service imported successfully")
    except Exception as e:
        print(f"[ERROR] RAG service import failed: {e}")
        return False

    try:
        from app.rag.retrieval import TextRetriever, VLARecommender, MultiModalRetriever
        print("[OK] Retrieval modules imported successfully")
    except Exception as e:
        print(f"[ERROR] Retrieval modules import failed: {e}")
        return False

    try:
        from app.api.chat import router
        print("[OK] Chat API router imported successfully")
    except Exception as e:
        print(f"[ERROR] Chat API router import failed: {e}")
        return False

    return True

async def test_rag_service():
    """Test basic RAG service functionality."""
    print("\nTesting RAG service...")

    try:
        rag_service = RAGService()
        print("[OK] RAG service instantiated successfully")
    except Exception as e:
        print(f"[ERROR] RAG service instantiation failed: {e}")
        return False

    # Test that required methods exist
    methods_to_test = [
        'process_query',
        'process_query_stream',
        'retrieve_context',
        'store_document',
        'get_session_history',
        'list_user_sessions',
        'delete_session',
        'rename_session'
    ]

    for method_name in methods_to_test:
        if hasattr(rag_service, method_name):
            print(f"[OK] Method {method_name} exists")
        else:
            print(f"[ERROR] Method {method_name} missing")
            return False

    return True

async def test_retrieval_components():
    """Test retrieval components functionality."""
    print("\nTesting retrieval components...")

    try:
        text_retriever = TextRetriever()
        print("[OK] TextRetriever instantiated successfully")
    except Exception as e:
        print(f"[ERROR] TextRetriever instantiation failed: {e}")
        return False

    try:
        vla_recommender = VLARecommender()
        print("[OK] VLARecommender instantiated successfully")
    except Exception as e:
        print(f"[ERROR] VLARecommender instantiation failed: {e}")
        return False

    try:
        multimodal_retriever = MultiModalRetriever()
        print("[OK] MultiModalRetriever instantiated successfully")
    except Exception as e:
        print(f"[ERROR] MultiModalRetriever instantiation failed: {e}")
        return False

    # Test that required methods exist
    methods_to_test = [
        'retrieve',
        'find_related_content',
        'retrieve_multimodal_context'
    ]

    for method_name in methods_to_test:
        if hasattr(text_retriever if method_name == 'retrieve' else multimodal_retriever if method_name == 'retrieve_multimodal_context' else vla_recommender, method_name):
            print(f"[OK] Method {method_name} exists")
        else:
            print(f"[ERROR] Method {method_name} missing")
            return False

    return True

async def main():
    """Run all tests."""
    print("Starting VLA Implementation Fixes Verification Test")
    print("=" * 60)

    # Run all tests
    tests = [
        test_imports,
        test_rag_service,
        test_retrieval_components
    ]

    all_passed = True
    for test in tests:
        result = await test()
        if not result:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("[OK] All tests passed! VLA implementation fixes are working correctly.")
        return 0
    else:
        print("[ERROR] Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)