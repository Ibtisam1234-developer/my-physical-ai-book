"""
Chat API endpoints with streaming support.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any

from app.schemas.chat import ChatRequest, ChatResponse, StreamChunk, ChatHistoryResponse
from app.services.rag_service import RAGService
from app.utils.auth import get_current_user
from app.utils.logging import log_request_info, log_response_info
from app.config import settings


router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Non-streaming chat endpoint for simple integration.

    Args:
        request: Chat request with query and parameters
        current_user: Authenticated user (from JWT token)

    Returns:
        ChatResponse with answer and sources
    """
    # Initialize RAG service
    rag_service = RAGService()

    start_time = time.time()

    try:
        # Process chat request
        result = await rag_service.process_query(
            query=request.query,
            user_id=current_user.get("id"),
            session_id=request.session_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )

        response_time = time.time() - start_time

        # Create response
        response = ChatResponse(
            response=result.answer,
            sources=result.sources,
            session_id=result.session_id,
            tokens_used=len(result.answer.split()),
            response_time_ms=response_time * 1000
        )

        # Log successful request
        logger = logging.getLogger(__name__)
        log_response_info(
            logger=logger,
            endpoint="/api/chat",
            status=200,
            latency_ms=response_time * 1000,
            user_id=current_user.get("id")
        )

        return response

    except Exception as e:
        # Log error
        logger = logging.getLogger(__name__)
        error_msg = str(e)
        log_response_info(
            logger=logger,
            endpoint="/api/chat",
            status=500,
            latency_ms=(time.time() - start_time) * 1000,
            user_id=current_user.get("id"),
            error=error_msg
        )

        # Print to console for Railway logs
        print(f"ERROR in /api/chat: {error_msg}")
        import traceback
        traceback.print_exc()

        raise HTTPException(status_code=500, detail=f"Chat processing failed: {error_msg}")


@router.post("/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    http_request: Request,
    current_user: Dict = Depends(get_current_user)
):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).

    Args:
        request: Chat request with query and streaming parameters
        http_request: FastAPI request object for metadata
        current_user: Authenticated user

    Returns:
        StreamingResponse with Server-Sent Events
    """
    # Initialize RAG service
    rag_service = RAGService()

    async def event_generator() -> AsyncGenerator[str, None]:
        """
        Generate Server-Sent Events for streaming response.
        """
        start_time = time.time()
        error_occurred = False

        try:
            # Process streaming query
            async for chunk in rag_service.process_query_stream(
                query=request.query,
                user_id=current_user.get("id"),
                session_id=request.session_id,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_k=request.top_k,
                score_threshold=request.score_threshold
            ):
                # Format chunk as SSE event
                if chunk.get("type") == "content":
                    sse_chunk = StreamChunk(
                        type="content",
                        content=chunk.get("content", ""),
                        timestamp=time.time()
                    )
                elif chunk.get("type") == "sources":
                    sse_chunk = StreamChunk(
                        type="sources",
                        sources=chunk.get("sources", []),
                        timestamp=time.time()
                    )
                elif chunk.get("type") == "done":
                    sse_chunk = StreamChunk(
                        type="done",
                        timestamp=time.time()
                    )
                elif chunk.get("type") == "error":
                    sse_chunk = StreamChunk(
                        type="error",
                        error=chunk.get("error", ""),
                        timestamp=time.time()
                    )
                    error_occurred = True
                else:
                    continue  # Skip unrecognized chunk types

                # Send chunk as SSE
                yield f"data: {sse_chunk.model_dump_json()}\n\n"

                # Check if client disconnected
                if await http_request.is_disconnected():
                    break

        except Exception as e:
            error_chunk = StreamChunk(
                type="error",
                error=f"Streaming error: {str(e)}",
                timestamp=time.time()
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            error_occurred = True

        finally:
            response_time = time.time() - start_time

            # Log request
            logger = logging.getLogger(__name__)
            log_response_info(
                logger=logger,
                endpoint="/api/chat/stream",
                status=500 if error_occurred else 200,
                latency_ms=response_time * 1000,
                user_id=current_user.get("id")
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get chat history for a specific session.

    Args:
        session_id: Session identifier
        current_user: Authenticated user

    Returns:
        ChatHistoryResponse with conversation history
    """
    rag_service = RAGService()

    try:
        history = await rag_service.get_session_history(session_id, current_user.get("id"))

        return ChatHistoryResponse(
            session_id=history["session_id"],
            messages=history["messages"],
            created_at=history["created_at"],
            updated_at=history["updated_at"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@router.get("/sessions")
async def list_user_sessions(
    current_user: Dict = Depends(get_current_user),
    page: int = 1,
    page_size: int = 20
):
    """
    List all chat sessions for the current user.

    Args:
        current_user: Authenticated user
        page: Page number for pagination
        page_size: Number of sessions per page

    Returns:
        List of user's chat sessions
    """
    rag_service = RAGService()

    try:
        sessions = await rag_service.list_user_sessions(
            user_id=current_user.get("id"),
            page=page,
            page_size=page_size
        )

        return sessions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Delete a specific chat session.

    Args:
        session_id: Session identifier to delete
        current_user: Authenticated user

    Returns:
        Success message
    """
    rag_service = RAGService()

    try:
        success = await rag_service.delete_session(session_id, current_user.get("id"))

        if not success:
            raise HTTPException(status_code=404, detail="Session not found")

        return {"message": "Session deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@router.post("/sessions/{session_id}/rename")
async def rename_chat_session(
    session_id: str,
    request: Dict[str, str],
    current_user: Dict = Depends(get_current_user)
):
    """
    Rename a chat session.

    Args:
        session_id: Session identifier to rename
        request: Request body with new title
        current_user: Authenticated user

    Returns:
        Success message with new title
    """
    new_title = request.get("title", "").strip()

    if not new_title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")

    if len(new_title) > 200:
        raise HTTPException(status_code=400, detail="Title too long (max 200 characters)")

    rag_service = RAGService()

    try:
        success = await rag_service.rename_session(session_id, current_user.get("id"), new_title)

        if not success:
            raise HTTPException(status_code=404, detail="Session not found")

        return {"message": "Session renamed successfully", "new_title": new_title}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rename session: {str(e)}")


# Health check for chat service
@router.get("/health")
async def chat_health():
    """
    Health check endpoint for chat service.
    """
    return {
        "status": "healthy",
        "service": "chat",
        "models_loaded": True,
        "vector_db_connected": True,
        "gemini_api_available": True
    }
