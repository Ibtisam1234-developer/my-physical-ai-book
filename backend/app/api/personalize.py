"""
Personalization API endpoints for generating personalized content based on user background.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any
import hashlib
import time
from datetime import datetime, timedelta

from app.middleware.auth_middleware import get_current_user
from app.services.rag_service import RAGService
from app.config import settings

router = APIRouter(prefix="/api", tags=["Personalization"])


class PersonalizeRequest(BaseModel):
    chapterId: str
    chapterTitle: str
    softwareBackground: str
    hardwareBackground: str


class PersonalizeResponse(BaseModel):
    content: str
    cached: bool
    generated_at: str


@router.post("/personalize", response_model=PersonalizeResponse)
async def personalize_content(
    request: PersonalizeRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Generate personalized content for a chapter based on user's background.

    Args:
        request: Personalization request with chapter info and user background
        current_user: Authenticated user (from JWT token)

    Returns:
        Personalized content response
    """
    logger = logging.getLogger(__name__)

    start_time = time.time()

    try:
        # Create a hash of the chapter content for caching
        content_hash = hashlib.sha256(f"{request.chapterId}:{request.chapterTitle}".encode()).hexdigest()

        # Initialize RAG service
        rag_service = RAGService()

        # Generate personalized content using the new method
        personalized_content = await rag_service.generate_personalized_content(
            chapter_title=request.chapterTitle,
            software_background=request.softwareBackground,
            hardware_background=request.hardwareBackground,
            user_id=current_user.get("sub"),
            temperature=0.7,  # Slightly creative for personalization
            max_tokens=2000
        )

        response_time = time.time() - start_time

        # Log successful request
        logger.info(f"Personalization successful for user {current_user.get('sub')}, chapter {request.chapterId}")

        return PersonalizeResponse(
            content=personalized_content,
            cached=False,  # For now, always generating fresh
            generated_at=datetime.now().isoformat()
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Personalization failed for user {current_user.get('sub')}: {error_msg}")

        # Print to console for Railway logs
        print(f"ERROR in /api/personize: {error_msg}")
        import traceback
        traceback.print_exc()

        raise HTTPException(status_code=500, detail=f"Personalization failed: {error_msg}")


@router.get("/personalize/cache/{chapter_id}")
async def get_cached_personalization(
    chapter_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get cached personalized content for a chapter.

    Args:
        chapter_id: ID of the chapter
        current_user: Authenticated user

    Returns:
        Cached personalized content or 404 if not found
    """
    logger = logging.getLogger(__name__)

    try:
        # Create content hash
        content_hash = hashlib.sha256(chapter_id.encode()).hexdigest()

        # Check if we have cached content for this user + chapter + background
        cache_key = f"personalized_content:{current_user.get('sub')}:{content_hash}:{current_user.get('software_background')}:{current_user.get('hardware_background')}"

        # For now, return 404 since we don't have actual caching implemented
        # In a real implementation, we'd check the database/cache
        raise HTTPException(status_code=404, detail="Cached content not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache lookup failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Cache lookup failed")


# Health check for personalization service
@router.get("/personalize/health")
async def personalize_health():
    """
    Health check endpoint for personalization service.
    """
    return {
        "status": "healthy",
        "service": "personalization",
        "models_loaded": True,
        "gemini_api_available": True
    }