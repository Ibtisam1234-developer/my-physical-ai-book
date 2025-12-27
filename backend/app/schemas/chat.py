"""
Pydantic schemas for chat functionality.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    """User roles for authorization."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SourceCitation(BaseModel):
    """Schema for source citations in responses."""
    source: str = Field(..., description="Source document or URL")
    section: str = Field(..., description="Section or chapter name")
    filename: str = Field(..., description="Source file name")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    text_snippet: str = Field(..., description="Relevant text snippet from source")

    @validator('relevance_score')
    def validate_relevance_score(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Relevance score must be between 0.0 and 1.0')
        return v


class ChatMessage(BaseModel):
    """Schema for individual chat messages."""
    role: UserRole = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., min_length=1, max_length=10000, description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    sources: Optional[List[SourceCitation]] = Field(default=[], description="Source citations for assistant messages")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional message metadata")

    @validator('content')
    def validate_content_length(cls, v):
        if len(v) > 10000:
            raise ValueError('Message content must be less than 10,000 characters')
        return v


class ChatRequest(BaseModel):
    """Schema for chat requests."""
    query: str = Field(..., min_length=1, max_length=10000, description="User query")
    session_id: Optional[str] = Field(None, description="Existing session ID (creates new if not provided)")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Response creativity (0.0-2.0)")
    max_tokens: int = Field(1024, ge=1, le=4096, description="Maximum response tokens")
    stream: bool = Field(True, description="Whether to stream response")
    top_k: int = Field(7, ge=1, le=20, description="Number of context chunks to retrieve")
    score_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum relevance score for retrieval")
    user_preferences: Optional[Dict[str, Any]] = Field(default={}, description="User preferences for response")

    @validator('query')
    def validate_query_length(cls, v):
        if len(v) > 10000:
            raise ValueError('Query must be less than 10,000 characters')
        return v


class ChatResponse(BaseModel):
    """Schema for chat responses."""
    response: str = Field(..., description="Generated response")
    sources: List[SourceCitation] = Field(default=[], description="Source citations")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    tokens_used: int = Field(..., description="Number of tokens in response")
    response_time_ms: float = Field(..., description="Response generation time in milliseconds")


class ChatSession(BaseModel):
    """Schema for chat session management."""
    id: str = Field(..., description="Session identifier")
    title: str = Field(..., min_length=1, max_length=200, description="Session title")
    user_id: Optional[str] = Field(None, description="Associated user ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last activity time")
    message_count: int = Field(0, description="Number of messages in session")
    is_active: bool = Field(True, description="Whether session is active")
    metadata: Dict[str, Any] = Field(default={}, description="Session metadata")


class ChatHistoryResponse(BaseModel):
    """Schema for chat history responses."""
    session_id: str = Field(..., description="Session identifier")
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    created_at: datetime = Field(..., description="Session creation time")
    updated_at: datetime = Field(..., description="Last activity time")


class StreamChunk(BaseModel):
    """Schema for streaming response chunks."""
    type: str = Field(..., description="Chunk type: content, sources, done, error")  # "content", "sources", "done", "error"
    content: Optional[str] = Field(None, description="Content chunk (for type=content)")
    sources: Optional[List[SourceCitation]] = Field(None, description="Sources (for type=sources)")
    error: Optional[str] = Field(None, description="Error message (for type=error)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Chunk timestamp")


class DeleteChatSessionRequest(BaseModel):
    """Schema for deleting chat sessions."""
    session_id: str = Field(..., description="ID of session to delete")


class ListChatSessionsResponse(BaseModel):
    """Schema for listing chat sessions."""
    sessions: List[ChatSession] = Field(..., description="List of user sessions")
    total_count: int = Field(..., description="Total number of sessions")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(20, description="Number of sessions per page")


# Example usage validation
def validate_chat_request_example():
    """Example of schema validation."""
    try:
        # Valid request
        valid_request = ChatRequest(
            query="What is Physical AI?",
            session_id="session_123",
            temperature=0.7,
            max_tokens=512,
            stream=True
        )
        print("Valid request created:", valid_request.query)

        # Invalid request (too long)
        try:
            invalid_request = ChatRequest(
                query="x" * 15000,  # Too long
                temperature=0.7
            )
        except ValidationError as e:
            print("Validation error caught:", e)

    except ImportError:
        print("Pydantic not available for validation example")


if __name__ == "__main__":
    # Test schema validation
    from pydantic import ValidationError
    validate_chat_request_example()
