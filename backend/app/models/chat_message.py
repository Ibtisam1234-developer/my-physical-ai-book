"""
Chat message model for storing conversation history.
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base import Base


class ChatMessage(Base):
    """
    Chat message model for storing conversation history.
    """

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)  # Source citations (JSON array)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSON, nullable=True)  # Additional message metadata

    # Relationship to session
    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, session_id={self.session_id}, role='{self.role}')>"


# Add relationship to ChatSession model (already defined in chat_session.py)
# ChatSession.messages = relationship("ChatMessage", back_populates="session")