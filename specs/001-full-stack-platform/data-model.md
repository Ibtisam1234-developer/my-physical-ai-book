# Phase 1: Data Model Design

**Feature**: 001-full-stack-platform | **Date**: 2025-12-25 | **Status**: Complete

## Overview

Data structures for the Physical AI & Humanoid Robotics Platform: relational database schemas (PostgreSQL), vector database structures (Qdrant), and Pydantic models.

---

## Relational Database (Neon PostgreSQL)

### Schema: `users`

| Column          | Type                      | Constraints                 | Description                          |
|-----------------|---------------------------|-----------------------------|--------------------------------------|
| `id`            | `SERIAL`                  | `PRIMARY KEY`               | Unique user identifier              |
| `email`         | `VARCHAR(255)`            | `UNIQUE`, `NOT NULL`        | User's email (login identifier)     |
| `hashed_password` | `VARCHAR(255)`          | `NOT NULL`                  | bcrypt hash (cost 12)               |
| `created_at`    | `TIMESTAMP WITH TIME ZONE`| `DEFAULT NOW()`, `NOT NULL` | Account creation timestamp          |

**Indexes**: PRIMARY KEY on `id`, UNIQUE on `email`

**SQLAlchemy Model**:
```python
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
```

---

### Schema: `chat_sessions`

| Column       | Type                      | Constraints                              | Description                     |
|--------------|---------------------------|------------------------------------------|---------------------------------|
| `id`         | `SERIAL`                  | `PRIMARY KEY`                            | Unique session identifier      |
| `user_id`    | `INTEGER`                 | `REFERENCES users(id) ON DELETE CASCADE` | Foreign key to user            |
| `history`    | `JSONB`                   | `DEFAULT '[]'`, `NOT NULL`               | Array of chat messages         |
| `created_at` | `TIMESTAMP WITH TIME ZONE`| `DEFAULT NOW()`                          | Session creation timestamp     |
| `updated_at` | `TIMESTAMP WITH TIME ZONE`| `DEFAULT NOW()`                          | Last message timestamp         |

**Indexes**: PRIMARY KEY on `id`, INDEX on `user_id`, INDEX on `updated_at DESC`

**JSONB `history` Structure**:
```json
[
  {
    "role": "user",
    "content": "What is Physical AI?",
    "timestamp": "2025-12-25T10:35:00Z"
  },
  {
    "role": "assistant",
    "content": "Physical AI refers to...",
    "timestamp": "2025-12-25T10:35:03Z",
    "sources": [{"file": "docs/intro.mdx", "section": "Introduction"}]
  }
]
```

**SQLAlchemy Model**:
```python
from sqlalchemy import Column, Integer, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    history = Column(JSONB, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), index=True)

    user = relationship("User", backref="chat_sessions")
```

---

## Vector Database (Qdrant)

### Collection: `physical_ai_docs`

**Configuration**:
```python
from qdrant_client.models import Distance, VectorParams

collection_config = {
    "collection_name": "physical_ai_docs",
    "vectors_config": VectorParams(size=768, distance=Distance.COSINE)
}
```

**Payload Schema**:
```json
{
  "text": "ROS 2 nodes are independent processes...",
  "source_file": "docs/ros2/nodes.mdx",
  "section": "What Are Nodes?",
  "topic": "ROS 2",
  "module": "ros2",
  "chunk_index": 0,
  "char_count": 1420
}
```

---

## Pydantic Models

### Authentication

**SignUpRequest**:
```python
from pydantic import BaseModel, EmailStr, Field

class SignUpRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
```

**TokenResponse**:
```python
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: UserResponse
```

---

### Chat

**ChatRequest**:
```python
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[int] = None
```

**ChatMessage**:
```python
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    sources: Optional[List[SourceCitation]] = None
```

---

## Validation Rules

- **Email**: Valid format, unique, case-insensitive
- **Password**: 8-100 characters, bcrypt cost 12
- **Chat Query**: 1-10,000 characters
- **Session History**: Maximum 1000 messages per session

---

## Security Considerations

- SQL injection: SQLAlchemy parameterized queries
- XSS: React auto-escaping, CSP headers
- JWT: HTTP-only cookies, SameSite=Strict, 24-hour expiration
- Rate limiting: slowapi (per-IP or per-user)
