---
name: neon-db-schemas
description: Neon PostgreSQL database schema patterns including user authentication tables, chat session storage with JSONB, SQLAlchemy async models, raw asyncpg usage, and index optimization. Use when designing database schemas, implementing user authentication, storing chat history, or working with Neon serverless Postgres.
tags: [neon, postgresql, sqlalchemy, asyncpg, database, schemas]
---

# Neon DB Schema Patterns

## Overview

Neon is a serverless PostgreSQL database optimized for modern applications. Key features:
- **Serverless**: Auto-scaling and pay-per-use
- **Branching**: Database branches for development/testing
- **Connection Pooling**: Built-in pooling for serverless environments
- **PostgreSQL Compatible**: Full PostgreSQL support with extensions

## Connection Setup

### Environment Variables
```bash
# .env
DATABASE_URL=postgresql://user:password@ep-xyz.neon.tech/dbname?sslmode=require
```

### SQLAlchemy Async Connection
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
import os

# Connection string for async
DATABASE_URL = os.getenv("DATABASE_URL")
# Convert postgres:// to postgresql+asyncpg://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Log SQL queries (disable in production)
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True  # Verify connections before using
)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()
```

### Raw asyncpg Connection
```python
import asyncpg
import os

async def get_db_pool():
    """Create asyncpg connection pool"""
    return await asyncpg.create_pool(
        os.getenv("DATABASE_URL"),
        min_size=1,
        max_size=10,
        command_timeout=60
    )

# Usage
pool = await get_db_pool()

async with pool.acquire() as conn:
    result = await conn.fetch("SELECT * FROM users")
```

## Core Database Schemas

### Users Table

**SQLAlchemy Model**:
```python
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"
```

**Raw SQL Schema**:
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for fast email lookup
CREATE UNIQUE INDEX idx_users_email ON users(email);
```

### Chat Sessions Table

**SQLAlchemy Model**:
```python
from sqlalchemy import Column, Integer, ForeignKey, TIMESTAMP, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    history = Column(JSONB, nullable=False, default=list)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())

    # Relationship
    user = relationship("User", backref="chat_sessions")

    def __repr__(self):
        return f"<ChatSession(id={self.id}, user_id={self.user_id})>"
```

**Raw SQL Schema**:
```sql
CREATE TABLE chat_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    history JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for fast user_id lookup
CREATE INDEX idx_chat_sessions_user_id ON chat_sessions(user_id);

-- GIN index for JSONB queries (optional, for complex filtering)
CREATE INDEX idx_chat_sessions_history ON chat_sessions USING GIN(history);
```

### JSONB History Structure
```json
[
    {
        "role": "user",
        "content": "What is Physical AI?",
        "timestamp": "2024-01-15T10:30:00Z"
    },
    {
        "role": "assistant",
        "content": "Physical AI combines perception, actuation, and learning...",
        "timestamp": "2024-01-15T10:30:05Z"
    }
]
```

## Database Operations

### Using SQLAlchemy Async

**Create Tables**:
```python
async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables created")
```

**Create User**:
```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def create_user(email: str, password: str) -> User:
    """Create new user with hashed password"""
    async with async_session_maker() as session:
        # Hash password
        hashed_password = pwd_context.hash(password)

        # Create user
        user = User(
            email=email,
            hashed_password=hashed_password
        )

        session.add(user)
        await session.commit()
        await session.refresh(user)

        return user
```

**Get User by Email**:
```python
from sqlalchemy import select

async def get_user_by_email(email: str) -> User | None:
    """Retrieve user by email"""
    async with async_session_maker() as session:
        result = await session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
```

**Verify Password**:
```python
async def authenticate_user(email: str, password: str) -> User | None:
    """Authenticate user with email and password"""
    user = await get_user_by_email(email)

    if not user:
        return None

    if not pwd_context.verify(password, user.hashed_password):
        return None

    return user
```

**Create Chat Session**:
```python
async def create_chat_session(user_id: int) -> ChatSession:
    """Create new chat session for user"""
    async with async_session_maker() as session:
        chat_session = ChatSession(
            user_id=user_id,
            history=[]
        )

        session.add(chat_session)
        await session.commit()
        await session.refresh(chat_session)

        return chat_session
```

**Append to Chat History**:
```python
from sqlalchemy import update

async def add_message_to_session(
    session_id: int,
    role: str,
    content: str
):
    """Add message to chat session history"""
    async with async_session_maker() as session:
        # Fetch current session
        result = await session.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        chat_session = result.scalar_one()

        # Append message
        new_message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }

        chat_session.history.append(new_message)

        # Mark as modified (required for JSONB)
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(chat_session, "history")

        await session.commit()
```

**Get User's Chat Sessions**:
```python
async def get_user_chat_sessions(user_id: int) -> list[ChatSession]:
    """Retrieve all chat sessions for a user"""
    async with async_session_maker() as session:
        result = await session.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.created_at.desc())
        )
        return result.scalars().all()
```

### Using Raw asyncpg

**Create User**:
```python
async def create_user_raw(pool: asyncpg.Pool, email: str, hashed_password: str):
    """Create user using raw asyncpg"""
    async with pool.acquire() as conn:
        user_id = await conn.fetchval(
            """
            INSERT INTO users (email, hashed_password)
            VALUES ($1, $2)
            RETURNING id
            """,
            email,
            hashed_password
        )
        return user_id
```

**Get User by Email**:
```python
async def get_user_by_email_raw(pool: asyncpg.Pool, email: str):
    """Get user by email using raw asyncpg"""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM users WHERE email = $1",
            email
        )
        return dict(row) if row else None
```

**Create Chat Session**:
```python
async def create_chat_session_raw(pool: asyncpg.Pool, user_id: int):
    """Create chat session using raw asyncpg"""
    async with pool.acquire() as conn:
        session_id = await conn.fetchval(
            """
            INSERT INTO chat_sessions (user_id, history)
            VALUES ($1, '[]'::jsonb)
            RETURNING id
            """,
            user_id
        )
        return session_id
```

**Append to Chat History (JSONB)**:
```python
import json

async def add_message_raw(
    pool: asyncpg.Pool,
    session_id: int,
    role: str,
    content: str
):
    """Add message to chat history using JSONB operations"""
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    }

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE chat_sessions
            SET history = history || $1::jsonb
            WHERE id = $2
            """,
            json.dumps([message]),
            session_id
        )
```

**Query JSONB**:
```python
async def get_recent_messages(pool: asyncpg.Pool, session_id: int, limit: int = 10):
    """Get recent messages from chat history"""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT jsonb_array_elements(history) AS message
            FROM chat_sessions
            WHERE id = $1
            ORDER BY (message->>'timestamp') DESC
            LIMIT $2
            """,
            session_id,
            limit
        )
        return row
```

## FastAPI Integration

### Dependency for Database Session
```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Database session dependency"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

### User Registration Endpoint
```python
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, EmailStr

app = FastAPI()

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    created_at: datetime

    class Config:
        from_attributes = True

@app.post("/users", response_model=UserResponse, status_code=201)
async def register_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register new user"""
    # Check if user exists
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create user
    hashed_password = pwd_context.hash(user_data.password)
    user = User(email=user_data.email, hashed_password=hashed_password)

    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user
```

### Chat Session Endpoints
```python
class ChatMessageCreate(BaseModel):
    role: str
    content: str

class ChatSessionResponse(BaseModel):
    id: int
    user_id: int
    history: list
    created_at: datetime

    class Config:
        from_attributes = True

@app.post("/chat/sessions", response_model=ChatSessionResponse, status_code=201)
async def create_session(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create new chat session"""
    session = ChatSession(user_id=current_user.id, history=[])
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session

@app.post("/chat/sessions/{session_id}/messages")
async def add_message(
    session_id: int,
    message: ChatMessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Add message to chat session"""
    # Get session
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.id == session_id)
        .where(ChatSession.user_id == current_user.id)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Append message
    new_message = {
        "role": message.role,
        "content": message.content,
        "timestamp": datetime.utcnow().isoformat()
    }
    session.history.append(new_message)

    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(session, "history")

    await db.commit()

    return {"message": "Message added", "history": session.history}

@app.get("/chat/sessions", response_model=list[ChatSessionResponse])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List user's chat sessions"""
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.created_at.desc())
    )
    return result.scalars().all()
```

## Index Optimization

### Creating Indexes
```sql
-- B-tree index on user_id (default, good for equality and range queries)
CREATE INDEX idx_chat_sessions_user_id ON chat_sessions(user_id);

-- Unique index on email
CREATE UNIQUE INDEX idx_users_email ON users(email);

-- GIN index for JSONB (enables efficient queries on JSON content)
CREATE INDEX idx_chat_sessions_history ON chat_sessions USING GIN(history);

-- Partial index (only index active users)
CREATE INDEX idx_active_users ON users(email) WHERE is_active = TRUE;

-- Multi-column index
CREATE INDEX idx_sessions_user_created ON chat_sessions(user_id, created_at DESC);
```

### JSONB Query Optimization
```sql
-- Query messages by role
SELECT * FROM chat_sessions
WHERE history @> '[{"role": "user"}]'::jsonb;

-- Check if history contains specific content
SELECT * FROM chat_sessions
WHERE history @> '[{"content": "Physical AI"}]'::jsonb;

-- Count messages in history
SELECT id, jsonb_array_length(history) AS message_count
FROM chat_sessions;
```

## Alembic Migrations

### Setup Alembic
```bash
pip install alembic
alembic init alembic
```

**alembic.ini** (edit):
```ini
sqlalchemy.url = postgresql+asyncpg://user:password@host/dbname
```

**alembic/env.py** (configure async):
```python
from sqlalchemy.ext.asyncio import create_async_engine
from alembic import context
import asyncio

# Import your models
from app.models import Base

config = context.config
target_metadata = Base.metadata

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = create_async_engine(
        config.get_main_option("sqlalchemy.url")
    )

    async def do_run_migrations(connection):
        await connection.run_sync(do_migrations)

    async def do_migrations(connection):
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

    asyncio.run(do_run_migrations(connectable))
```

### Create Migration
```bash
# Auto-generate migration
alembic revision --autogenerate -m "Create users and chat_sessions tables"

# Apply migration
alembic upgrade head
```

## Best Practices

### Schema Design
- **Use SERIAL or BIGSERIAL**: Auto-incrementing primary keys
- **UNIQUE constraints**: Enforce uniqueness at database level (email)
- **Foreign keys with CASCADE**: Automatic cleanup on user deletion
- **JSONB over JSON**: Better performance and indexing support
- **Timestamps**: Always include `created_at` and `updated_at`

### Indexing
- **Index foreign keys**: Always index `user_id` and other FK columns
- **Unique indexes**: On email, username for fast lookups
- **GIN indexes**: For JSONB when querying nested data
- **Avoid over-indexing**: Each index has write overhead

### Connection Management
- **Connection pooling**: Use SQLAlchemy pool or asyncpg pool
- **Pool size**: 5-10 connections for serverless
- **Pool pre-ping**: Verify connections before use (serverless idle timeout)

### JSONB Usage
- **flag_modified()**: Required for SQLAlchemy to detect JSONB changes
- **|| operator**: Append to JSONB arrays efficiently
- **GIN indexes**: Enable fast JSONB queries
- **Structure**: Keep consistent structure for easier querying

## Best Practices Checklist

- [ ] Use `SERIAL PRIMARY KEY` for auto-incrementing IDs
- [ ] Add `UNIQUE` constraint on email
- [ ] Create index on `user_id` in chat_sessions
- [ ] Use `JSONB` for flexible chat history storage
- [ ] Include `created_at` and `updated_at` timestamps
- [ ] Set up foreign key with `ON DELETE CASCADE`
- [ ] Configure connection pooling (SQLAlchemy or asyncpg)
- [ ] Use `flag_modified()` when updating JSONB with SQLAlchemy
- [ ] Hash passwords with bcrypt before storing
- [ ] Create GIN index on JSONB for complex queries
- [ ] Use Alembic for database migrations
- [ ] Enable `pool_pre_ping` for Neon serverless connections

---

**Usage Note**: Apply these patterns when working with Neon PostgreSQL databases. Use SQLAlchemy async for ORM convenience or asyncpg for raw performance. Always index foreign keys and use JSONB for flexible schema needs like chat history.
