# Phase 0: Research & Technical Unknowns

**Feature**: 001-full-stack-platform | **Date**: 2025-12-25 | **Status**: Complete

## Overview

This document resolves technical unknowns and validates technology choices for the Physical AI & Humanoid Robotics Platform.

## Research Questions & Findings

### 1. Gemini SDK Integration with OpenAI Agent SDK

**Question**: Can we use OpenAI Agent SDK with Gemini endpoints?

**Finding**: YES - Gemini API supports OpenAI-compatible endpoints via Google AI Studio.

**Implementation Pattern**:
```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Embeddings
response = client.embeddings.create(
    model="text-embedding-004",
    input="Your text here"
)

# Chat/Generation with streaming
stream = client.chat.completions.create(
    model="gemini-2.0-flash-exp",
    messages=[{"role": "user", "content": "Query"}],
    stream=True
)
```

**Decision**: Use OpenAI Python SDK with Gemini base URL. No custom SDK required.

---

### 2. Better Auth Integration

**Question**: How does Better Auth work with FastAPI backend?

**Finding**: Better Auth is JavaScript/TypeScript only (no Python support).

**Alternative**: Implement custom JWT authentication using python-jose, passlib[bcrypt].

**Implementation Pattern**:
```python
from jose import jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm="HS256")
```

**Decision**: Custom FastAPI JWT implementation with HTTP-only cookies (same security pattern as Better Auth).

---

### 3. Qdrant Integration

**Question**: Self-hosted vs. cloud Qdrant?

**Finding**: Qdrant Cloud provides free tier (1GB storage, sufficient for initial launch).

**Decision**: Use Qdrant Cloud free tier initially. Migrate to self-hosted if storage exceeds 1GB.

---

### 4. Streaming Response Pattern

**Question**: SSE vs. WebSockets for chat streaming?

**Finding**: Server-Sent Events (SSE) via FastAPI `StreamingResponse` is simpler for unidirectional streaming.

**Decision**: Use SSE. More efficient than WebSockets, better browser support, automatic reconnection.

---

### 5. Database Connection Pooling

**Question**: How to efficiently manage Neon connections in FastAPI?

**Finding**: Use SQLAlchemy async with asyncpg driver, configure pool_size=20 (Neon free tier limit).

**Decision**: SQLAlchemy async with pool_pre_ping=True to handle serverless cold starts.

---

### 6. Document Chunking Strategy

**Question**: How to chunk MDX for optimal RAG performance?

**Finding**: Hybrid approach - respect markdown headings but enforce 1500-character max with 200-character overlap.

**Decision**: Use RecursiveCharacterTextSplitter with markdown-aware separators.

---

### 7. Rate Limiting Implementation

**Question**: How to implement rate limiting in FastAPI?

**Finding**: slowapi library (Flask-Limiter port) works well without Redis dependency.

**Decision**: Use slowapi. Rates: 10 req/min for /api/chat, 5 failed attempts/15min for /api/auth/sign-in.

---

## Summary of Decisions

| Technology/Pattern | Decision | Rationale |
|-------------------|----------|-----------|
| Gemini Integration | OpenAI SDK with Gemini base_url | Official compatibility |
| Authentication | Custom FastAPI JWT + HTTP-only cookies | Better Auth is JS-only |
| Vector Database | Qdrant Cloud free tier | Managed service, 1GB sufficient |
| Streaming | Server-Sent Events (SSE) | Simpler than WebSockets |
| Database | SQLAlchemy async + asyncpg + Neon | Async-first, serverless optimized |
| Chunking | Hybrid (headings, 1500 char max, 200 overlap) | Context + precision balance |
| Rate Limiting | slowapi library | Simple, no Redis needed |
