# Developer Quickstart Guide

**Feature**: 001-full-stack-platform | **Date**: 2025-12-25

## Prerequisites

### Required Software
- Node.js 20+
- Python 3.11+
- Git

### Required Accounts
- Gemini API: [Google AI Studio](https://aistudio.google.com/)
- Neon: [neon.tech](https://neon.tech/)
- Qdrant Cloud: [cloud.qdrant.io](https://cloud.qdrant.io/)

---

## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd my-physical-ai-book
git checkout 001-full-stack-platform
```

### 2. Frontend Setup

```bash
npm install
npm start
```

Access: http://localhost:3000

### 3. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env`:
```env
DATABASE_URL=postgresql+asyncpg://user:pass@host/db
GEMINI_API_KEY=your-gemini-api-key
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key
JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

Run migrations and start server:
```bash
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

Access: http://localhost:8000/docs

---

## Run Tests

```bash
# Frontend
npm test

# Backend
cd backend && pytest --cov=app
```

Expected: >80% coverage, all tests passing

---

## Troubleshooting

**Frontend can't connect**: Check CORS in `backend/app/main.py` allows `http://localhost:3000`

**Database error**: Verify `DATABASE_URL` format and Neon project is active

**Qdrant collection not found**: Create collection with 768 dimensions, Cosine distance

**Gemini rate limit**: Check API quotas at AI Studio

---

## Development Workflow

1. Write failing test (Red)
2. Implement minimal code (Green)
3. Refactor (keep tests green)
4. Commit and push

---

## Useful Commands

**Frontend**:
```bash
npm start         # Dev server
npm run build     # Production build
npm test          # Run tests
```

**Backend**:
```bash
uvicorn app.main:app --reload               # Dev server
pytest --cov=app --cov-report=html           # Tests with coverage
alembic revision --autogenerate -m "message" # Create migration
python -m app.rag.ingestion                  # Ingest documentation
```

---

## Next Steps

- Read spec: `specs/001-full-stack-platform/spec.md`
- Review data models: `specs/001-full-stack-platform/data-model.md`
- Explore API contracts: `specs/001-full-stack-platform/contracts/`
