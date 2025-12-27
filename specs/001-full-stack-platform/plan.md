# Implementation Plan: Physical AI & Humanoid Robotics Platform

**Branch**: `001-full-stack-platform` | **Date**: 2025-12-25 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-full-stack-platform/spec.md`

## Summary

Build a full-stack educational platform for Physical AI and Humanoid Robotics consisting of:
- **Frontend**: Docusaurus-based documentation site with TypeScript, React, MDX content, dark mode, responsive design, and embedded chatbot UI
- **Backend**: FastAPI async API with RAG pipeline, authentication, streaming chat endpoint, structured logging
- **AI/ML**: Gemini embeddings (text-embedding-004) and generation (gemini-2.0-flash-exp) via OpenAI Agent SDK
- **Data Layer**: Neon PostgreSQL for user/session data, Qdrant vector database for semantic search
- **Auth**: Custom FastAPI JWT implementation with HTTP-only cookies, bcrypt password hashing (cost 12)
- **Deployment**: Vercel (frontend), Railway (backend), GitHub Actions CI/CD with PR previews

**Primary Technical Approach**: Modular architecture with 8 specialist agents (PM, Frontend, Backend, RAG, Auth, Test, Deploy, Content) implementing features in 8 phases from planning through human-approved production deployment.

## Technical Context

**Language/Version**:
- Frontend: TypeScript 5.6.2, Node.js 20+, React 19.0, Docusaurus 3.9.2
- Backend: Python 3.11+

**Primary Dependencies**:
- Frontend: @docusaurus/core 3.9.2, @mdx-js/react, react-dom, clsx, prism-react-renderer
- Backend: fastapi, uvicorn, gunicorn, sqlalchemy[asyncio], asyncpg, pydantic[email], python-jose, passlib[bcrypt], qdrant-client, openai (configured for Gemini), python-dotenv, slowapi

**Storage**:
- Relational: Neon PostgreSQL with tables `users` (id, email, hashed_password, created_at), `chat_sessions` (id, user_id, history JSONB, created_at, updated_at)
- Vector: Qdrant collection `physical_ai_docs` with 768-dimensional embeddings, metadata (source file, section, topic, chunk text)
- Static: Docusaurus MDX files in `docs/` directory

**Testing**:
- Frontend: Jest + React Testing Library, >80% coverage
- Backend: pytest + pytest-asyncio + pytest-cov, >80% coverage
- Integration: End-to-end API tests with mocked Gemini/Qdrant
- CI: GitHub Actions runs tests before deployment

**Target Platform**:
- Frontend: Browser (Chrome, Firefox, Safari, Edge last 2 years), mobile-first responsive (320px to 4K)
- Backend: Linux server (Railway), Python 3.11 runtime

**Project Type**: Web application (monorepo with frontend + backend separation)

**Performance Goals**:
- Frontend: FCP <1.5s, TTI <3.5s, page load <3s on 3G
- Backend: Non-streaming API <500ms, Qdrant search <200ms, auth <100ms
- Chat: First token <2s, full response <10s, handle 100 concurrent users

**Constraints**:
- Security: OWASP Top 10 compliance, no secrets in code/logs, HTTPS only, rate limiting (10 req/min chat, 5 failed logins per 15min)
- TDD: Red-Green-Refactor mandatory, tests before code, no merge without passing tests
- Accessibility: WCAG 2.1 AA (semantic HTML, ARIA, keyboard nav, screen reader support)
- Modularity: 8 agents with strict domain boundaries, no cross-domain code modifications

**Scale/Scope**:
- Users: <1000 concurrent (initial launch), scalable to 10k+
- Content: 40+ documentation pages across 5 modules (ROS 2, Gazebo/Unity, NVIDIA Isaac, VLA, Capstone)
- Vector DB: 10,000 document chunks initially, indexed for fast retrieval

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Security (NON-NEGOTIABLE) - ✅ PASS

- [x] Authentication endpoints use custom FastAPI JWT implementation with HTTP-only cookies
- [x] All secrets in environment variables (GEMINI_API_KEY, DATABASE_URL, JWT_SECRET, QDRANT_URL)
- [x] Passwords hashed with bcrypt cost factor 12
- [x] HTTPS/TLS required in production (Vercel/Railway enforce)
- [x] Input validation with Pydantic models prevents injection
- [x] SQL injection prevented via SQLAlchemy parameterized queries
- [x] XSS prevention through React's automatic escaping and CSP headers
- [x] CSRF protection via SameSite=Strict cookies
- [x] Rate limiting on chat (10/min) and auth (5 failed/15min) endpoints

### II. Test-Driven Development (NON-NEGOTIABLE) - ✅ PASS

- [x] Backend: pytest + pytest-asyncio + pytest-cov configured
- [x] Frontend: Jest + React Testing Library configured
- [x] Red-Green-Refactor enforced in all phases
- [x] Unit tests for business logic (RAG pipeline, auth flows, chat endpoint)
- [x] Integration tests for API endpoints with mocked externals (Gemini, Qdrant)
- [x] Coverage threshold: 80% minimum (enforced in CI)
- [x] Tests run in GitHub Actions before deployment
- [x] No code merged without passing tests

### III. User Experience - ✅ PASS

- [x] WCAG 2.1 AA compliance (semantic HTML, ARIA labels, keyboard nav)
- [x] Mobile-first responsive design (320px to 4K)
- [x] Real-time streaming for chat responses (SSE)
- [x] Performance targets: FCP <1.5s, TTI <3.5s
- [x] User-friendly error messages with actionable guidance
- [x] Interactive code examples with syntax highlighting and copy-to-clipboard
- [x] Dark mode respects user's preferred color scheme (Docusaurus built-in)

### IV. Gemini Usage - ✅ PASS

- [x] Embeddings: text-embedding-004 (768 dimensions)
- [x] Generation: gemini-2.0-flash-exp for RAG responses
- [x] Streaming: SSE for chat responses
- [x] SDK: OpenAI SDK configured for Gemini endpoints
- [x] Rate limiting: Exponential backoff and request queuing
- [x] Error handling: Graceful degradation when API unavailable
- [x] Prompt engineering: System prompts emphasize accuracy for Physical AI content
- [x] Citation: RAG responses cite source documents

### V. Modularity - ✅ PASS

- [x] Project Manager: Coordinates workflows, manages dependencies, ensures constitution compliance
- [x] Frontend Specialist: Docusaurus config, React components, UI/UX, client-side routing
- [x] Backend Specialist: FastAPI endpoints, Pydantic models, async operations, business logic
- [x] RAG Specialist: Document ingestion, embeddings, Qdrant operations, prompt augmentation
- [x] Auth Specialist: Authentication flows, JWT management, session handling, security
- [x] Test Specialist: Test generation, mocking, coverage analysis, CI/CD integration
- [x] Deployment Specialist: Vercel/Railway config, env vars, PR previews
- [x] Content Generation Specialist: Educational content, robotics tutorials, curriculum design

### VI. Human Approval - ✅ PASS

- [x] Production deployment requires human review of deployment checklist
- [x] Database migrations require human verification before production apply
- [x] Bulk data ingestion requires human approval before production indexing
- [x] Secret rotation requires human verification
- [x] Breaking changes require human review of impact analysis and migration plan

**Constitution Compliance**: ALL GATES PASSED ✅

## Project Structure

### Documentation (this feature)

```text
specs/001-full-stack-platform/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 - Technical unknowns resolved
├── data-model.md        # Phase 1 - Database schemas and vector structures
├── quickstart.md        # Phase 1 - Developer setup guide
├── contracts/           # Phase 1 - OpenAPI specs
│   ├── chat-api.yaml
│   ├── auth-api.yaml
│   └── sessions-api.yaml
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Phase 2 (/sp.tasks - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
# FRONTEND (Docusaurus site - repository root)
src/
├── components/
│   ├── HomepageFeatures/    # Existing
│   ├── ChatBot/             # NEW: Chat interface
│   │   ├── ChatBot.tsx
│   │   ├── MessageList.tsx
│   │   ├── ChatInput.tsx
│   │   └── styles.module.css
│   ├── Auth/                # NEW: Authentication
│   │   ├── SignUpForm.tsx
│   │   ├── SignInForm.tsx
│   │   └── AuthContext.tsx
│   └── Layout/
│       └── DarkModeToggle.tsx
├── pages/
│   ├── index.tsx
│   └── markdown-page.md
├── css/
│   └── custom.css
└── utils/                   # NEW
    └── api.ts               # API client

docs/                        # Educational content
├── intro.md
├── ros2/                    # NEW
│   ├── nodes.mdx
│   ├── topics.mdx
│   └── labs/
├── simulation/              # NEW
│   ├── gazebo-basics.mdx
│   └── unity-ml-agents.mdx
├── isaac/                   # NEW
│   ├── isaac-sim.mdx
│   └── isaac-gym.mdx
├── vla/                     # NEW
│   └── vla-models.mdx
└── capstone/                # NEW
    └── project-guide.mdx

static/
├── img/
└── files/

# BACKEND (FastAPI)
backend/                     # NEW directory
├── app/
│   ├── main.py             # FastAPI entry point
│   ├── config.py           # Settings
│   ├── models/             # SQLAlchemy models
│   │   ├── user.py
│   │   └── chat_session.py
│   ├── schemas/            # Pydantic schemas
│   │   ├── auth.py
│   │   ├── chat.py
│   │   └── session.py
│   ├── api/                # Endpoints
│   │   ├── auth.py
│   │   ├── chat.py
│   │   └── sessions.py
│   ├── services/           # Business logic
│   │   ├── auth_service.py
│   │   ├── rag_service.py
│   │   └── chat_service.py
│   ├── db/
│   │   ├── database.py     # SQLAlchemy engine
│   │   └── migrations/     # Alembic
│   ├── rag/
│   │   ├── ingestion.py
│   │   ├── embeddings.py
│   │   ├── retrieval.py
│   │   └── prompts.py
│   └── middleware/
│       ├── auth_middleware.py
│       └── rate_limit.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── alembic.ini
├── requirements.txt
├── requirements-dev.txt
└── Procfile


.env.example
.gitignore
package.json
docusaurus.config.ts
sidebars.ts
tsconfig.json
README.md
```

**Structure Decision**: Web application structure with frontend (Docusaurus) in repository root and backend (FastAPI) in `backend/` subdirectory. Monorepo approach simplifies CI/CD and local development. Frontend deploys to Vercel (static site), backend deploys to Railway (containerized Python app).

## Complexity Tracking

> **No violations - all Constitution requirements satisfied**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| N/A       | N/A        | N/A                                  |

**Notes**: Project complexity is inherent to educational platform requirements (frontend + backend + RAG + auth). No artificial complexity introduced; all components necessary for feature requirements. Modularity principle satisfied through 8-agent structure.
