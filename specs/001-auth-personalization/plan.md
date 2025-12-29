# Implementation Plan: Authentication & Personalization System

**Branch**: `001-auth-personalization` | **Date**: 2025-12-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-auth-personalization/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement a stateless authentication and personalization system consisting of three components: (1) Node.js authentication server providing RS256 JWT-based signup/signin with mandatory software_background and hardware_background fields, (2) Docusaurus frontend integration with auth client and "Personalize for Me" button on chapter pages, and (3) Python FastAPI backend personalization service that verifies JWTs via JWKS, extracts user background from token claims, generates tailored content using Gemini LLM, and caches results in Neon PostgreSQL for 7 days.

## Technical Context

**Language/Version**:
- Node.js 20.x LTS (Auth server)
- Python 3.11+ (Backend personalization service)
- TypeScript 5.x (Docusaurus frontend)

**Primary Dependencies**:
- **Auth Server**: better-auth (core), @better-auth/jwt (RS256 + JWKS plugin), better-auth/adapters/neon (database adapter), hono (web framework)
- **Backend**: FastAPI, PyJWT[crypto] (or PyJWT + cryptography), httpx (JWKS fetching), asyncpg, google-generativeai (Gemini SDK)
- **Frontend**: Docusaurus 3.x, React 18, native fetch API

**Storage**: Neon PostgreSQL (serverless) with two schemas:
- Authentication DB: users table (auth server writes, backend reads for verification context)
- Personalization DB: personalized_content table (backend writes/reads for caching)

**Testing**:
- **Auth Server**: Jest + Supertest (API testing)
- **Backend**: pytest + pytest-asyncio + pytest-cov + httpx.AsyncClient (FastAPI testing)
- **Frontend**: Jest + React Testing Library

**Target Platform**:
- Auth Server: Railway (Node.js deployment)
- Backend: Railway (Python deployment)
- Frontend: Vercel (Docusaurus static site with API proxy)

**Project Type**: Web application (Option 2) with three services: auth-server (Node.js), backend (Python FastAPI), frontend (Docusaurus)

**Performance Goals**:
- Signup: <3 seconds server response time
- Signin: <1 second JWT generation and return
- JWT verification: <100ms (cached JWKS public key)
- Personalization: <5 seconds end-to-end (including LLM generation)
- 100 concurrent personalization requests without degradation

**Constraints**:
- JWT access token: 1 hour expiration (per constitution)
- JWT refresh token: 30 days expiration (per constitution)
- Personalized content cache: 7 days TTL (per constitution)
- JWKS public key cache: 24 hours (per constitution)
- Rate limiting: 10 personalization requests per user per minute
- No PII (software_background/hardware_background) in logs (per constitution)

**Scale/Scope**:
- Initial: 100 concurrent users
- Target: 1,000 registered users with 10,000 personalization requests per day
- Cache hit rate goal: 60% (reduces LLM API costs)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Security (NON-NEGOTIABLE)
- ✅ **PASS**: JWT authentication using RS256 (asymmetric signing)
- ✅ **PASS**: Passwords hashed with bcrypt (cost factor 12 specified in FR-003)
- ✅ **PASS**: Secrets stored in environment variables (JWT private key, database credentials, Gemini API key)
- ✅ **PASS**: HTTPS/TLS required for all production endpoints (FR-027)
- ✅ **PASS**: Input validation on signup/signin (FR-002: enum validation for background fields)
- ✅ **PASS**: SQL injection prevention via parameterized queries (using pg/asyncpg with prepared statements)
- ✅ **PASS**: Generic error messages for auth failures (FR-008)
- ✅ **PASS**: CSRF protection not required (stateless JWT, no cookies for auth in primary flow)

### Principle II: Test-Driven Development (NON-NEGOTIABLE)
- ✅ **PASS**: pytest for Python backend (specified in constitution)
- ✅ **PASS**: Jest for Node.js auth server and Docusaurus frontend (specified in constitution)
- ✅ **PASS**: 80% coverage requirement (to be enforced in tasks phase)
- ✅ **PASS**: Integration tests for API endpoints (required by TDD principle)
- ✅ **PASS**: Mock external services (Gemini API, JWKS endpoint) in tests

### Principle III: User Experience
- ✅ **PASS**: Responsive design for signup/signin forms (mobile-first)
- ✅ **PASS**: Accessible UI (WCAG 2.1 AA compliance for forms and buttons)
- ✅ **PASS**: User-friendly error messages (FR-008 for auth, acceptance scenario 5 in User Story 3 for personalization errors)
- ✅ **PASS**: Loading states during personalization (implied in FR-014)
- ✅ **PASS**: Performance targets align with constitution (<1.5s FCP, <3.5s TTI)

### Principle IV: Gemini Usage
- ✅ **PASS**: Gemini `gemini-2.0-flash-exp` for personalized content generation (FR-020)
- ⚠️ **CLARIFICATION NEEDED**: Constitution specifies "OpenAI Agent SDK configured for Gemini" but user input mentions "OpenAI (Prompt: ...)" - Need to confirm if using OpenAI SDK with Gemini endpoint or native Gemini SDK
- ✅ **PASS**: Streaming not required for personalization feature (modal/inline display, not chat)
- ✅ **PASS**: Error handling for LLM failures (User Story 3, Scenario 5)

### Principle V: Modularity
- ✅ **PASS**: Auth Specialist handles Node.js auth server implementation
- ✅ **PASS**: Backend Specialist handles Python FastAPI personalization service
- ✅ **PASS**: Frontend Specialist handles Docusaurus integration
- ✅ **PASS**: Clear domain boundaries (auth server doesn't do personalization, backend doesn't do auth token issuance)

### Principle VI: Human Approval
- ✅ **PASS**: Production deployment requires human approval (per constitution)
- ✅ **PASS**: Database schema migrations (users, personalized_content tables) require human review before production

### Principle VII: Identity & Personalization (NON-NEGOTIABLE)
- ✅ **PASS**: RS256 JWT tokens with user_id, email, software_background, hardware_background claims (FR-005)
- ✅ **PASS**: JWKS endpoint at `/.well-known/jwks.json` (FR-006)
- ✅ **PASS**: Mandatory software_background and hardware_background at signup (FR-001, FR-002)
- ✅ **PASS**: Enum validation for background fields (FR-002, constitution specifies exact enum values)
- ✅ **PASS**: NOT NULL constraints in database (FR-003)
- ✅ **PASS**: Personalized content cached with 7-day expiration (FR-021)
- ✅ **PASS**: No PII logging (FR-025: software_background/hardware_background never logged)
- ✅ **PASS**: JWKS public key cached for 24 hours (FR-017)
- ✅ **PASS**: Token expiration: 1 hour access, 30 days refresh (FR-004, FR-005)

**Gate Status (Initial)**: ⚠️ **CONDITIONAL PASS** - One clarification needed on Gemini SDK choice (OpenAI Agent SDK vs native Gemini SDK). Proceeding to Phase 0 research to resolve.

**Gate Status (After Phase 1 Design)**: ✅ **FULL PASS** - All constitutional requirements satisfied after technology research:

- **Principle IV (Gemini Usage)**: Resolved via research.md. Using `google-generativeai` (native Gemini Python SDK) which aligns with constitution's intent and existing backend patterns. The existing backend already uses this SDK in `backend/app/api/chat.py`, ensuring consistency.
- **All Other Principles**: No changes from initial assessment. All gates remain green.

## Project Structure

### Documentation (this feature)

```text
specs/001-auth-personalization/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (SDK choice, auth framework, deployment)
├── data-model.md        # Phase 1 output (users, personalized_content schemas)
├── quickstart.md        # Phase 1 output (local dev setup guide)
├── contracts/           # Phase 1 output (OpenAPI specs)
│   ├── auth-api.yaml    # Node.js auth server endpoints
│   └── personalization-api.yaml  # Python backend endpoints
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

This feature spans three existing services in the monorepo:

```text
auth-server/              # NEW SERVICE (to be created)
├── src/
│   ├── routes/
│   │   ├── signup.ts    # POST /api/auth/signup
│   │   ├── signin.ts    # POST /api/auth/signin
│   │   ├── refresh.ts   # POST /api/auth/refresh
│   │   └── jwks.ts      # GET /.well-known/jwks.json
│   ├── middleware/
│   │   ├── validation.ts   # Input validation for signup/signin
│   │   └── errorHandler.ts # Global error handler
│   ├── services/
│   │   ├── authService.ts  # Business logic (password hashing, JWT generation)
│   │   └── dbService.ts    # Database operations (user CRUD)
│   ├── utils/
│   │   ├── jwt.ts          # JWT signing/verification utilities
│   │   └── jwks.ts         # JWKS generation from RSA key pair
│   └── server.ts           # Express/Fastify app entry point
├── tests/
│   ├── integration/
│   │   ├── signup.test.ts
│   │   ├── signin.test.ts
│   │   └── jwks.test.ts
│   └── unit/
│       └── authService.test.ts
├── package.json
└── tsconfig.json

backend/                  # EXISTING SERVICE (extend)
├── app/
│   ├── api/
│   │   └── personalize.py     # NEW: POST /api/personalize endpoint
│   ├── services/
│   │   ├── jwt_service.py     # NEW: JWT verification via JWKS
│   │   ├── personalization_service.py  # NEW: LLM content generation
│   │   └── cache_service.py   # NEW: Personalized content caching
│   ├── models/
│   │   └── personalized_content.py  # NEW: SQLAlchemy model
│   └── middleware/
│       └── auth_middleware.py  # NEW: JWT verification dependency
├── tests/
│   ├── test_personalize_api.py       # NEW: Integration tests
│   ├── test_jwt_service.py           # NEW: Unit tests
│   └── test_personalization_service.py  # NEW: Unit tests (mock Gemini)
└── requirements.txt      # UPDATE: Add python-jose, httpx

frontend/                 # EXISTING SERVICE (extend)
├── src/
│   ├── components/
│   │   ├── auth/                 # NEW: Auth components
│   │   │   ├── SignupForm.tsx
│   │   │   ├── SigninForm.tsx
│   │   │   └── AuthContext.tsx   # React context for auth state
│   │   └── PersonalizeButton.tsx # NEW: Personalize for Me button
│   ├── services/
│   │   ├── authService.ts        # NEW: API calls to auth server
│   │   └── personalizationService.ts  # NEW: API calls to backend
│   ├── hooks/
│   │   └── useAuth.ts            # NEW: Auth state management hook
│   └── theme/
│       └── AuthModal.tsx         # NEW: Modal for personalized content
├── tests/
│   ├── components/
│   │   ├── SignupForm.test.tsx
│   │   ├── SigninForm.test.tsx
│   │   └── PersonalizeButton.test.tsx
│   └── services/
│       └── authService.test.ts
└── docusaurus.config.js  # UPDATE: Add auth server proxy config
```

**Structure Decision**: Selected Option 2 (Web application) with modifications. The repository already contains `backend/` (Python FastAPI) and `frontend/` (Docusaurus). This feature adds a new `auth-server/` directory for the standalone Node.js authentication service, following the principle of service separation. The auth server is deployed independently to enable cross-service JWT verification via JWKS, as specified in Constitution Principle VII and the user's architecture flow.

## Complexity Tracking

> **No violations - table not needed**

All constitution gates pass (one SDK clarification to be resolved in Phase 0). The three-service architecture (auth-server, backend, frontend) aligns with the existing repository structure and the requirement for stateless JWT authentication with JWKS public key distribution.

---

## Phase 0: Research & Technology Decisions

### Unknowns to Resolve

1. **NEEDS CLARIFICATION**: Gemini SDK choice
   - Constitution Principle IV specifies "OpenAI Agent SDK configured for Gemini endpoints"
   - User input mentions "OpenAI (Prompt: ...)"
   - Options: (a) OpenAI SDK with Gemini-compatible endpoint, (b) Native `google-generativeai` Python SDK
   - **Research needed**: Verify which SDK approach aligns with constitution and existing backend patterns

2. **Auth framework for Node.js server**
   - Options: Express.js, Fastify, better-auth (batteries-included framework)
   - **Decision**: better-auth + Hono (documented in research.md R2)
   - Rationale: Built-in user management, JWT plugin with RS256/JWKS, custom fields support, reduces implementation complexity from ~77 tasks to ~30 tasks

3. **JWT verification library for Python backend**
   - Options: python-jose, PyJWT
   - **Decision**: PyJWT + cryptography (documented in research.md R3)
   - Rationale: Simple API, native PyJWKClient for JWKS fetching/caching, most popular Python JWT library

4. **Token storage in Docusaurus frontend**
   - Constitution mentions "HTTP-only cookies preferred, or secure localStorage with XSS mitigations" (FR-010)
   - Options: (a) HTTP-only cookies, (b) localStorage with XSS protections
   - **Research needed**: Docusaurus SSR constraints, auth server CORS configuration, security trade-offs

5. **Database schema for personalized content caching**
   - Content identifier: SHA-256 hash of chapter markdown (Assumption 5 in spec)
   - **Research needed**: Index strategy for fast cache lookups, JSONB structure for content_payload

6. **Deployment configuration**
   - Auth server: Railway deployment for Node.js (needs package.json, Procfile, environment vars)
   - Backend: Existing Railway setup (extend with new endpoints)
   - Frontend: Vercel with API proxy to auth server (needs vercel.json rewrite rules)
   - **Research needed**: CORS configuration for cross-origin requests (Docusaurus → Auth server, Docusaurus → Backend)

### Research Tasks

Documented in `research.md` (COMPLETED):

1. **Task R1**: ✅ Research Gemini SDK options → **Decision**: `google-generativeai` (native Python SDK)
2. **Task R2**: ✅ Evaluate auth frameworks → **Decision**: better-auth + Hono (batteries-included, JWT plugin, custom fields)
3. **Task R3**: ✅ Compare JWT verification libraries → **Decision**: PyJWT + cryptography (PyJWKClient for JWKS)
4. **Task R4**: ✅ Design token storage strategy → **Decision**: localStorage with CSP mitigations
5. **Task R5**: ✅ Design database schema and indexing → **Decision**: Composite unique index (user_id, content_hash, content_type)
6. **Task R6**: ✅ Document CORS configuration → **Decision**: Direct CORS requests (specific origins, no wildcards)

---

*Phase 0 output (`research.md`) has documented all technology decisions. All research tasks are complete.*
