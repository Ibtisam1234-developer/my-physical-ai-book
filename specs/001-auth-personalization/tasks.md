---

description: "Task list for Authentication & Personalization System"
---

# Tasks: Authentication & Personalization System

**Input**: Design documents from `/specs/001-auth-personalization/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Tests are MANDATORY per Constitution Principle II (TDD). All tasks include test requirements following red-green-refactor cycle.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Auth Server**: `auth-server/` (NEW service at repository root)
- **Backend**: `backend/` (EXISTING service, extend)
- **Frontend**: `frontend/` (EXISTING service, extend)
- **Database Migrations**: `specs/001-auth-personalization/migrations/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure for new auth-server service

- [ ] T001 Create auth-server directory structure at repository root with src/, tests/, and keys/ folders
- [ ] T002 Initialize auth-server package.json with TypeScript 5.x, Node.js 20.x, Fastify 4.25+, jose 5.1+, bcrypt 5.1+, pg 8.11+
- [ ] T003 [P] Configure TypeScript in auth-server/tsconfig.json with strict mode and ES2022 target
- [ ] T004 [P] Generate RSA key pair for JWT signing in auth-server/keys/ (private.pem, public.pem) using openssl
- [ ] T005 [P] Create auth-server/.env.example with DATABASE_URL, JWT_PRIVATE_KEY_PATH, JWT_PUBLIC_KEY_PATH, PORT, ALLOWED_ORIGINS
- [ ] T006 [P] Add auth-server/keys/ to .gitignore to prevent committing private keys
- [ ] T007 Update backend/requirements.txt to add python-jose[cryptography] and httpx for JWT verification
- [ ] T008 Update frontend/package.json dependencies (no new packages needed, uses native fetch)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T009 Run database migration 001_initial_schema.sql to create users and personalized_content tables with enums in Neon PostgreSQL
- [ ] T010 [P] Implement auth-server/src/utils/jwt.ts with functions: signJWT(payload, privateKey), verifyJWT(token, publicKey)
- [ ] T011 [P] Implement auth-server/src/utils/jwks.ts with function: generateJWKS(publicKey) returning JWK format
- [ ] T012 [P] Implement auth-server/src/services/dbService.ts with database connection pool using pg library
- [ ] T013 [P] Implement auth-server/src/middleware/errorHandler.ts for global Fastify error handling
- [ ] T014 [P] Implement backend/app/services/jwt_service.py with JWKSCache class (fetch from auth server, 24-hour cache)
- [ ] T015 [P] Implement backend/app/middleware/auth_middleware.py with get_current_user dependency (JWT verification)
- [ ] T016 [P] Implement frontend/src/services/authService.ts with API client functions (signup, signin, refresh)
- [ ] T017 [P] Implement frontend/src/hooks/useAuth.ts with React hook for auth state management (localStorage integration)
- [ ] T018 [P] Implement frontend/src/components/auth/AuthContext.tsx with React Context provider for global auth state

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - New User Signup with Background Profile (Priority: P1) 🎯 MVP

**Goal**: Enable new users to create accounts with email, password, software_background, and hardware_background fields

**Independent Test**: Submit signup form with valid credentials, verify account in database, confirm JWT token issuance with embedded claims

### Tests for User Story 1 (MANDATORY per TDD) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T019 [P] [US1] Integration test for POST /api/auth/signup happy path in auth-server/tests/integration/signup.test.ts (valid signup, returns JWT)
- [ ] T020 [P] [US1] Integration test for POST /api/auth/signup validation errors in auth-server/tests/integration/signup.test.ts (missing fields, invalid enums, duplicate email)
- [ ] T021 [P] [US1] Unit test for AuthService.hashPassword in auth-server/tests/unit/authService.test.ts (bcrypt cost factor 12)
- [ ] T022 [P] [US1] Unit test for AuthService.createUser in auth-server/tests/unit/authService.test.ts (database insertion)
- [ ] T023 [P] [US1] Component test for SignupForm validation in frontend/tests/components/SignupForm.test.tsx (React Testing Library)

### Implementation for User Story 1

- [ ] T024 [US1] Implement Fastify server entry point in auth-server/src/server.ts with CORS, helmet, and route registration
- [ ] T025 [US1] Implement auth-server/src/services/authService.ts with hashPassword(password), createUser(email, passwordHash, softwareBackground, hardwareBackground)
- [ ] T026 [US1] Implement auth-server/src/middleware/validation.ts with JSON Schema for signup request body (email format, password minLength 8, background enums)
- [ ] T027 [US1] Implement POST /api/auth/signup route in auth-server/src/routes/signup.ts (validate input, hash password, create user, sign JWT, return tokens)
- [ ] T028 [P] [US1] Implement frontend/src/components/auth/SignupForm.tsx with form fields (email, password, software_background dropdown, hardware_background dropdown)
- [ ] T029 [P] [US1] Implement signup page in frontend/src/pages/signup.tsx that renders SignupForm and handles submission
- [ ] T030 [US1] Integrate signup flow: SignupForm submits to authService.signup(), stores tokens in localStorage, redirects to dashboard

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently. Run tests: `npm test` (auth-server), `npm test` (frontend)

---

## Phase 4: User Story 2 - User Sign-In with JWT Token (Priority: P1)

**Goal**: Enable returning users to sign in with email/password and receive JWT tokens with background claims

**Independent Test**: Submit signin form, verify JWT token issuance with correct claims (user_id, email, software_background, hardware_background)

### Tests for User Story 2 (MANDATORY per TDD) ⚠️

- [ ] T031 [P] [US2] Integration test for POST /api/auth/signin happy path in auth-server/tests/integration/signin.test.ts (valid credentials, returns JWT with claims)
- [ ] T032 [P] [US2] Integration test for POST /api/auth/signin error cases in auth-server/tests/integration/signin.test.ts (invalid email, wrong password, generic error messages)
- [ ] T033 [P] [US2] Unit test for AuthService.verifyPassword in auth-server/tests/unit/authService.test.ts (bcrypt comparison)
- [ ] T034 [P] [US2] Component test for SigninForm in frontend/tests/components/SigninForm.test.tsx (form submission, error display)

### Implementation for User Story 2

- [ ] T035 [US2] Implement AuthService.verifyPassword(password, passwordHash) in auth-server/src/services/authService.ts using bcrypt.compare
- [ ] T036 [US2] Implement AuthService.getUserByEmail(email) in auth-server/src/services/authService.ts (database query)
- [ ] T037 [US2] Implement POST /api/auth/signin route in auth-server/src/routes/signin.ts (validate email, verify password, fetch user, sign JWT with claims, return tokens)
- [ ] T038 [P] [US2] Implement frontend/src/components/auth/SigninForm.tsx with email and password fields
- [ ] T039 [P] [US2] Implement signin page in frontend/src/pages/signin.tsx that renders SigninForm
- [ ] T040 [US2] Integrate signin flow: SigninForm submits to authService.signin(), stores tokens in localStorage, updates AuthContext

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently. Users can signup and signin.

---

## Phase 5: User Story 4 - JWKS Public Key Verification (Priority: P1)

**Goal**: Enable Python backend to verify JWT tokens using public keys from auth server JWKS endpoint

**Independent Test**: Backend fetches JWKS on startup, verifies valid token (succeeds), verifies invalid token (fails), refreshes after 24 hours

**Note**: Implementing US4 before US3 because personalization (US3) depends on JWT verification (US4)

### Tests for User Story 4 (MANDATORY per TDD) ⚠️

- [ ] T041 [P] [US4] Integration test for GET /.well-known/jwks.json endpoint in auth-server/tests/integration/jwks.test.ts (returns valid JWK)
- [ ] T042 [P] [US4] Unit test for generateJWKS utility in auth-server/tests/unit/jwks.test.ts (exports RSA public key in JWK format)
- [ ] T043 [P] [US4] Unit test for JWKSCache.fetch_keys in backend/tests/test_jwt_service.py (httpx mock, cache expiration)
- [ ] T044 [P] [US4] Unit test for JWTService.verify_token in backend/tests/test_jwt_service.py (valid token, expired token, invalid signature)
- [ ] T045 [P] [US4] Integration test for auth_middleware in backend/tests/test_auth_middleware.py (requires valid JWT in Authorization header)

### Implementation for User Story 4

- [ ] T046 [US4] Implement GET /.well-known/jwks.json route in auth-server/src/routes/jwks.ts (export public key using jose exportJWK)
- [ ] T047 [US4] Implement POST /api/auth/refresh route in auth-server/src/routes/refresh.ts (verify refresh token, issue new access token)
- [ ] T048 [US4] Implement JWKSCache class in backend/app/services/jwt_service.py (fetch JWKS from auth server, cache for 24 hours with httpx)
- [ ] T049 [US4] Implement JWTService.verify_token in backend/app/services/jwt_service.py (use python-jose to verify RS256 signature, extract claims)
- [ ] T050 [US4] Initialize JWKS cache on backend startup in backend/app/main.py lifespan event (fetch public keys eagerly)
- [ ] T051 [US4] Update auth_middleware.py to call JWTService.verify_token and extract user_id, software_background, hardware_background from claims

**Checkpoint**: Backend can now verify JWT tokens. Test by calling backend with valid token from signin (should succeed), invalid token (should return 401).

---

## Phase 6: User Story 3 - Personalized Content Generation (Priority: P2)

**Goal**: Generate personalized chapter explanations based on user's software/hardware background using Gemini LLM with 7-day caching

**Independent Test**: Authenticate user, click "Personalize for Me", verify backend receives markdown+JWT, generates personalized content, caches result

### Tests for User Story 3 (MANDATORY per TDD) ⚠️

- [ ] T052 [P] [US3] Integration test for POST /api/personalize happy path in backend/tests/test_personalize_api.py (valid JWT, chapter markdown, returns personalized content)
- [ ] T053 [P] [US3] Integration test for POST /api/personalize cache hit in backend/tests/test_personalize_api.py (same request twice, second is cached)
- [ ] T054 [P] [US3] Integration test for POST /api/personalize unauthorized in backend/tests/test_personalize_api.py (missing JWT, invalid JWT)
- [ ] T055 [P] [US3] Unit test for PersonalizationService.generate_content in backend/tests/test_personalization_service.py (mock Gemini API, verify prompt includes background)
- [ ] T056 [P] [US3] Unit test for CacheService.get_cached_content in backend/tests/test_cache_service.py (SHA-256 hash, database lookup)
- [ ] T057 [P] [US3] Component test for PersonalizeButton in frontend/tests/components/PersonalizeButton.test.tsx (click triggers API call, displays loading state)

### Implementation for User Story 3

- [ ] T058 [P] [US3] Create SQLAlchemy model PersonalizedContent in backend/app/models/personalized_content.py (id, user_id, content_hash, content_type, content_payload JSONB, generated_at, expires_at)
- [ ] T059 [P] [US3] Implement CacheService in backend/app/services/cache_service.py with get_cached_content(user_id, content_hash) and store_content(user_id, content_hash, payload)
- [ ] T060 [US3] Implement PersonalizationService.generate_content in backend/app/services/personalization_service.py (call Gemini API with system prompt including software/hardware background)
- [ ] T061 [US3] Implement POST /api/personalize endpoint in backend/app/api/personalize.py (verify JWT, compute SHA-256 hash, check cache, generate if miss, store, return)
- [ ] T062 [P] [US3] Implement PersonalizeButton component in frontend/src/components/PersonalizeButton.tsx (button with loading state, calls personalizationService.personalize)
- [ ] T063 [P] [US3] Implement personalizationService.personalize in frontend/src/services/personalizationService.ts (POST to backend with JWT in Authorization header)
- [ ] T064 [P] [US3] Implement PersonalizationModal in frontend/src/theme/PersonalizationModal.tsx to display personalized content
- [ ] T065 [US3] Integrate PersonalizeButton into Docusaurus MDX pages (add button to chapter layout via swizzling)
- [ ] T066 [US3] Handle unauthenticated state: show signin prompt when PersonalizeButton clicked without JWT in localStorage

**Checkpoint**: All user stories should now be independently functional. Test end-to-end: signup → signin → navigate to chapter → click "Personalize for Me" → see personalized content → repeat (cache hit).

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T067 [P] Add rate limiting middleware to backend POST /api/personalize endpoint (10 requests per user per minute using slowapi)
- [ ] T068 [P] Implement GET /health endpoint in backend/app/api/health.py (check database, JWKS cache, Gemini API connectivity)
- [ ] T069 [P] Add CSP headers to frontend docusaurus.config.js to prevent XSS attacks (per research.md R4 decision)
- [ ] T070 [P] Implement token refresh logic in frontend useAuth hook (automatically refresh access token before API calls if expired)
- [ ] T071 [P] Add logging for personalization requests in backend (user_id, success/failure, no PII per Constitution Principle VII)
- [ ] T072 [P] Create database cleanup cron job script in backend/scripts/cleanup_expired_cache.py (DELETE FROM personalized_content WHERE expires_at < NOW())
- [ ] T073 Update frontend/docusaurus.config.js to add API proxy rewrites for local development (optional, for CORS simplification)
- [ ] T074 [P] Add loading spinners to SignupForm and SigninForm during API requests
- [ ] T075 [P] Add "Sign Out" button to frontend navbar that clears localStorage and resets AuthContext
- [ ] T076 Validate quickstart.md instructions by following setup steps on fresh environment
- [ ] T077 Run full test suite: `npm test` (auth-server), `pytest --cov=app` (backend), `npm test` (frontend) - ensure >80% coverage

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User Story 1 (Signup): Can start after Foundational - No dependencies on other stories
  - User Story 2 (Signin): Can start after Foundational - No dependencies on other stories
  - User Story 4 (JWKS): Can start after Foundational - No dependencies on other stories
  - User Story 3 (Personalization): Depends on User Story 4 (JWKS verification) - Backend must verify JWT before personalization
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1 - Signup)**: Independent - Can implement and test standalone
- **User Story 2 (P1 - Signin)**: Independent - Can implement and test standalone (assumes User Story 1 created test users)
- **User Story 4 (P1 - JWKS)**: Independent - Can implement and test standalone
- **User Story 3 (P2 - Personalization)**: Depends on User Story 4 for JWT verification - Backend needs JWKS to verify tokens

### Within Each User Story

- Tests MUST be written and FAIL before implementation (red-green-refactor cycle)
- Models before services
- Services before endpoints/routes
- Backend endpoints before frontend components
- Components before page integration
- Story complete before moving to next priority

### Parallel Opportunities

- **Phase 1 (Setup)**: All tasks marked [P] can run in parallel (T002-T008)
- **Phase 2 (Foundational)**: All tasks marked [P] can run in parallel (T010-T018)
- **User Story 1**: Tests T019-T023 in parallel, Frontend components T028-T029 in parallel
- **User Story 2**: Tests T031-T034 in parallel, Frontend components T038-T039 in parallel
- **User Story 4**: Tests T041-T045 in parallel
- **User Story 3**: Tests T052-T057 in parallel, Models/Services T058-T059 in parallel, Frontend components T062-T064 in parallel
- **Phase 7 (Polish)**: All tasks marked [P] can run in parallel (T067-T075)

**Different user stories can be worked on in parallel by different team members once Foundational phase completes**

---

## Parallel Example: User Story 1 (Signup)

```bash
# Launch all tests for User Story 1 together:
Task T019: Integration test POST /api/auth/signup happy path
Task T020: Integration test POST /api/auth/signup validation errors
Task T021: Unit test AuthService.hashPassword
Task T022: Unit test AuthService.createUser
Task T023: Component test SignupForm validation

# Launch frontend components together (after backend implementation):
Task T028: Implement SignupForm.tsx
Task T029: Implement signup page
```

---

## Implementation Strategy

### MVP First (User Stories 1, 2, 4 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Signup)
4. Complete Phase 4: User Story 2 (Signin)
5. Complete Phase 5: User Story 4 (JWKS verification)
6. **STOP and VALIDATE**: Test User Stories 1, 2, 4 independently
7. Deploy MVP to staging (signup/signin working with JWT verification)

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 (Signup) → Test independently → Deploy (users can create accounts)
3. Add User Story 2 (Signin) → Test independently → Deploy (users can sign in)
4. Add User Story 4 (JWKS) → Test independently → Deploy (backend can verify tokens)
5. Add User Story 3 (Personalization) → Test independently → Deploy (full feature complete)
6. Add Polish (Phase 7) → Final production deployment

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - **Developer A (Auth Specialist)**: User Story 1 + User Story 2 + User Story 4 (auth-server focus)
   - **Developer B (Backend Specialist)**: User Story 4 (backend JWT verification) → User Story 3 (personalization)
   - **Developer C (Frontend Specialist)**: User Story 1 frontend → User Story 2 frontend → User Story 3 frontend
3. Stories integrate at natural boundaries (e.g., Developer B waits for User Story 4 JWKS endpoint from Developer A before implementing backend verification)

---

## Notes

- [P] tasks = different files, no dependencies, can run in parallel
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail (RED) before implementing (GREEN)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Run full test suite after each user story to ensure no regressions
- Tests are MANDATORY per Constitution Principle II: TDD is NON-NEGOTIABLE

---

## Task Summary

**Total Tasks**: 77
- **Phase 1 (Setup)**: 8 tasks
- **Phase 2 (Foundational)**: 10 tasks
- **Phase 3 (User Story 1 - Signup, P1)**: 12 tasks (5 tests + 7 implementation)
- **Phase 4 (User Story 2 - Signin, P1)**: 10 tasks (4 tests + 6 implementation)
- **Phase 5 (User Story 4 - JWKS, P1)**: 11 tasks (5 tests + 6 implementation)
- **Phase 6 (User Story 3 - Personalization, P2)**: 15 tasks (6 tests + 9 implementation)
- **Phase 7 (Polish)**: 11 tasks

**Parallel Opportunities**: 45 tasks marked [P] can run in parallel when dependencies are met

**MVP Scope**: Phases 1-5 (47 tasks) deliver signup + signin + JWT verification

**Independent Test Criteria**:
- **US1**: Submit signup form → verify database account → verify JWT issued → PASS
- **US2**: Submit signin form → verify JWT with claims → PASS
- **US4**: Backend startup → fetch JWKS → verify token → PASS
- **US3**: Signin → click Personalize → verify LLM generation → verify cache → PASS

**Test Coverage Goal**: >80% (Constitution Principle II)
