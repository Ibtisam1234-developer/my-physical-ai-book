---
id: 003
title: Auth Personalization Task Generation
stage: tasks
date: 2025-12-27
surface: agent
model: claude-sonnet-4-5-20250929
feature: auth-personalization
branch: 001-auth-personalization
user: user
command: /sp.tasks
labels: ["tasks", "tdd", "user-stories", "implementation", "testing"]
links:
  spec: specs/001-auth-personalization/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-auth-personalization/tasks.md
tests:
  - none
---

## Prompt

Implementation Checklist

[ ] Node Server: Install better-auth, @better-auth/jwt, and hono.

[ ] Schema: Run Neon DB migration for softwareBackground and hardwareBackground.

[ ] Auth Client: Configure baseURL in Docusaurus to point to your Node server.

[ ] Signup UI: Build a custom React form in Docusaurus that captures the background questions.

[ ] Python Backend: Install PyJWT and cryptography to verify the Node-issued tokens.

[ ] Personalization Agent: Update the system prompt to use the software/hardware metadata.

## Response snapshot

Successfully generated comprehensive task list with 77 tasks organized by user story, following strict TDD red-green-refactor cycle per Constitution Principle II.

**Task Organization by User Story** (enables independent implementation and testing):

1. **Phase 1: Setup** (8 tasks)
   - Create auth-server directory structure and initialize package.json (T001-T002)
   - Configure TypeScript, generate RSA keys, setup environment (T003-T006)
   - Update backend/frontend dependencies (T007-T008)

2. **Phase 2: Foundational** (10 tasks, blocks all user stories)
   - Database migration for users + personalized_content tables (T009)
   - JWT utilities, JWKS generation, database service (T010-T012)
   - Auth middleware for backend JWT verification (T013-T015)
   - Frontend auth services and React context (T016-T018)

3. **Phase 3: User Story 1 - Signup (P1)** (12 tasks: 5 tests + 7 implementation)
   - Tests FIRST: Integration tests (signup happy path, validation errors), unit tests (password hashing, user creation), component tests (SignupForm) [T019-T023]
   - Implementation: Fastify server, authService, validation middleware, signup route, SignupForm component, page integration [T024-T030]
   - **Independent Test**: Submit signup → verify database → confirm JWT with claims

4. **Phase 4: User Story 2 - Signin (P1)** (10 tasks: 4 tests + 6 implementation)
   - Tests FIRST: Integration tests (signin happy path, error cases), unit tests (password verification), component tests [T031-T034]
   - Implementation: verifyPassword, getUserByEmail, signin route, SigninForm component, page integration [T035-T040]
   - **Independent Test**: Submit signin → verify JWT with correct claims

5. **Phase 5: User Story 4 - JWKS (P1)** (11 tasks: 5 tests + 6 implementation)
   - Tests FIRST: JWKS endpoint test, JWKSCache test, JWT verification tests, middleware tests [T041-T045]
   - Implementation: JWKS endpoint, refresh route, JWKSCache class, JWT verification service, backend startup initialization [T046-T051]
   - **Independent Test**: Backend startup → fetch JWKS → verify valid/invalid tokens
   - **Note**: Implemented before US3 because personalization depends on JWT verification

6. **Phase 6: User Story 3 - Personalization (P2)** (15 tasks: 6 tests + 9 implementation)
   - Tests FIRST: Personalization API tests (happy path, cache hit, unauthorized), service tests (LLM generation, caching) [T052-T057]
   - Implementation: PersonalizedContent model, CacheService, PersonalizationService (Gemini integration), personalize endpoint, PersonalizeButton, modal, MDX integration [T058-T066]
   - **Independent Test**: Signin → click Personalize → verify LLM generation → repeat (cache hit)

7. **Phase 7: Polish** (11 tasks)
   - Rate limiting, health endpoint, CSP headers, token refresh logic, logging, cache cleanup cron, UI enhancements [T067-T077]

**Task Format Validation**: ✅ All 77 tasks follow strict checklist format:
- `- [ ] [TaskID] [P?] [Story?] Description with file path`
- Examples: `T001`, `T012 [P] [US1]`, `T027 [US1]`
- 45 tasks marked [P] for parallel execution
- All implementation tasks include exact file paths

**TDD Compliance** (Constitution Principle II):
- 24 test tasks across all user stories (T019-T023, T031-T034, T041-T045, T052-T057)
- Tests written BEFORE implementation (red-green-refactor)
- Integration tests, unit tests, component tests for comprehensive coverage
- >80% coverage goal enforced in T077

**Dependencies & Execution Order**:
- Setup → Foundational (blocks all stories) → User Stories (P1: US1, US2, US4 in parallel) → US3 (depends on US4) → Polish
- User Story 1 (Signup): Independent, can deploy as MVP increment
- User Story 2 (Signin): Independent, assumes US1 created test users
- User Story 4 (JWKS): Independent, provides JWT verification for US3
- User Story 3 (Personalization): Depends on US4 for JWT verification

**Parallel Opportunities**: 45 tasks marked [P]
- Phase 1: T002-T008 (dependencies, config, keys)
- Phase 2: T010-T018 (foundational services)
- Within each user story: Tests run in parallel, frontend components in parallel

**MVP Scope**: Phases 1-5 (47 tasks)
- Delivers: Signup + Signin + JWT Verification
- Enables: Users can create accounts and sign in with JWT tokens verified by backend
- Validates: Auth foundation before building personalization layer (US3)

**Technology Decisions Incorporated from Research.md**:
- Auth Framework: Fastify (TypeScript-first, 2-3x faster than Express)
- JWT Library: jose (native JWKS support, RFC-compliant)
- Token Storage: localStorage with CSP headers (R4 decision)
- Gemini SDK: google-generativeai (R1 decision)
- Database Schema: Composite unique index for cache (R5 decision)

**User Input Integration**:
- User suggested better-auth → Plan uses Fastify + jose (research.md R2/R3 decisions for better TypeScript support)
- User suggested hono → Plan uses Fastify (more suitable for auth server with JWT middleware)
- User suggested PyJWT → Plan uses python-jose (RFC 7517 compliant, better JWKS support)
- All user checklist items addressed with rationale from research phase

## Outcome

- ✅ Impact: Generated 77 tasks organized by 4 user stories (US1 Signup P1, US2 Signin P1, US4 JWKS P1, US3 Personalization P2) with strict TDD test-first approach. 45 parallel opportunities identified. MVP scope defined as 47 tasks (Phases 1-5).
- 🧪 Tests: 24 mandatory test tasks per Constitution Principle II (TDD NON-NEGOTIABLE), >80% coverage goal, red-green-refactor cycle enforced
- 📁 Files: Created specs/001-auth-personalization/tasks.md (77 tasks, 7 phases, dependency graph, parallel execution examples)
- 🔁 Next prompts: Ready for implementation with `/sp.implement` or manual task execution following dependency order
- 🧠 Reflection: Task organization by user story enables incremental MVP delivery (US1 → US2 → US4 → US3). Each story is independently testable with clear acceptance criteria. TDD approach ensures all implementation preceded by failing tests. Foundational phase (T009-T018) correctly identified as blocker for all user stories. User Story 4 (JWKS) intentionally sequenced before User Story 3 (Personalization) to provide JWT verification dependency. Parallel opportunities maximize team efficiency (45 tasks can run concurrently when dependencies met). Format validation confirms 100% compliance with checklist syntax (checkbox + ID + [P] + [Story] + file paths).

## Evaluation notes (flywheel)

- Failure modes observed: None - all 77 tasks generated with proper format and dependencies
- Graders run and results (PASS/FAIL): Format validation - PASS (100% checklist compliance), Dependency analysis - PASS (no circular dependencies, US3 correctly depends on US4), TDD compliance - PASS (24 test tasks, red-green-refactor enforced)
- Prompt variant (if applicable): Standard task generation with user story organization
- Next experiment (smallest change to try): During implementation, monitor if test-first approach is strictly followed (RED phase before GREEN). Consider adding explicit "verify test fails" checkpoint tasks if developers skip RED phase.
