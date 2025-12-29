# Feature Specification: Authentication & Personalization System

**Feature Branch**: `001-auth-personalization`
**Created**: 2025-12-27
**Status**: Draft
**Input**: User description: "Node Auth Server with extended signup for software/hardware background, JWT issuance with background claims, Docusaurus integration with auth-client, and personalize button in chapters that sends markdown and JWT to Python backend"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - New User Signup with Background Profile (Priority: P1)

A new student visits the Physical AI learning platform and creates an account by providing their email, password, software experience level, and hardware experience level. This information is captured during signup to enable immediate personalization of learning content without requiring additional profile setup steps.

**Why this priority**: This is the entry point for all users. Without signup, no other features are accessible. Capturing background information at signup eliminates friction in the user journey by avoiding separate profile configuration steps.

**Independent Test**: Can be fully tested by submitting the signup form with valid credentials and background selections, verifying account creation in the database, and confirming JWT token issuance with embedded profile claims.

**Acceptance Scenarios**:

1. **Given** a new user on the signup page, **When** they enter email "student@example.com", password "SecurePass123!", software_background "intermediate", and hardware_background "hobbyist", **Then** account is created, JWT token is issued with embedded claims, and user is redirected to dashboard
2. **Given** a new user on the signup page, **When** they submit the form without selecting software_background, **Then** validation error displays "Software background is required" and form does not submit
3. **Given** a new user on the signup page, **When** they submit the form without selecting hardware_background, **Then** validation error displays "Hardware background is required" and form does not submit
4. **Given** a new user on the signup page, **When** they enter an email that already exists, **Then** error message displays "Email already registered" without revealing whether account exists (security)
5. **Given** a new user on the signup page, **When** they enter a weak password (less than 8 characters), **Then** validation error displays "Password must be at least 8 characters"

---

### User Story 2 - User Sign-In with JWT Token (Priority: P1)

A returning student signs in with their email and password, receiving a JWT token that contains their user ID and background profile information. This token enables stateless authentication across the platform (Docusaurus frontend and Python backend) without requiring database lookups on every request.

**Why this priority**: Sign-in is as critical as signup for returning users. JWT-based stateless authentication is foundational for the personalization features in subsequent user stories.

**Independent Test**: Can be fully tested by submitting valid credentials, verifying JWT token issuance with correct claims (user_id, email, software_background, hardware_background), and confirming token can be verified by both Node auth server and Python backend using JWKS public key.

**Acceptance Scenarios**:

1. **Given** an existing user with email "student@example.com" and password "SecurePass123!", **When** they submit the sign-in form, **Then** JWT token is returned with claims {user_id, email, software_background, hardware_background} and expires_in set to 3600 seconds (1 hour)
2. **Given** an existing user on the sign-in page, **When** they enter incorrect password, **Then** generic error message displays "Invalid email or password" (no indication of which is wrong for security)
3. **Given** an existing user on the sign-in page, **When** they enter an email that doesn't exist, **Then** generic error message displays "Invalid email or password"
4. **Given** an existing user with expired access token, **When** they attempt to access protected content, **Then** system prompts for token refresh or re-authentication
5. **Given** the Python backend receives a request with JWT token, **When** it verifies the token using the Node auth server's JWKS public key, **Then** token signature is validated and claims are extracted successfully

---

### User Story 3 - Personalized Content Generation (Priority: P2)

A student reading a chapter on "ROS 2 Navigation Stack" clicks the "Personalize for Me" button. The system sends the chapter markdown content and the student's JWT token to the Python backend, which uses the student's software_background and hardware_background to generate a personalized explanation tailored to their experience level.

**Why this priority**: This is the core value proposition of the personalization system. It builds on the authentication foundation (P1 stories) to deliver adaptive learning experiences. Prioritized as P2 because it requires signup/signin to be functional first.

**Independent Test**: Can be fully tested by authenticating a user, navigating to any chapter, clicking "Personalize for Me", and verifying the backend receives the chapter markdown and JWT, extracts background from JWT claims, generates personalized content via LLM, and returns tailored explanation to the frontend.

**Acceptance Scenarios**:

1. **Given** an authenticated user (intermediate software, hobbyist hardware) viewing the "ROS 2 Navigation" chapter, **When** they click "Personalize for Me", **Then** personalized content is generated emphasizing practical hobbyist applications with intermediate-level code examples
2. **Given** an authenticated user (beginner software, no hardware) viewing the same chapter, **When** they click "Personalize for Me", **Then** personalized content is generated with more foundational explanations, glossary of terms, and simplified examples
3. **Given** an authenticated user viewing a chapter, **When** they click "Personalize for Me" twice within 5 minutes, **Then** cached personalized content is returned from database instead of generating new content (cost optimization)
4. **Given** an unauthenticated user viewing a chapter, **When** they click "Personalize for Me", **Then** sign-in prompt displays explaining "Sign in to get content personalized to your experience level"
5. **Given** an authenticated user viewing a chapter, **When** the personalization request fails due to LLM API error, **Then** user-friendly error message displays "Unable to generate personalized content. Please try again." with option to retry

---

### User Story 4 - JWKS Public Key Verification (Priority: P1)

The Python backend verifies JWT tokens from the frontend by fetching the public key from the Node auth server's JWKS endpoint (`/.well-known/jwks.json`). This enables stateless authentication where the Python backend can validate tokens without direct database access or shared secrets.

**Why this priority**: This is a foundational security requirement. Without proper JWT verification, the system is vulnerable to forged tokens and unauthorized access. Grouped with P1 because it's essential for secure operation of all authenticated features.

**Independent Test**: Can be fully tested by configuring the Python backend to fetch JWKS from the Node auth server on startup, attempting to verify a valid JWT token (should succeed), attempting to verify a token with invalid signature (should fail), and confirming public key cache refresh after 24 hours.

**Acceptance Scenarios**:

1. **Given** the Python backend starts up, **When** it initializes the JWT verification module, **Then** it fetches the JWKS from Node auth server at `https://auth.yourdomain.com/.well-known/jwks.json` and caches the public keys
2. **Given** the Python backend receives a request with valid JWT token, **When** it verifies the token signature using cached public key, **Then** verification succeeds and claims are extracted
3. **Given** the Python backend receives a request with JWT token signed by wrong key, **When** it attempts verification, **Then** HTTP 401 Unauthorized is returned with error "Invalid token signature"
4. **Given** the Python backend's public key cache is 24 hours old, **When** it attempts to verify a token, **Then** it refreshes the JWKS from the Node auth server before verification
5. **Given** the Python backend cannot reach the JWKS endpoint, **When** it attempts to verify a token and cache is stale, **Then** it logs the error and returns HTTP 503 Service Unavailable with message "Authentication service temporarily unavailable"

---

### Edge Cases

- What happens when a user changes their software_background or hardware_background after signup? (Assumption: profile fields are immutable after creation per constitution; users must explicitly request profile edit which invalidates cached personalized content)
- How does the system handle concurrent personalization requests from the same user? (Assumption: requests are processed independently; cache prevents redundant LLM generation)
- What happens if the JWKS endpoint is down during token verification? (Handled in User Story 4, Scenario 5: use cached keys if available, return 503 if cache stale)
- How does the system handle malformed JWT tokens? (Assumption: return HTTP 401 with clear error message "Invalid token format")
- What happens if a chapter has no content (empty markdown)? (Assumption: personalize button is disabled or returns error "No content available to personalize")
- How long are personalized content cache entries retained? (Per constitution: 7 days expiration, invalidated on profile update)
- What happens if LLM returns inappropriate or incorrect content? (Out of scope for this feature; content moderation is a separate concern)
- How are rate limits enforced on personalization requests? (Assumption: backend implements per-user rate limiting, e.g., 10 requests per minute, returns HTTP 429 if exceeded)

## Requirements *(mandatory)*

### Functional Requirements

**Authentication Service (Node.js)**:

- **FR-001**: System MUST expose a signup endpoint at `/api/auth/signup` accepting email, password, software_background (enum: beginner|intermediate|advanced|expert), and hardware_background (enum: none|hobbyist|student|professional)
- **FR-002**: System MUST validate that software_background and hardware_background are provided and match allowed enum values during signup, returning HTTP 400 with specific field errors if validation fails
- **FR-003**: System MUST store user account data in Neon PostgreSQL users table with columns: id (UUID primary key), email (unique, not null), password_hash (bcrypt with cost factor 12), software_background (enum, not null), hardware_background (enum, not null), created_at (timestamp), updated_at (timestamp)
- **FR-004**: System MUST expose a sign-in endpoint at `/api/auth/signin` accepting email and password, returning JWT access token (1 hour expiration) and refresh token (30 days expiration) on success
- **FR-005**: System MUST generate RS256-signed JWT tokens with claims: user_id (UUID), email (string), software_background (string), hardware_background (string), iat (issued at timestamp), exp (expiration timestamp)
- **FR-006**: System MUST expose a JWKS endpoint at `/.well-known/jwks.json` returning the RSA public key in JWK format for token verification by other services
- **FR-007**: System MUST implement token refresh endpoint at `/api/auth/refresh` accepting refresh token and returning new access token without requiring re-authentication
- **FR-008**: System MUST return generic error messages for authentication failures (e.g., "Invalid email or password") without revealing whether email exists or which field is incorrect

**Docusaurus Frontend Integration**:

- **FR-009**: System MUST provide a client-side authentication module that communicates with the Node auth server endpoints (`/api/auth/signup`, `/api/auth/signin`, `/api/auth/refresh`)
- **FR-010**: System MUST store JWT access token and refresh token securely in browser (HTTP-only cookies preferred, or secure localStorage with XSS mitigations)
- **FR-011**: System MUST display "Personalize for Me" button on each documentation chapter page for authenticated users
- **FR-012**: System MUST hide or disable "Personalize for Me" button for unauthenticated users, showing sign-in prompt when clicked
- **FR-013**: System MUST send POST request to Python backend personalization endpoint with chapter markdown content and JWT token in Authorization header when "Personalize for Me" is clicked
- **FR-014**: System MUST display personalized content returned from backend in a modal, sidebar, or inline expansion (UI design decision deferred to implementation)
- **FR-015**: System MUST handle token expiration gracefully by attempting token refresh automatically before re-prompting for sign-in

**Python Backend Personalization**:

- **FR-016**: System MUST expose a personalization endpoint at `/api/personalize` (POST) accepting chapter_markdown (string) and JWT token in Authorization header
- **FR-017**: System MUST verify JWT token signature using public key fetched from Node auth server JWKS endpoint on backend startup and cached for 24 hours
- **FR-018**: System MUST extract user_id, software_background, and hardware_background from verified JWT claims
- **FR-019**: System MUST check personalized_content cache table for existing entry matching (user_id, chapter_markdown hash, not expired) before generating new content
- **FR-020**: System MUST generate personalized content by calling Gemini LLM with system prompt injecting software_background and hardware_background context, and user prompt containing chapter_markdown
- **FR-021**: System MUST store generated personalized content in Neon PostgreSQL personalized_content table with columns: id (UUID), user_id (foreign key to users.id), content_type (enum: 'curriculum_path'), content_payload (JSONB containing personalized_text), generated_at (timestamp), expires_at (timestamp, default generated_at + 7 days)
- **FR-022**: System MUST return HTTP 401 Unauthorized if JWT token is missing, malformed, expired, or has invalid signature
- **FR-023**: System MUST return HTTP 503 Service Unavailable if JWKS endpoint is unreachable and cached public keys are stale (older than 24 hours)
- **FR-024**: System MUST log all personalization requests with user_id, timestamp, and success/failure status for audit purposes (without logging PII fields software_background/hardware_background per constitution)

**Privacy & Security**:

- **FR-025**: System MUST NOT log software_background or hardware_background values in application logs, error messages, or analytics systems (PII per constitution VII)
- **FR-026**: System MUST use software_background and hardware_background ONLY in LLM prompt context for personalization, never in user-visible UI outside profile settings
- **FR-027**: System MUST encrypt all database connections (TLS) and enforce HTTPS for all API endpoints in production
- **FR-028**: System MUST implement rate limiting on personalization endpoint (assumption: 10 requests per user per minute) returning HTTP 429 Too Many Requests if exceeded

### Key Entities

- **User Account**: Represents a student or educator with email, password, software experience level (beginner/intermediate/advanced/expert), and hardware experience level (none/hobbyist/student/professional). Stored in Neon PostgreSQL users table. Related to PersonalizedContent entries via user_id foreign key.

- **JWT Token**: Represents authentication credential containing user_id, email, software_background, hardware_background, issued-at timestamp, and expiration timestamp. Signed by Node auth server using RS256 private key, verified by Python backend using JWKS public key. Not persisted in database (stateless).

- **Personalized Content**: Represents LLM-generated explanation of a chapter tailored to user's background. Contains user_id (who requested it), content hash (which chapter), personalized_text (LLM output), generation timestamp, and expiration timestamp (7 days). Stored in Neon PostgreSQL personalized_content table.

- **JWKS Public Key**: Represents RSA public key used to verify JWT signatures. Fetched from Node auth server `/.well-known/jwks.json` endpoint. Cached by Python backend for 24 hours to reduce network overhead.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: New users can complete signup (including email, password, and both background selections) in under 90 seconds from landing on signup page to receiving authenticated session
- **SC-002**: Returning users can sign in and receive valid JWT token in under 3 seconds from form submission to token receipt
- **SC-003**: Personalized content generation completes in under 5 seconds from button click to content display for 95% of requests (excluding LLM API latency outliers)
- **SC-004**: JWT token verification by Python backend succeeds in under 100 milliseconds for cached public keys (no network fetch required)
- **SC-005**: System handles 100 concurrent personalization requests without degradation or timeouts
- **SC-006**: Personalized content cache reduces LLM API costs by at least 60% (measured by cache hit rate: cached responses / total requests)
- **SC-007**: Zero instances of software_background or hardware_background appearing in application logs, error messages, or analytics over 30-day period (PII compliance)
- **SC-008**: 90% of users successfully complete signup on first attempt without validation errors (indicates clear form UX and helpful error messages)
- **SC-009**: Token refresh mechanism reduces sign-in friction by 80% (users remain authenticated for 30-day refresh token lifetime without re-entering password)

## Assumptions

- **Assumption 1**: Email/password authentication is sufficient for MVP; social logins (Google, GitHub) are out of scope for this feature
- **Assumption 2**: Password reset functionality is out of scope for this feature; will be addressed in future iteration
- **Assumption 3**: Profile editing (changing software_background or hardware_background after signup) is out of scope for this feature; fields are immutable after creation per constitution
- **Assumption 4**: The Node auth server is deployed as a standalone service at a subdomain (e.g., `auth.yourdomain.com`) accessible to both Docusaurus frontend and Python backend
- **Assumption 5**: Personalized content cache uses SHA-256 hash of chapter markdown as the content identifier for cache lookups
- **Assumption 6**: Rate limiting on personalization endpoint is 10 requests per user per minute; this can be adjusted based on monitoring
- **Assumption 7**: JWKS public key cache refresh (24 hours) is sufficient for key rotation scenarios; if keys are rotated more frequently, cache TTL can be reduced
- **Assumption 8**: Personalized content is displayed in the same language as the original chapter (localization/translation is out of scope)
- **Assumption 9**: The "Personalize for Me" button is available on all chapter pages; specific chapter exclusions (e.g., introduction, appendix) are out of scope
- **Assumption 10**: LLM prompt engineering for personalization quality is handled during implementation; spec defines the mechanism, not the prompt content
