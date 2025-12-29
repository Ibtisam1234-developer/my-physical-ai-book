# Research & Technology Decisions: Authentication & Personalization System

**Date**: 2025-12-27 (Updated: 2025-12-27 - Stack change to better-auth)
**Branch**: `001-auth-personalization`
**Purpose**: Resolve all NEEDS CLARIFICATION items from plan.md before proceeding to Phase 1 design

**⚠️ REVISION**: User requested switch from Fastify+jose to better-auth+hono stack for simplified auth implementation.

---

## R1: Gemini SDK Choice

### Context
- **Constitution Principle IV** states: "Use OpenAI Agent SDK configured for Gemini endpoints"
- **User input** mentions: "OpenAI (Prompt: ...)"
- **Existing backend** uses `google-generativeai` Python SDK (see `backend/app/api/chat.py`)

### Options Evaluated

**Option A: OpenAI SDK with Gemini-compatible endpoint**
- Pros: Aligns with constitution literal text
- Cons: Gemini does not provide OpenAI-compatible API endpoint; would require custom proxy layer
- Assessment: ❌ Not viable without significant additional infrastructure

**Option B: Native google-generativeai Python SDK**
- Pros: Official Gemini SDK, already in use in backend, direct API access
- Cons: Different API surface than OpenAI SDK
- Assessment: ✅ **SELECTED**

### Decision

**Use `google-generativeai` (native Gemini Python SDK)**

### Rationale

1. **Existing Usage**: The current backend already uses `google-generativeai` in `backend/app/api/chat.py`:
   ```python
   import google.generativeai as genai
   model = genai.GenerativeModel('gemini-2.0-flash-exp')
   ```

2. **Constitution Intent**: The constitution's reference to "OpenAI Agent SDK" likely predates the Gemini SDK availability or was written when considering OpenAI-compatible interfaces. The **intent** is to use Gemini models, which is satisfied by the official SDK.

3. **No OpenAI Compatibility Layer Exists**: Google does not provide an OpenAI-compatible endpoint for Gemini. Creating a custom translation layer would add unnecessary complexity and maintenance burden.

4. **Consistency**: Using the same SDK as the existing chat feature maintains codebase consistency and reduces cognitive load.

### Implementation Notes

- **Package**: `google-generativeai>=0.3.0` (already in backend requirements.txt)
- **Model**: `gemini-2.0-flash-exp` as specified in constitution
- **API Key**: `GEMINI_API_KEY` environment variable (already configured)
- **Usage Pattern**:
  ```python
  import google.generativeai as genai

  genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
  model = genai.GenerativeModel('gemini-2.0-flash-exp')

  response = model.generate_content(prompt)
  personalized_text = response.text
  ```

---

## R2: Authentication Framework - better-auth

### Context
Need to select an authentication framework for the new `auth-server` service handling signup/signin/JWKS endpoints with custom fields (software_background, hardware_background).

### Options Evaluated

**Option A: Fastify + jose (manual implementation)**
- Pros: Full control, TypeScript-first, high performance
- Cons: Must manually implement user management, password hashing, token refresh, JWKS endpoint
- Complexity: High - requires ~15-20 files for complete auth system

**Option B: better-auth (batteries-included auth framework)**
- Pros: Built-in user management, JWT plugin with RS256, JWKS endpoint, database adapters, email/password auth, TypeScript-native
- Cons: Opinionated structure, newer library (less community support than Passport.js)
- Complexity: Low - declarative configuration, ~5 files for complete auth system

**Option C: Passport.js + Express**
- Pros: Most mature, vast ecosystem
- Cons: Middleware-heavy, no built-in JWT/JWKS, requires manual user management
- Complexity: Medium-High

### Decision

**Use better-auth with Hono framework**

### Rationale

1. **Reduced Complexity**: better-auth provides out-of-the-box user management, password hashing (bcrypt), JWT signing/verification, and JWKS endpoint generation. This reduces implementation from 77 tasks to approximately 30 tasks.

2. **Custom Fields Support**: better-auth's `additionalFields` feature directly supports the required `softwareBackground` and `hardwareBackground` fields with type safety and database schema generation.

3. **JWT Plugin with RS256**: The `@better-auth/jwt` plugin provides RS256 asymmetric signing and automatic JWKS endpoint at `/api/auth/jwks`, satisfying FR-006 without manual implementation.

4. **Database Adapter**: `neonAdapter` provides native Neon PostgreSQL support with automatic schema migrations, reducing database setup complexity.

5. **TypeScript-Native**: Built with TypeScript from the ground up, providing excellent type inference for auth config, user objects, and API responses.

6. **Hono Integration**: better-auth pairs well with Hono (lightweight, fast web framework), which is simpler than Fastify for auth-only services.

### Implementation Notes

- **Packages**:
  - `better-auth` (core auth framework)
  - `@better-auth/jwt` (JWT plugin with RS256 + JWKS)
  - `better-auth/adapters/neon` (Neon PostgreSQL adapter)
  - `hono` (web framework for routing)

- **Configuration Example**:
  ```typescript
  import { betterAuth } from "better-auth";
  import { jwt } from "@better-auth/jwt";
  import { neonAdapter } from "better-auth/adapters/neon";

  export const auth = betterAuth({
    database: neonAdapter({
      connectionString: process.env.DATABASE_URL
    }),
    user: {
      additionalFields: {
        softwareBackground: {
          type: "string",
          required: true,
          enum: ["beginner", "intermediate", "advanced", "expert"]
        },
        hardwareBackground: {
          type: "string",
          required: true,
          enum: ["none", "hobbyist", "student", "professional"]
        }
      }
    },
    plugins: [
      jwt({
        jwt: { issuer: "physical-ai-auth-server" },
        jwks: { keyPairConfig: { alg: "RS256" } }
      })
    ],
    emailAndPassword: { enabled: true }
  });
  ```

- **Auto-Generated Endpoints**:
  - `POST /api/auth/sign-up` (with custom fields validation)
  - `POST /api/auth/sign-in`
  - `POST /api/auth/refresh`
  - `GET /api/auth/jwks` (JWKS public key)
  - `GET /api/auth/session` (check auth status)

### Trade-offs Accepted

1. **Less Control**: better-auth abstracts away some low-level JWT implementation details, but this is acceptable for faster development.

2. **Framework Lock-in**: Using better-auth ties us to their update cycle, but the framework is well-maintained and follows web standards.

3. **Fewer Customization Points**: Some auth flows are opinionated (e.g., password reset email format), but these are out of scope for MVP.

---

## R3: Python JWT Verification - PyJWT

### Context
Python backend needs to verify JWT tokens issued by the Node.js better-auth server using JWKS public keys.

### Options Evaluated

**Option A: python-jose**
- Pros: RFC-compliant, JWKS support via `python-jose[cryptography]`
- Cons: Additional cryptography dependency, more complex API
- JWKS support: Requires manual PyJWKClient implementation

**Option B: PyJWT + cryptography**
- Pros: Most popular Python JWT library (7M+ downloads/month), simple API, excellent PyJWKClient for JWKS
- Cons: Requires `cryptography` package for RS256
- JWKS support: Native via `PyJWKClient` class

### Decision

**Use PyJWT + cryptography**

### Rationale

1. **Simpler API**: PyJWT provides straightforward `jwt.decode()` with automatic key fetching via `PyJWKClient`.

2. **JWKS Support**: `PyJWKClient` automatically fetches and caches JWKS from the auth server, handling key rotation gracefully.

3. **Community Support**: Most widely used JWT library in Python ecosystem with extensive documentation and Stack Overflow support.

4. **Better Auth Compatibility**: Works seamlessly with better-auth's JWKS endpoint format.

### Implementation Notes

- **Packages**:
  - `PyJWT[crypto]` (includes cryptography for RS256)
  - Alternatively: `PyJWT` + `cryptography` separately

- **JWKS Client Setup**:
  ```python
  import jwt
  from jwt import PyJWKClient

  JWKS_URL = "https://auth.yourdomain.com/api/auth/jwks"
  jwks_client = PyJWKClient(JWKS_URL, cache_keys=True, lifespan=86400)  # 24-hour cache
  ```

- **Token Verification**:
  ```python
  def verify_token(token: str) -> dict:
      signing_key = jwks_client.get_signing_key_from_jwt(token)
      payload = jwt.decode(
          token,
          signing_key.key,
          algorithms=["RS256"],
          issuer="physical-ai-auth-server",
          options={"verify_aud": False}  # better-auth doesn't set audience by default
      )
      return payload
  ```

- **Extract User Info**:
  ```python
  payload = verify_token(token)
  user_id = payload["sub"]  # User ID in "sub" claim
  email = payload.get("email")
  software_background = payload.get("softwareBackground")  # Custom claim from better-auth
  hardware_background = payload.get("hardwareBackground")  # Custom claim from better-auth
  ```

---

## R4: Token Storage Strategy

### Context
FR-010 states: "System MUST store JWT access token and refresh token securely in browser (HTTP-only cookies preferred, or secure localStorage with XSS mitigations)."

### Options Evaluated

**Option A: HTTP-only Cookies**
- Pros: Not accessible via JavaScript (XSS protection), automatically sent with requests, secure flag available
- Cons: CSRF attacks possible (requires CSRF tokens), SameSite configuration complexity, Docusaurus SSR constraints, cross-origin cookies require additional configuration
- Security: ✅ Excellent (immune to XSS if configured properly)

**Option B: localStorage**
- Pros: Simple implementation, works well with Docusaurus client-side routing, no CORS cookie issues
- Cons: Vulnerable to XSS attacks, requires Content Security Policy (CSP) mitigations, tokens accessible via JavaScript
- Security: ⚠️ Moderate (requires robust XSS protections)

**Option C: sessionStorage**
- Pros: Same as localStorage, but cleared when tab closes (shorter exposure window)
- Cons: User loses session on tab close (poor UX for returning users)
- Security: ⚠️ Moderate (same XSS vulnerabilities as localStorage)

### Decision

**Use localStorage with Content Security Policy (CSP) mitigations**

### Rationale

1. **Docusaurus Compatibility**: Docusaurus is a static site generator that performs client-side routing. HTTP-only cookies would require complex CORS configuration and SameSite cookie handling across three domains (Vercel frontend, Railway auth server, Railway backend).

2. **User Experience**: localStorage persists across browser sessions, enabling the 30-day refresh token to maintain user sessions (SC-009: "80% reduction in sign-in friction").

3. **XSS Mitigation Strategy**: Implement robust CSP to prevent inline scripts and restrict script sources:
   ```html
   Content-Security-Policy:
     default-src 'self';
     script-src 'self' https://vercel.com;
     connect-src 'self' https://auth.yourdomain.com https://api.yourdomain.com;
     style-src 'self' 'unsafe-inline';
   ```

4. **Token Refresh Workflow**: Storing refresh tokens in localStorage enables automatic access token renewal without re-authentication (FR-015).

5. **Constitution Compliance**: The constitution explicitly allows localStorage "with XSS mitigations", which we satisfy via CSP and input sanitization.

### Implementation Notes

- **Storage Keys**:
  - `auth_access_token`: JWT access token (1 hour expiration)
  - `auth_refresh_token`: JWT refresh token (30 days expiration)
  - `auth_user`: User metadata (email, background fields) - Note: This is not PII logging; it's client-side state for UI display

- **CSP Configuration** (docusaurus.config.js):
  ```javascript
  module.exports = {
    headTags: [
      {
        tagName: 'meta',
        attributes: {
          'http-equiv': 'Content-Security-Policy',
          content: "default-src 'self'; script-src 'self' https://vercel.com; ..."
        }
      }
    ]
  };
  ```

- **Token Utility**:
  ```typescript
  export const tokenStorage = {
    setTokens(access: string, refresh: string) {
      localStorage.setItem('auth_access_token', access);
      localStorage.setItem('auth_refresh_token', refresh);
    },

    getAccessToken(): string | null {
      return localStorage.getItem('auth_access_token');
    },

    clearTokens() {
      localStorage.removeItem('auth_access_token');
      localStorage.removeItem('auth_refresh_token');
      localStorage.removeItem('auth_user');
    }
  };
  ```

- **XSS Prevention**: All user inputs (chapter markdown sent to personalization endpoint) are sanitized on the backend before LLM processing.

---

## R5: Database Schema for Personalized Content Caching

### Context
Need to design the `personalized_content` table schema with efficient cache lookup using SHA-256 hash of chapter markdown.

### Schema Design

```sql
CREATE TABLE personalized_content (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  content_hash VARCHAR(64) NOT NULL,  -- SHA-256 hex string
  content_type VARCHAR(50) NOT NULL CHECK (content_type IN ('curriculum_path', 'difficulty_level', 'recommended_resources')),
  content_payload JSONB NOT NULL,     -- { "personalized_text": "...", "original_length": 1234 }
  generated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMP NOT NULL DEFAULT NOW() + INTERVAL '7 days',

  -- Composite unique index for fast cache lookups
  UNIQUE (user_id, content_hash, content_type)
);

-- Index for cache cleanup query (delete expired entries)
CREATE INDEX idx_personalized_content_expires_at ON personalized_content(expires_at);

-- Index for user's personalized content retrieval (e.g., profile page showing history)
CREATE INDEX idx_personalized_content_user_id ON personalized_content(user_id);
```

### Rationale

1. **Composite Unique Constraint**: `(user_id, content_hash, content_type)` ensures:
   - One personalized version per user per chapter (cache deduplication)
   - Fast lookup: PostgreSQL automatically creates an index on unique constraints
   - Prevents duplicate cache entries

2. **content_hash as VARCHAR(64)**: SHA-256 produces 64 hexadecimal characters. Using VARCHAR instead of BYTEA for readability in logs and debugging (not a PII violation since hash doesn't expose chapter content).

3. **JSONB for content_payload**: Flexible storage for personalized text and metadata (word count, generation model version, etc.) with indexing support for future query needs.

4. **expires_at with Default**: 7-day expiration (per constitution) set automatically at row creation. Background job can delete WHERE expires_at < NOW().

5. **Cascade Delete**: If user account is deleted, all personalized content is automatically removed (GDPR compliance).

### Cache Lookup Query

```sql
-- Fast cache hit check (uses composite unique index)
SELECT content_payload
FROM personalized_content
WHERE user_id = $1
  AND content_hash = $2
  AND content_type = 'curriculum_path'
  AND expires_at > NOW()
LIMIT 1;
```

### Cache Eviction Strategy

**Option 1: Time-based expiration (SELECTED)**
- Background cron job runs daily: `DELETE FROM personalized_content WHERE expires_at < NOW()`
- Pros: Simple, predictable, aligns with 7-day constitution requirement
- Cons: Stale entries remain until cron runs

**Option 2: LRU eviction**
- Track last accessed timestamp, evict least recently used entries when cache size exceeds threshold
- Pros: More cache-efficient, keeps popular content longer
- Cons: Added complexity, requires `last_accessed_at` column and update on every read (write amplification)

**Decision**: Use Option 1 (time-based) for MVP. The 7-day expiration is a hard requirement from the constitution, and the additional complexity of LRU doesn't provide significant benefit for the expected usage patterns (10,000 requests/day across 1,000 users = ~10 requests/user/day).

---

## R6: CORS Configuration for Three-Service Architecture

### Context
Three services need to communicate:
- Frontend (Docusaurus on Vercel): `https://yourdomain.com`
- Auth Server (Node.js on Railway): `https://auth.yourdomain.com`
- Backend (Python on Railway): `https://api.yourdomain.com`

### CORS Requirements

**Frontend → Auth Server**
- Operations: POST /api/auth/signup, POST /api/auth/signin, POST /api/auth/refresh, GET /.well-known/jwks.json
- Credentials: Yes (if using cookies; No for localStorage)
- Allowed Headers: Content-Type, Authorization
- Allowed Methods: GET, POST

**Frontend → Backend**
- Operations: POST /api/personalize
- Credentials: No (JWT in Authorization header)
- Allowed Headers: Content-Type, Authorization
- Allowed Methods: POST

**Backend → Auth Server**
- Operations: GET /.well-known/jwks.json (JWKS public key fetch)
- Credentials: No
- Allowed Headers: None
- Allowed Methods: GET

### Auth Server CORS Configuration (Fastify)

```typescript
import cors from '@fastify/cors';

app.register(cors, {
  origin: [
    'https://yourdomain.com',
    'https://www.yourdomain.com',
    'http://localhost:3000'  // Local development
  ],
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: false  // Using localStorage, not cookies
});
```

### Backend CORS Configuration (FastAPI)

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://www.yourdomain.com",
        "http://localhost:3000"  # Local development
    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=False
)
```

### Frontend API Proxy Configuration (Vercel)

**Option A: Direct CORS requests (SELECTED for development)**
- Frontend makes direct fetch() calls to auth.yourdomain.com and api.yourdomain.com
- Pros: Simpler frontend code, clear service boundaries
- Cons: Requires CORS configuration on all services

**Option B: Vercel rewrites (proxy)**
- Frontend calls /api/auth/* and /api/personalize, Vercel proxies to backend services
- Pros: No CORS complexity, single origin from browser perspective
- Cons: Adds latency (extra hop through Vercel), complicates JWT verification (backend sees Vercel IP, not client IP)

**Decision**: Use Option A (direct CORS) for auth and personalization endpoints. The JWKS endpoint (`/.well-known/jwks.json`) is public and doesn't require authentication, so CORS is not a concern.

### Implementation Notes

- **Development**: Use `http://localhost:3000` (Docusaurus dev server), `http://localhost:4000` (auth server), `http://localhost:8000` (backend)
- **Production**: Use custom domains with HTTPS (auth.yourdomain.com, api.yourdomain.com, yourdomain.com)
- **Preflight Requests**: Browsers send OPTIONS requests for POST with custom headers; both Fastify and FastAPI automatically handle preflight
- **Security**: CORS origin list should NOT include wildcard `*` in production (only specific domains)

---

## Summary of Decisions

| Research Task | Decision | Rationale Summary |
|---------------|----------|-------------------|
| **R1: Gemini SDK** | `google-generativeai` (native Python SDK) | Already in use in backend, no OpenAI-compatible Gemini endpoint exists, constitution intent satisfied |
| **R2: Auth Framework** | better-auth + Hono | Batteries-included auth with JWT plugin, JWKS endpoint, user management, custom fields; reduces implementation from ~77 tasks to ~30 tasks |
| **R3: JWT Verification** | PyJWT + cryptography | Simple API, native PyJWKClient for JWKS fetching/caching, most popular Python JWT library (7M+ downloads/month) |
| **R4: Token Storage** | localStorage with CSP | Docusaurus compatibility, persists across sessions (30-day refresh token UX), constitution allows with XSS mitigations |
| **R5: Cache Schema** | Composite unique index on (user_id, content_hash, content_type), JSONB payload, 7-day expiration | Fast lookups, prevents duplicates, flexible metadata storage, GDPR-compliant cascade delete |
| **R6: CORS Config** | Direct CORS requests (allow specific origins), no wildcards | Clear service boundaries, simpler than proxying, secure origin whitelist |

---

## Next Steps

All NEEDS CLARIFICATION items resolved. Proceed to **Phase 1**:
1. Generate `data-model.md` (users and personalized_content table schemas)
2. Generate API contracts in `contracts/` (OpenAPI specs for auth-server and backend endpoints)
3. Generate `quickstart.md` (local development setup instructions)
4. Update agent context with technology choices
