# Quickstart Guide: Authentication & Personalization System

**Date**: 2025-12-27
**Branch**: `001-auth-personalization`
**Estimated Setup Time**: 30 minutes

This guide walks through local development setup for the three-service authentication and personalization system.

---

## Prerequisites

- **Node.js**: v20.x LTS ([download](https://nodejs.org/))
- **Python**: 3.11+ ([download](https://www.python.org/downloads/))
- **PostgreSQL**: 14+ (or Neon database connection string)
- **Git**: Latest version
- **Text Editor**: VS Code recommended

### Required Environment Variables

Create three `.env` files:

**auth-server/.env**:
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/physical_ai_dev

# JWT Signing (generate RSA key pair - see "Key Generation" section below)
JWT_PRIVATE_KEY_PATH=./keys/private.pem
JWT_PUBLIC_KEY_PATH=./keys/public.pem

# Server
PORT=4000
NODE_ENV=development

# CORS (Docusaurus dev server)
ALLOWED_ORIGINS=http://localhost:3000
```

**backend/.env** (update existing):
```bash
# Existing env vars...

# JWT Verification
JWKS_URL=http://localhost:4000/.well-known/jwks.json

# Gemini API (already configured)
GEMINI_API_KEY=your_gemini_api_key_here
```

**frontend/.env** (update existing):
```bash
# API Endpoints
REACT_APP_AUTH_SERVER_URL=http://localhost:4000
REACT_APP_BACKEND_URL=http://localhost:8000
```

---

## Step 1: Clone Repository & Checkout Branch

```bash
# If not already in repository
cd my-physical-ai-book

# Checkout feature branch
git checkout 001-auth-personalization

# Pull latest changes
git pull origin 001-auth-personalization
```

---

## Step 2: Database Setup

### Option A: Local PostgreSQL

```bash
# Create development database
createdb physical_ai_dev

# Run migration script
psql physical_ai_dev -f specs/001-auth-personalization/migrations/001_initial_schema.sql

# Verify tables created
psql physical_ai_dev -c "\dt"
# Expected output: users, personalized_content
```

### Option B: Neon Serverless PostgreSQL

1. Create Neon project: [https://neon.tech/](https://neon.tech/)
2. Create branch `dev` from `main`
3. Copy connection string
4. Run migration via `psql`:
   ```bash
   psql "postgresql://user:password@ep-xxx.neon.tech/physical_ai_dev?sslmode=require" \
     -f specs/001-auth-personalization/migrations/001_initial_schema.sql
   ```

---

## Step 3: Auth Server Setup (Node.js)

### 3.1: Generate RSA Key Pair

```bash
cd auth-server

# Create keys directory
mkdir -p keys

# Generate private key (RS256, 2048-bit)
openssl genrsa -out keys/private.pem 2048

# Extract public key
openssl rsa -in keys/private.pem -pubout -out keys/public.pem

# Verify key generation
ls -lh keys/
# Expected output: private.pem (1.7KB), public.pem (450 bytes)
```

**Security Note**: Add `keys/` to `.gitignore`. Never commit private keys to version control.

### 3.2: Install Dependencies

```bash
# Install Node.js packages
npm install

# Expected packages:
# - fastify@^4.25.0
# - @fastify/cors
# - @fastify/jwt
# - jose@^5.1.0
# - bcrypt@^5.1.1
# - pg@^8.11.0
# - @types/node, @types/pg (dev dependencies)
```

### 3.3: Run Development Server

```bash
# Start auth server
npm run dev

# Expected output:
# Server listening at http://localhost:4000
# JWKS endpoint available at http://localhost:4000/.well-known/jwks.json
```

### 3.4: Verify Auth Server

```bash
# Test JWKS endpoint
curl http://localhost:4000/.well-known/jwks.json

# Expected response:
# {
#   "keys": [
#     {
#       "kty": "RSA",
#       "use": "sig",
#       "kid": "auth-server-key-2025",
#       "alg": "RS256",
#       "n": "xGOr-H7A...",
#       "e": "AQAB"
#     }
#   ]
# }

# Test signup endpoint
curl -X POST http://localhost:4000/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{
    "email": "testuser@example.com",
    "password": "TestPass123!",
    "softwareBackground": "intermediate",
    "hardwareBackground": "hobbyist"
  }'

# Expected response:
# {
#   "accessToken": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
#   "refreshToken": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
#   "expiresIn": 3600,
#   "user": {
#     "id": "123e4567-e89b-12d3-a456-426614174000",
#     "email": "testuser@example.com",
#     "softwareBackground": "intermediate",
#     "hardwareBackground": "hobbyist"
#   }
# }
```

---

## Step 4: Backend Setup (Python FastAPI)

### 4.1: Install Dependencies

```bash
cd ../backend

# Create virtual environment (if not already created)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install updated requirements
pip install -r requirements.txt

# Expected new packages:
# - python-jose[cryptography]
# - httpx
# (google-generativeai already installed)
```

### 4.2: Run Development Server

```bash
# Start backend server
uvicorn app.main:app --reload --port 8000

# Expected output:
# INFO:     Uvicorn running on http://localhost:8000
# INFO:     Application startup complete
# INFO:     JWKS cache initialized from http://localhost:4000/.well-known/jwks.json
```

### 4.3: Verify Backend

```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "timestamp": "2025-12-27T12:34:56Z",
#   "dependencies": {
#     "database": "healthy",
#     "jwksCache": "healthy",
#     "geminiApi": "healthy"
#   }
# }

# Test personalization endpoint (requires JWT from auth server)
# 1. Get JWT token from signup/signin (see Step 3.4)
# 2. Use token in Authorization header:

curl -X POST http://localhost:8000/api/personalize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN_HERE" \
  -d '{
    "chapterMarkdown": "# ROS 2 Navigation\n\nThe ROS 2 Navigation Stack (Nav2) is a collection of packages that enable autonomous navigation for mobile robots. It provides sophisticated algorithms for path planning, obstacle avoidance, and localization...",
    "contentType": "curriculum_path"
  }'

# Expected response (cache miss, first request):
# {
#   "personalizedText": "For someone with intermediate software skills and hobbyist hardware experience, the ROS 2 Navigation Stack...",
#   "model": "gemini-2.0-flash-exp",
#   "tokens": 856,
#   "cacheHit": false,
#   "generatedAt": "2025-12-27T12:34:56Z",
#   "expiresAt": "2026-01-03T12:34:56Z"
# }

# Repeat same request to test cache:
# Expected response (cache hit, <10ms):
# { ..., "cacheHit": true, "tokens": null }
```

---

## Step 5: Frontend Setup (Docusaurus)

### 5.1: Install Dependencies

```bash
cd ../frontend

# Install Node.js packages
npm install

# Expected new packages (to be added to package.json):
# - axios or native fetch (no additional dependency needed)
```

### 5.2: Update Docusaurus Configuration

Edit `docusaurus.config.js` to add CORS proxy (temporary for local dev):

```javascript
module.exports = {
  // ... existing config

  // Add Content Security Policy for XSS protection
  headTags: [
    {
      tagName: 'meta',
      attributes: {
        'http-equiv': 'Content-Security-Policy',
        content: "default-src 'self'; script-src 'self' 'unsafe-inline'; connect-src 'self' http://localhost:4000 http://localhost:8000"
      }
    }
  ],

  // ... rest of config
};
```

### 5.3: Run Development Server

```bash
# Start Docusaurus dev server
npm start

# Expected output:
# [INFO] Starting the development server...
# [SUCCESS] Docusaurus website is running at http://localhost:3000/
```

### 5.4: Verify Frontend

1. Open browser: `http://localhost:3000`
2. Navigate to any documentation chapter
3. Look for "Personalize for Me" button (to be implemented in tasks phase)
4. Click "Sign Up" (to be implemented)
5. Fill form with background fields
6. Submit and verify JWT token stored in localStorage

---

## Step 6: End-to-End Test

### 6.1: User Signup Flow

1. **Frontend**: Open `http://localhost:3000/signup`
2. **Fill Form**:
   - Email: `e2e-test@example.com`
   - Password: `TestPass123!`
   - Software Background: `intermediate`
   - Hardware Background: `hobbyist`
3. **Submit**: Form sends POST to `http://localhost:4000/api/auth/signup`
4. **Response**: JWT tokens stored in `localStorage`:
   - `auth_access_token`
   - `auth_refresh_token`
   - `auth_user` (user profile JSON)

### 6.2: Content Personalization Flow

1. **Frontend**: Navigate to `/docs/ros2-navigation` (example chapter)
2. **Click**: "Personalize for Me" button
3. **Request Flow**:
   - Frontend reads `auth_access_token` from localStorage
   - Sends POST to `http://localhost:8000/api/personalize` with:
     - Authorization: `Bearer <access_token>`
     - Body: `{ chapterMarkdown: "...", contentType: "curriculum_path" }`
4. **Backend Processing**:
   - Verifies JWT signature using JWKS public key (cached from auth server)
   - Extracts `software_background: "intermediate"` and `hardware_background: "hobbyist"` from JWT claims
   - Computes SHA-256 hash of chapter markdown: `a3d2e1f4...`
   - Cache lookup: `SELECT content_payload FROM personalized_content WHERE user_id=$1 AND content_hash=$2`
   - Cache miss (first request): Generate personalized content via Gemini LLM
   - Insert into cache: `INSERT INTO personalized_content (...) VALUES (...)`
5. **Response**: Personalized content displayed in modal/inline
6. **Repeat Request**: Cache hit (response <10ms, no LLM API call)

### 6.3: Token Refresh Flow

1. **Scenario**: Access token expires after 1 hour
2. **Frontend**: Automatic token refresh before API call
3. **Request**: POST to `http://localhost:4000/api/auth/refresh`
   - Body: `{ refreshToken: "..." }`
4. **Response**: New access token (1 hour expiration)
5. **Frontend**: Update `auth_access_token` in localStorage
6. **Retry**: Original API request with new token

---

## Troubleshooting

### Issue: Auth server fails to start

**Symptom**: `Error: ENOENT: no such file or directory, open './keys/private.pem'`

**Solution**: Generate RSA key pair (see Step 3.1)

---

### Issue: Backend cannot fetch JWKS

**Symptom**: `INFO: JWKS cache initialization failed: Connection refused`

**Solution**:
1. Ensure auth server is running on port 4000
2. Verify `JWKS_URL` in backend `.env` points to `http://localhost:4000/.well-known/jwks.json`
3. Test JWKS endpoint manually: `curl http://localhost:4000/.well-known/jwks.json`

---

### Issue: JWT verification fails

**Symptom**: `401 Unauthorized: Invalid token signature`

**Solution**:
1. Ensure backend fetched latest JWKS (restart backend after auth server key regeneration)
2. Verify JWT token is not expired (check `exp` claim): `jwt.io` decoder
3. Check auth server logs for JWT signing errors

---

### Issue: CORS errors in browser console

**Symptom**: `Access to fetch at 'http://localhost:4000/api/auth/signup' from origin 'http://localhost:3000' has been blocked by CORS policy`

**Solution**:
1. Verify auth server CORS configuration includes `http://localhost:3000` in `ALLOWED_ORIGINS`
2. Restart auth server after `.env` changes
3. Check browser Network tab for preflight OPTIONS request (should return 200 OK)

---

### Issue: Personalization cache not working

**Symptom**: Every personalization request hits LLM API (no cache hits)

**Solution**:
1. Verify `personalized_content` table exists: `psql $DATABASE_URL -c "\dt"`
2. Check unique constraint: `psql $DATABASE_URL -c "\d personalized_content"`
3. Query cache manually:
   ```sql
   SELECT user_id, content_hash, expires_at, content_payload->'cacheHit'
   FROM personalized_content
   WHERE user_id = 'YOUR_USER_ID';
   ```
4. Ensure SHA-256 hash is computed correctly (same chapter markdown should produce same hash)

---

## Next Steps

After verifying local development setup:

1. **Run Tests**: `npm test` (auth-server), `pytest` (backend), `npm test` (frontend)
2. **Code Coverage**: Ensure >80% coverage (per Constitution Principle II)
3. **Deploy to Staging**: Railway (auth-server, backend), Vercel (frontend)
4. **Integration Testing**: Test cross-service communication on staging
5. **Production Deployment**: Requires human approval (per Constitution Principle VI)

---

## Development Workflow

### Daily Development

```bash
# Terminal 1: Auth server
cd auth-server
npm run dev

# Terminal 2: Backend
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
uvicorn app.main:app --reload --port 8000

# Terminal 3: Frontend
cd frontend
npm start

# Terminal 4: Database (if local PostgreSQL)
psql physical_ai_dev
```

### Git Workflow

```bash
# Create feature branch for specific task (e.g., "implement signup endpoint")
git checkout -b 001-auth-personalization-signup

# Make changes, commit frequently
git add auth-server/src/routes/signup.ts
git commit -m "feat(auth): implement signup endpoint with background validation"

# Push to remote
git push origin 001-auth-personalization-signup

# Create PR targeting 001-auth-personalization branch
gh pr create --base 001-auth-personalization --title "Implement signup endpoint"
```

### Testing Commands

```bash
# Auth server tests
cd auth-server
npm test                    # Run all tests
npm run test:watch          # Watch mode
npm run test:coverage       # Generate coverage report

# Backend tests
cd backend
pytest                      # Run all tests
pytest --cov=app            # With coverage
pytest -k test_personalize  # Run specific test

# Frontend tests
cd frontend
npm test                    # Run all tests
npm test -- --coverage      # With coverage
```

---

## Resources

- **OpenAPI Specs**:
  - Auth API: `specs/001-auth-personalization/contracts/auth-api.yaml`
  - Personalization API: `specs/001-auth-personalization/contracts/personalization-api.yaml`
- **Data Model**: `specs/001-auth-personalization/data-model.md`
- **Research Decisions**: `specs/001-auth-personalization/research.md`
- **Implementation Plan**: `specs/001-auth-personalization/plan.md`

- **External Documentation**:
  - Fastify: https://fastify.dev/docs/latest/
  - jose (JWT library): https://github.com/panva/jose
  - FastAPI: https://fastapi.tiangolo.com/
  - python-jose: https://python-jose.readthedocs.io/
  - Docusaurus: https://docusaurus.io/docs
  - Neon PostgreSQL: https://neon.tech/docs
