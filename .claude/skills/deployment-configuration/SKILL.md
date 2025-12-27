---
name: deployment-configuration
description: Deployment configuration for Vercel (Docusaurus frontend), Railway (FastAPI backend with Neon PostgreSQL), environment variable management, and PR preview environments. Use when deploying applications, configuring CI/CD, or setting up preview deployments.
tags: [deployment, vercel, railway, ci-cd, environment-variables, previews]
---

# Deployment Configuration

## Overview

**Deployment Architecture**:
- **Frontend**: Vercel (Docusaurus/React)
- **Backend**: Railway (FastAPI + Gunicorn + Uvicorn)
- **Database**: Neon PostgreSQL (serverless)
- **Vector DB**: Qdrant (self-hosted or cloud)

**Key Features**:
- Git-based deployments (push to deploy)
- Automatic PR previews
- Environment variable management
- Zero-downtime deployments

## Vercel Deployment (Frontend)

### Initial Setup

**1. Connect Git Repository**:
```bash
# Install Vercel CLI (optional)
npm install -g vercel

# Login
vercel login

# Link project
vercel link
```

**Via Vercel Dashboard**:
1. Go to https://vercel.com/new
2. Import Git repository
3. Select your repository
4. Configure project settings

**2. Project Configuration**:

**vercel.json**:
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "build",
  "installCommand": "npm install",
  "framework": "docusaurus",
  "devCommand": "npm start",
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://your-backend.railway.app/api/:path*"
    }
  ]
}
```

**3. Root Directory Configuration**:

If your frontend is in a subdirectory:
- Go to Project Settings → General
- Set **Root Directory** to `frontend` or `docs` (path from repo root)
- Vercel will only build from this directory

**4. Build Settings**:
- **Framework Preset**: Docusaurus (auto-detected)
- **Build Command**: `npm run build` or `docusaurus build`
- **Output Directory**: `build` (Docusaurus default)
- **Install Command**: `npm install`
- **Node Version**: 18.x or 20.x (set in package.json or settings)

**package.json**:
```json
{
  "engines": {
    "node": ">=18.0.0"
  }
}
```

### Environment Variables (Vercel)

**Add via Dashboard**:
1. Project Settings → Environment Variables
2. Add variables for each environment:
   - Production
   - Preview
   - Development

**Common Frontend Environment Variables**:
```bash
# API Configuration
VITE_API_URL=https://your-backend.railway.app
VITE_API_TIMEOUT=30000

# Authentication (Better Auth)
AUTH_SECRET=your-secret-key-here
AUTH_URL=https://your-site.vercel.app

# Analytics (optional)
NEXT_PUBLIC_ANALYTICS_ID=your-analytics-id

# Feature Flags
VITE_ENABLE_CHATBOT=true
```

**Accessing in Code**:
```typescript
// Vite projects
const apiUrl = import.meta.env.VITE_API_URL;

// Next.js projects
const apiUrl = process.env.NEXT_PUBLIC_API_URL;
```

### Vercel CLI Commands
```bash
# Deploy to production
vercel --prod

# Deploy preview
vercel

# List deployments
vercel ls

# View logs
vercel logs <deployment-url>

# Environment variables
vercel env add VITE_API_URL
vercel env pull  # Download to .env.local
```

### Custom Domains

**Add Domain**:
1. Project Settings → Domains
2. Add custom domain: `your-site.com`
3. Configure DNS (Vercel provides instructions)
4. Add SSL certificate (automatic with Vercel)

**DNS Configuration**:
```
Type: A
Name: @
Value: 76.76.21.21

Type: CNAME
Name: www
Value: cname.vercel-dns.com
```

## Railway Deployment (Backend)

### Initial Setup

**1. Create Railway Account**:
- Sign up at https://railway.app
- Connect GitHub account

**2. Create New Project**:
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Link to existing project
railway link
```

**3. Deploy from Git**:
- Dashboard → New Project → Deploy from GitHub repo
- Select repository
- Railway auto-detects FastAPI/Python

**4. Project Configuration**:

**railway.json** (optional):
```json
{
  "build": {
    "builder": "NIXPACKS",
    "buildCommand": "pip install -r requirements.txt"
  },
  "deploy": {
    "startCommand": "gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

**Procfile** (alternative):
```
web: gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
```

**nixpacks.toml** (Railway's build system):
```toml
[phases.setup]
nixPkgs = ["python39", "postgresql"]

[phases.install]
cmds = ["pip install -r requirements.txt"]

[start]
cmd = "gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT"
```

### Requirements for Production

**requirements.txt**:
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
gunicorn==21.2.0
sqlalchemy[asyncio]==2.0.25
asyncpg==0.29.0
psycopg2-binary==2.9.9
qdrant-client==1.7.0
google-generativeai==0.3.2
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
pydantic==2.5.3
pydantic-settings==2.1.0
```

**runtime.txt** (specify Python version):
```
python-3.11.7
```

### Environment Variables (Railway)

**Add via Dashboard**:
1. Project → Variables
2. Add variables (available to all services)

**Common Backend Environment Variables**:
```bash
# Database (from Neon)
DATABASE_URL=postgresql://user:pass@host.neon.tech/db?sslmode=require

# Gemini API
GEMINI_API_KEY=your-gemini-api-key

# Qdrant
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key

# Authentication
JWT_SECRET_KEY=your-jwt-secret
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS Origins
CORS_ORIGINS=https://your-site.vercel.app,https://www.your-site.com

# App Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Railway-specific (auto-provided)
PORT=8000  # Railway sets this automatically
RAILWAY_ENVIRONMENT=production
```

**Access in Code**:
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    gemini_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    jwt_secret_key: str
    cors_origins: list[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"

settings = Settings()
```

### Railway CLI Commands
```bash
# Deploy
railway up

# View logs
railway logs

# Open in browser
railway open

# Environment variables
railway variables
railway variables set KEY=value

# Connect to database
railway connect
```

## Neon Database Integration

### Create Neon Database

**1. Create Database**:
- Sign up at https://neon.tech
- Create new project
- Note connection string

**2. Get Connection String**:
```
postgresql://user:password@ep-xyz.neon.tech/dbname?sslmode=require
```

**3. Add to Railway**:
- Railway Project → Variables
- Add `DATABASE_URL` with Neon connection string

**4. Database Migrations**:

**Alembic Setup**:
```bash
# Install
pip install alembic

# Initialize
alembic init alembic

# Configure alembic.ini
sqlalchemy.url = postgresql+asyncpg://...

# Create migration
alembic revision --autogenerate -m "Initial schema"

# Apply migration
alembic upgrade head
```

**Add to Deployment**:
```bash
# Railway start command
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --preload
```

**Startup script (migrate.sh)**:
```bash
#!/bin/bash
# Run migrations before starting server
alembic upgrade head
exec gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
```

**Update Procfile**:
```
web: bash migrate.sh
```

### Neon Branching

**Create Database Branch** (for preview environments):
```bash
# Neon CLI
neonctl branches create --name preview-123

# Get branch connection string
neonctl connection-string preview-123
```

## PR Preview Environments

### Vercel PR Previews (Automatic)

**Enable in Settings**:
1. Project Settings → Git
2. Enable **Automatic Deployments from GitHub**
3. Enable **Preview Deployments** for all branches

**How it Works**:
- Every PR gets a unique preview URL
- Preview URL: `your-project-git-branch-username.vercel.app`
- Comments on PR with preview link
- Updates automatically on new commits

**Configure Preview Environment Variables**:
- Project Settings → Environment Variables
- Select **Preview** environment
- Add preview-specific variables:
  ```bash
  VITE_API_URL=https://preview-api-$RAILWAY_PR_ID.railway.app
  ```

### Railway PR Previews

**Enable in Settings**:
1. Project Settings → Environments
2. Create **PR Environment** template
3. Enable **Deploy PR Environments**

**How it Works**:
- Each PR creates ephemeral environment
- Unique URL per PR
- Tears down when PR merged/closed

**PR Environment Variables**:
```bash
# Railway provides
RAILWAY_PR_ID=123
RAILWAY_BRANCH_NAME=feature/new-endpoint

# Set dynamic DATABASE_URL for PR branch
DATABASE_URL=postgresql://...preview-branch...
```

**GitHub Actions for PR Previews**:

**.github/workflows/preview.yml**:
```yaml
name: Deploy PR Preview

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  deploy-preview:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to Railway
        run: |
          railway link ${{ secrets.RAILWAY_PROJECT_ID }}
          railway up --environment pr-${{ github.event.pull_request.number }}
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}

      - name: Comment PR with URL
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '🚀 Preview deployed to https://preview-${{ github.event.pull_request.number }}.railway.app'
            })
```

## Complete Deployment Workflow

### 1. Initial Setup

**Frontend (Vercel)**:
```bash
# Push to GitHub
git push origin main

# Vercel auto-deploys
# Visit: https://your-project.vercel.app
```

**Backend (Railway)**:
```bash
# Push to GitHub
git push origin main

# Railway auto-deploys
# Visit: https://your-project.railway.app
```

### 2. Environment Variables

**Vercel**:
```bash
vercel env add VITE_API_URL production
vercel env add AUTH_SECRET production
```

**Railway**:
```bash
railway variables set DATABASE_URL="postgresql://..."
railway variables set GEMINI_API_KEY="your-key"
railway variables set QDRANT_URL="https://..."
```

### 3. Connect Services

**Update Frontend API URL**:
```typescript
// src/lib/api.ts
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function fetchData(endpoint: string) {
  const response = await fetch(`${API_URL}${endpoint}`);
  return response.json();
}
```

**Update Backend CORS**:
```python
# main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-site.vercel.app",
        "https://your-site.com",
        "http://localhost:3000"  # Development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 4. PR Workflow

**Developer creates PR**:
```bash
git checkout -b feature/new-endpoint
git push origin feature/new-endpoint
# Create PR on GitHub
```

**Automatic Actions**:
1. ✅ Vercel deploys preview: `your-project-git-feature-new-endpoint.vercel.app`
2. ✅ Railway deploys preview: `feature-new-endpoint-pr-123.railway.app`
3. ✅ GitHub bot comments with URLs
4. ✅ Team reviews using preview links
5. ✅ Merge PR → auto-deploy to production
6. ✅ Preview environments destroyed

## Security Best Practices

### Environment Variables
- ✅ Never commit secrets to Git
- ✅ Use different secrets for production/preview
- ✅ Rotate secrets periodically
- ✅ Use secret management (Vercel/Railway encrypted storage)
- ✅ Limit access to production secrets

### Example .env.example (commit this)
```bash
# Database
DATABASE_URL=postgresql://user:pass@host/db

# API Keys (replace with your keys)
GEMINI_API_KEY=your-key-here
QDRANT_API_KEY=your-key-here

# Authentication
JWT_SECRET_KEY=generate-secure-random-string

# CORS
CORS_ORIGINS=https://your-site.com,http://localhost:3000
```

### .gitignore
```
.env
.env.local
.env.production
.env.*.local
railway.json
```

## Monitoring and Logs

### Vercel Logs
```bash
# View deployment logs
vercel logs <deployment-url>

# Real-time logs
vercel logs --follow

# Filter logs
vercel logs --since 1h
```

### Railway Logs
```bash
# View logs
railway logs

# Follow logs
railway logs --follow

# Filter by service
railway logs --service backend
```

### Health Check Endpoints

**Backend**:
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": os.getenv("RAILWAY_ENVIRONMENT")
    }
```

**Frontend (public/health.json)**:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## Deployment Checklist

### Pre-Deployment
- [ ] All tests passing (`pytest --cov`, `npm test`)
- [ ] Environment variables configured for all environments
- [ ] CORS configured with production domains
- [ ] Database migrations ready (`alembic upgrade head`)
- [ ] Build succeeds locally
- [ ] No secrets in Git history

### Vercel Setup
- [ ] Repository connected
- [ ] Root directory configured (if needed)
- [ ] Environment variables added (production, preview)
- [ ] Build settings verified
- [ ] Custom domain configured (optional)
- [ ] PR previews enabled

### Railway Setup
- [ ] Repository connected
- [ ] Environment variables added
- [ ] Database connection string configured
- [ ] Start command configured (Gunicorn + Uvicorn)
- [ ] Health check endpoint working
- [ ] PR environments enabled

### Post-Deployment
- [ ] Production site loads correctly
- [ ] API endpoints responding
- [ ] Database connection working
- [ ] Authentication working
- [ ] External services connected (Gemini, Qdrant)
- [ ] Monitoring/logs configured
- [ ] Custom domains working (if applicable)

## Troubleshooting

### Vercel Build Failures
```bash
# Check build logs
vercel logs <deployment-url>

# Common issues:
# - Wrong Node version → Set in package.json
# - Missing environment variables → Add in settings
# - Build command failed → Check package.json scripts
```

### Railway Deployment Failures
```bash
# Check logs
railway logs

# Common issues:
# - Port binding → Use $PORT from Railway
# - Database connection → Check DATABASE_URL
# - Missing dependencies → Update requirements.txt
# - Build timeout → Optimize Docker build
```

### Database Connection Issues
```python
# Test connection
import asyncpg

async def test_connection():
    conn = await asyncpg.connect(DATABASE_URL)
    version = await conn.fetchval('SELECT version()')
    print(version)
    await conn.close()
```

---

**Usage Note**: Apply these deployment patterns when deploying full-stack applications to Vercel (frontend) and Railway (backend) with Neon PostgreSQL. Always configure environment variables securely, enable PR previews for team collaboration, and monitor deployments with health checks and logs.
