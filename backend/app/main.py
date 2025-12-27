"""
FastAPI application entry point.

This module initializes the FastAPI app with:
- CORS middleware
- Rate limiting
- Lifespan events for database initialization
- API routes
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings
from app.db.database import close_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events - startup and shutdown."""
    # Startup
    try:
        await init_db()
    except Exception as e:
        # Log database connection failure but allow app to start
        print(f"Warning: Database initialization failed: {e}")
        print("API will run without database connectivity")

    yield

    # Shutdown
    try:
        await close_db()
    except Exception as e:
        print(f"Warning: Database cleanup failed: {e}")


# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    enabled=settings.RATE_LIMIT_ENABLED,
)

# Create FastAPI app
app = FastAPI(
    title="Physical AI & Humanoid Robotics API",
    description="Backend API for educational platform with RAG-powered chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Physical AI & Humanoid Robotics API",
        "docs": "/docs",
        "health": "/health",
    }


# Register API routers
from app.api import chat

app.include_router(chat.router, tags=["Chat"])

# TODO: Add remaining routers as they are implemented
# app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
# app.include_router(sessions.router, prefix="/api/sessions", tags=["Sessions"])
# app.include_router(labs.router, prefix="/api/labs", tags=["Labs"])
