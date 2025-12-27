---
name: fastapi-patterns
description: FastAPI best practices and patterns including async endpoints, dependency injection with Depends(), Pydantic v2 models, lifespan events, middleware (CORS, logging), exception handlers, and production deployment with uvicorn. Use when building or modifying FastAPI applications, API endpoints, or backend services.
tags: [fastapi, python, async, pydantic, rest-api, backend]
---

# FastAPI Patterns and Best Practices

## Core Patterns

### Async Endpoints
Always use `async def` for I/O-bound operations (database, external APIs, file operations):

```python
from fastapi import FastAPI, HTTPException
from typing import List

app = FastAPI()

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    """Async endpoint for I/O operations"""
    # Database query, API call, etc.
    item = await database.fetch_one("SELECT * FROM items WHERE id = :id", {"id": item_id})
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@app.get("/items")
async def list_items(skip: int = 0, limit: int = 10) -> List[dict]:
    """List items with pagination"""
    items = await database.fetch_all(
        "SELECT * FROM items LIMIT :limit OFFSET :skip",
        {"skip": skip, "limit": limit}
    )
    return items
```

**When to use `async def` vs `def`**:
- **`async def`**: Database queries, HTTP requests, file I/O, Redis operations
- **`def`**: CPU-bound operations, simple calculations, in-memory operations

## Dependency Injection with Depends()

### Database Dependency
```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Database session dependency"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Endpoint with database dependency"""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

### Authentication Dependency
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Validate JWT token and return current user"""
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return user

@app.get("/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Protected endpoint requiring authentication"""
    return current_user
```

### Dependency Chaining
```python
async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Check if user is active"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_admin_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Check if user is admin"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user

@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Admin-only endpoint"""
    await db.execute(delete(User).where(User.id == user_id))
    await db.commit()
    return {"message": "User deleted"}
```

## Pydantic v2 Models

### Request/Response Models
```python
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    """Request model for user creation"""
    email: str = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v.lower()

class UserResponse(BaseModel):
    """Response model for user data"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    username: str
    created_at: datetime
    is_active: bool = True

    # Password is excluded from response

class UserUpdate(BaseModel):
    """Request model for user updates"""
    email: Optional[str] = None
    username: Optional[str] = None
    is_active: Optional[bool] = None

@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create new user with Pydantic v2 validation"""
    # Hash password
    hashed_password = hash_password(user_data.password)

    # Create user
    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user
```

### Pydantic v2 Configuration
```python
from pydantic import BaseModel, ConfigDict

class ItemBase(BaseModel):
    """Base model with Pydantic v2 configuration"""
    model_config = ConfigDict(
        from_attributes=True,      # Enable ORM mode (was orm_mode in v1)
        use_enum_values=True,       # Use enum values instead of enum objects
        validate_assignment=True,   # Validate on assignment
        str_strip_whitespace=True,  # Strip whitespace from strings
        json_schema_extra={         # Add OpenAPI schema examples
            "example": {
                "name": "Example Item",
                "price": 29.99
            }
        }
    )

    name: str
    price: float = Field(..., gt=0, description="Item price in USD")
```

## Lifespan Events (Startup/Shutdown)

### Modern Lifespan Context Manager
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    print("Starting up...")

    # Initialize database connection pool
    await database.connect()

    # Initialize Redis connection
    await redis.connect()

    # Load ML models
    app.state.model = load_model("model.pkl")

    # Start background tasks
    app.state.background_tasks = start_background_tasks()

    yield  # Application is running

    # Shutdown
    print("Shutting down...")

    # Close database connections
    await database.disconnect()

    # Close Redis connection
    await redis.disconnect()

    # Stop background tasks
    await app.state.background_tasks.stop()

app = FastAPI(lifespan=lifespan)

@app.get("/predict")
async def predict(data: dict):
    """Use model loaded during startup"""
    prediction = app.state.model.predict(data)
    return {"prediction": prediction}
```

### Legacy Startup/Shutdown Events (Pre-0.100)
```python
# Only use if stuck on older FastAPI version
@app.on_event("startup")
async def startup_event():
    """Legacy startup event"""
    await database.connect()

@app.on_event("shutdown")
async def shutdown_event():
    """Legacy shutdown event"""
    await database.disconnect()
```

## Middleware

### CORS Middleware
```python
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",           # Development frontend
        "https://your-domain.com",         # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],                   # Or specify: ["GET", "POST", "PUT", "DELETE"]
    allow_headers=["*"],                   # Or specify headers
    expose_headers=["X-Total-Count"],      # Headers accessible to frontend
)
```

### Logging Middleware
```python
import time
import logging
from fastapi import Request

logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url}")

    # Process request
    response = await call_next(request)

    # Calculate duration
    process_time = time.time() - start_time

    # Log response
    logger.info(
        f"Response: {response.status_code} "
        f"Duration: {process_time:.3f}s "
        f"Path: {request.url.path}"
    )

    # Add custom header
    response.headers["X-Process-Time"] = str(process_time)

    return response
```

### Authentication Middleware
```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class AuthMiddleware(BaseHTTPMiddleware):
    """Custom authentication middleware"""

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public endpoints
        if request.url.path in ["/docs", "/openapi.json", "/health"]:
            return await call_next(request)

        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if not api_key or not await validate_api_key(api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Add user info to request state
        request.state.user = await get_user_by_api_key(api_key)

        return await call_next(request)

app.add_middleware(AuthMiddleware)
```

## Exception Handlers

### Custom Exception Handlers
```python
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import IntegrityError

app = FastAPI()

# Custom exception
class ItemNotFoundException(Exception):
    def __init__(self, item_id: int):
        self.item_id = item_id

# Handler for custom exception
@app.exception_handler(ItemNotFoundException)
async def item_not_found_handler(request: Request, exc: ItemNotFoundException):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Item not found",
            "item_id": exc.item_id,
            "path": request.url.path
        }
    )

# Handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "details": exc.errors(),
            "body": exc.body
        }
    )

# Handler for database integrity errors
@app.exception_handler(IntegrityError)
async def integrity_error_handler(request: Request, exc: IntegrityError):
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "error": "Database integrity error",
            "detail": "Resource already exists or constraint violated"
        }
    )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )
```

## Production Deployment with Uvicorn

### Installation
```bash
# Install uvicorn with gunicorn for production
pip install "uvicorn[standard]" gunicorn
```

### Production Server Configuration

**Option 1: Uvicorn with Gunicorn (Recommended)**
```bash
# Start with Gunicorn managing multiple Uvicorn workers
gunicorn main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
```

**Option 2: Uvicorn Standalone**
```bash
# Start single Uvicorn process (good for development)
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info \
    --access-log \
    --proxy-headers
```

### Production Settings
```python
# main.py
from fastapi import FastAPI
import logging

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Physical AI API",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,  # Disable docs in production
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0"
    }
```

### Docker Configuration
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use gunicorn with uvicorn workers
CMD ["gunicorn", "main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
```

### Worker Count Calculation
```python
# Recommended: (2 x CPU cores) + 1
import multiprocessing

workers = (2 * multiprocessing.cpu_count()) + 1
print(f"Recommended workers: {workers}")
```

## Complete Application Structure

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession
import logging

logger = logging.getLogger(__name__)

# Lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    await database.connect()
    yield
    logger.info("Shutting down...")
    await database.disconnect()

# Create app
app = FastAPI(lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"{request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Status: {response.status_code}")
    return response

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# Dependency
async def get_db() -> AsyncSession:
    async with async_session_maker() as session:
        yield session

# Pydantic models
class ItemCreate(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    name: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)

class ItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    price: float

# Endpoints
@app.post("/items", response_model=ItemResponse, status_code=201)
async def create_item(
    item: ItemCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create new item"""
    # Implementation
    pass

@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(
    item_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get item by ID"""
    # Implementation
    pass
```

## Best Practices Checklist

- [ ] Use `async def` for all I/O-bound endpoints
- [ ] Implement `Depends()` for database sessions and authentication
- [ ] Use Pydantic v2 models with `ConfigDict(from_attributes=True)`
- [ ] Configure lifespan events for startup/shutdown
- [ ] Add CORS middleware with specific origins
- [ ] Implement logging middleware for request tracking
- [ ] Create custom exception handlers for common errors
- [ ] Deploy with `gunicorn` + `uvicorn.workers.UvicornWorker`
- [ ] Configure appropriate worker count: `(2 x CPU) + 1`
- [ ] Disable docs in production (`docs_url=None`)
- [ ] Add health check endpoint
- [ ] Use proper logging configuration
- [ ] Implement proper error responses with status codes

---

**Usage Note**: Apply these patterns when building FastAPI applications. Always use async patterns for I/O operations, leverage dependency injection for clean architecture, and deploy with appropriate worker configuration for production workloads.
