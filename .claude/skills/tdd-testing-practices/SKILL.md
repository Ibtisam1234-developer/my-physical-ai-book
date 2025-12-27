---
name: tdd-testing-practices
description: Test-Driven Development (TDD) best practices using pytest with coverage for Python/FastAPI backends and Jest with React Testing Library for frontend components. Includes mocking external services (Gemini, Qdrant), testing auth-protected endpoints, and strict red-green-refactor workflow. Use when writing tests or implementing TDD.
tags: [tdd, testing, pytest, jest, react-testing-library, mocking, coverage]
---

# TDD Testing Practices

## TDD Workflow: Red-Green-Refactor

### The Three Laws of TDD
1. **Red**: Write a failing test that defines desired functionality
2. **Green**: Write the minimal code to make the test pass
3. **Refactor**: Improve code quality while keeping tests green

### Strict TDD Process
```
1. Write failing test (Red)
   └─> Run tests: pytest -q --cov
       └─> Test fails ✗

2. Implement minimal code (Green)
   └─> Run tests: pytest -q --cov
       └─> Test passes ✓

3. Refactor code (Refactor)
   └─> Run tests: pytest -q --cov
       └─> Tests still pass ✓

4. Repeat for next feature
```

**Key Principle**: Never write production code without a failing test first.

## Backend Testing with Pytest

### Setup pytest with Coverage

**Installation**:
```bash
pip install pytest pytest-asyncio pytest-cov pytest-mock httpx
```

**pytest.ini**:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
addopts =
    -v
    --cov=app
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
```

**Run tests**:
```bash
# Run with coverage
pytest -q --cov

# Run with detailed coverage report
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_users.py -v

# Run tests matching pattern
pytest -k "test_create" -v
```

### TDD Example: User Registration

**Step 1: Write Failing Test (Red)**
```python
# tests/test_users.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_create_user_success():
    """Test: User can register with email and password"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/users",
            json={
                "email": "test@example.com",
                "password": "SecurePass123"
            }
        )

    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "id" in data
    assert "hashed_password" not in data  # Don't expose password

# Run: pytest -q --cov
# Result: FAILED (endpoint doesn't exist yet) ✗
```

**Step 2: Implement Minimal Code (Green)**
```python
# app/main.py
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

app = FastAPI()

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: str

@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    # Minimal implementation to make test pass
    hashed_password = hash_password(user_data.password)
    user = User(email=user_data.email, hashed_password=hashed_password)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

# Run: pytest -q --cov
# Result: PASSED ✓
```

**Step 3: Refactor (while keeping tests green)**
```python
# Extract password hashing to service layer
# Add validation
# Improve error handling
# Tests still pass ✓
```

### Database Testing with Fixtures

**conftest.py** (pytest fixtures):
```python
# tests/conftest.py
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.models import Base
from app.main import app
from app.database import get_db

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest_asyncio.fixture
async def test_db():
    """Create test database"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    yield async_session_maker

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture
async def db_session(test_db):
    """Get database session"""
    async with test_db() as session:
        yield session

@pytest_asyncio.fixture
async def client(db_session):
    """HTTP client with test database"""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
```

### Testing with Database Fixtures
```python
# tests/test_users.py
import pytest

@pytest.mark.asyncio
async def test_create_user_duplicate_email(client, db_session):
    """Test: Cannot register with duplicate email"""
    # Arrange: Create first user
    await client.post("/users", json={
        "email": "test@example.com",
        "password": "Pass123"
    })

    # Act: Try to create duplicate
    response = await client.post("/users", json={
        "email": "test@example.com",
        "password": "Pass456"
    })

    # Assert
    assert response.status_code == 400
    assert "already registered" in response.json()["detail"]

@pytest.mark.asyncio
async def test_get_user_by_id(client, db_session):
    """Test: Can retrieve user by ID"""
    # Arrange: Create user
    create_response = await client.post("/users", json={
        "email": "test@example.com",
        "password": "Pass123"
    })
    user_id = create_response.json()["id"]

    # Act: Get user
    response = await client.get(f"/users/{user_id}")

    # Assert
    assert response.status_code == 200
    assert response.json()["email"] == "test@example.com"
```

## Mocking External Services

### Mock Gemini API
```python
# tests/test_rag.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_generate_embedding_with_gemini(client):
    """Test: Generate embedding using Gemini (mocked)"""
    # Arrange: Mock Gemini response
    mock_embedding = [0.1] * 768  # 768-dimensional vector

    with patch("google.generativeai.embed_content") as mock_embed:
        mock_embed.return_value = {"embedding": mock_embedding}

        # Act: Call endpoint that uses Gemini
        response = await client.post("/embeddings", json={
            "text": "What is Physical AI?"
        })

    # Assert
    assert response.status_code == 200
    assert len(response.json()["embedding"]) == 768
    mock_embed.assert_called_once()

@pytest.mark.asyncio
async def test_rag_query_with_mocked_gemini(client):
    """Test: RAG query with mocked Gemini generation"""
    with patch("google.generativeai.GenerativeModel") as mock_model:
        # Mock generate_content method
        mock_instance = AsyncMock()
        mock_instance.generate_content.return_value.text = "Physical AI combines perception, actuation, and learning."
        mock_model.return_value = mock_instance

        # Act
        response = await client.post("/rag/query", json={
            "query": "What is Physical AI?"
        })

        # Assert
        assert response.status_code == 200
        assert "perception" in response.json()["answer"].lower()
```

### Mock Qdrant Client
```python
# tests/test_vector_search.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from qdrant_client.models import ScoredPoint

@pytest.mark.asyncio
async def test_search_documents_mocked_qdrant(client):
    """Test: Search documents with mocked Qdrant"""
    # Arrange: Mock Qdrant search results
    mock_results = [
        ScoredPoint(
            id=1,
            score=0.95,
            payload={
                "source": "docs/intro.md",
                "chunk_text": "Physical AI explanation...",
            },
            vector=None
        ),
        ScoredPoint(
            id=2,
            score=0.87,
            payload={
                "source": "docs/robotics.md",
                "chunk_text": "Robotics fundamentals...",
            },
            vector=None
        )
    ]

    with patch("qdrant_client.QdrantClient") as mock_qdrant:
        mock_client_instance = MagicMock()
        mock_client_instance.search.return_value = mock_results
        mock_qdrant.return_value = mock_client_instance

        # Act
        response = await client.post("/search", json={
            "query": "What is Physical AI?",
            "limit": 5
        })

        # Assert
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 2
        assert results[0]["score"] == 0.95
        assert results[0]["source"] == "docs/intro.md"
```

### Mock Database Operations
```python
# tests/test_chat_sessions.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_create_chat_session_mocked_db():
    """Test: Create chat session with mocked database"""
    with patch("app.database.async_session_maker") as mock_session:
        # Mock session behavior
        mock_db = AsyncMock()
        mock_session.return_value.__aenter__.return_value = mock_db

        # Mock commit and refresh
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        # Act
        response = await client.post("/chat/sessions")

        # Assert
        assert response.status_code == 201
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
```

## Testing Auth-Protected Endpoints

### Setup Authentication Fixtures
```python
# tests/conftest.py
import pytest
from app.auth import create_access_token

@pytest_asyncio.fixture
async def test_user(client):
    """Create test user"""
    response = await client.post("/users", json={
        "email": "test@example.com",
        "password": "TestPass123"
    })
    return response.json()

@pytest_asyncio.fixture
def auth_token(test_user):
    """Generate auth token for test user"""
    token = create_access_token({"sub": test_user["id"]})
    return token

@pytest_asyncio.fixture
def auth_headers(auth_token):
    """Authentication headers"""
    return {"Authorization": f"Bearer {auth_token}"}
```

### Test Protected Endpoints
```python
# tests/test_protected_endpoints.py
import pytest

@pytest.mark.asyncio
async def test_access_protected_endpoint_without_auth(client):
    """Test: Protected endpoint returns 401 without auth"""
    response = await client.get("/me")
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_access_protected_endpoint_with_auth(client, auth_headers):
    """Test: Can access protected endpoint with valid token"""
    response = await client.get("/me", headers=auth_headers)
    assert response.status_code == 200
    assert "email" in response.json()

@pytest.mark.asyncio
async def test_access_protected_endpoint_invalid_token(client):
    """Test: Invalid token returns 401"""
    response = await client.get(
        "/me",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_create_chat_session_requires_auth(client, auth_headers):
    """Test: Creating chat session requires authentication"""
    # Without auth
    response = await client.post("/chat/sessions")
    assert response.status_code == 401

    # With auth
    response = await client.post("/chat/sessions", headers=auth_headers)
    assert response.status_code == 201

@pytest.mark.asyncio
async def test_user_can_only_access_own_sessions(client, auth_headers, test_user):
    """Test: User can only access their own chat sessions"""
    # Create session
    response = await client.post("/chat/sessions", headers=auth_headers)
    session_id = response.json()["id"]

    # Access own session
    response = await client.get(f"/chat/sessions/{session_id}", headers=auth_headers)
    assert response.status_code == 200

    # Try to access another user's session (should fail)
    other_session_id = 9999
    response = await client.get(f"/chat/sessions/{other_session_id}", headers=auth_headers)
    assert response.status_code == 404
```

## Frontend Testing with Jest and React Testing Library

### Setup Jest with React Testing Library

**Installation**:
```bash
npm install --save-dev jest @testing-library/react @testing-library/jest-dom @testing-library/user-event
```

**jest.config.js**:
```javascript
module.exports = {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  moduleNameMapper: {
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    '^@/(.*)$': '<rootDir>/src/$1'
  },
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/**/*.d.ts',
    '!src/main.tsx',
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  }
};
```

**jest.setup.js**:
```javascript
import '@testing-library/jest-dom';
```

**package.json**:
```json
{
  "scripts": {
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage"
  }
}
```

### TDD Example: Login Component

**Step 1: Write Failing Test (Red)**
```typescript
// src/components/LoginForm.test.tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import LoginForm from './LoginForm';

describe('LoginForm', () => {
  it('should render email and password inputs', () => {
    render(<LoginForm />);

    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
  });

  it('should call onSubmit with email and password', async () => {
    const handleSubmit = jest.fn();
    const user = userEvent.setup();

    render(<LoginForm onSubmit={handleSubmit} />);

    await user.type(screen.getByLabelText(/email/i), 'test@example.com');
    await user.type(screen.getByLabelText(/password/i), 'password123');
    await user.click(screen.getByRole('button', { name: /log in/i }));

    expect(handleSubmit).toHaveBeenCalledWith({
      email: 'test@example.com',
      password: 'password123'
    });
  });
});

// Run: npm test
// Result: FAILED (component doesn't exist) ✗
```

**Step 2: Implement Component (Green)**
```typescript
// src/components/LoginForm.tsx
import React, { useState } from 'react';

interface LoginFormProps {
  onSubmit: (data: { email: string; password: string }) => void;
}

export default function LoginForm({ onSubmit }: LoginFormProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ email, password });
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
      </div>
      <div>
        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
      </div>
      <button type="submit">Log In</button>
    </form>
  );
}

// Run: npm test
// Result: PASSED ✓
```

### Mocking API Calls in React
```typescript
// src/components/ChatBot.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ChatBot from './ChatBot';

// Mock fetch
global.fetch = jest.fn();

describe('ChatBot', () => {
  beforeEach(() => {
    (fetch as jest.Mock).mockClear();
  });

  it('should send message and display response', async () => {
    // Mock API response
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        answer: 'Physical AI combines perception, actuation, and learning.'
      })
    });

    const user = userEvent.setup();
    render(<ChatBot />);

    // Type message
    const input = screen.getByPlaceholderText(/ask a question/i);
    await user.type(input, 'What is Physical AI?');
    await user.click(screen.getByRole('button', { name: /send/i }));

    // Wait for response
    await waitFor(() => {
      expect(screen.getByText(/perception, actuation, and learning/i)).toBeInTheDocument();
    });

    // Verify API call
    expect(fetch).toHaveBeenCalledWith('/api/rag/query', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ query: 'What is Physical AI?' })
    }));
  });

  it('should display error message on API failure', async () => {
    // Mock API error
    (fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

    const user = userEvent.setup();
    render(<ChatBot />);

    await user.type(screen.getByPlaceholderText(/ask a question/i), 'Test query');
    await user.click(screen.getByRole('button', { name: /send/i }));

    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
    });
  });
});
```

### Testing Authenticated Components
```typescript
// src/components/ProtectedRoute.test.tsx
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import ProtectedRoute from './ProtectedRoute';
import { AuthContext } from '@/contexts/AuthContext';

describe('ProtectedRoute', () => {
  it('should redirect to login when not authenticated', () => {
    const mockAuthContext = {
      user: null,
      loading: false
    };

    render(
      <AuthContext.Provider value={mockAuthContext}>
        <MemoryRouter>
          <ProtectedRoute>
            <div>Protected Content</div>
          </ProtectedRoute>
        </MemoryRouter>
      </AuthContext.Provider>
    );

    expect(screen.queryByText(/protected content/i)).not.toBeInTheDocument();
  });

  it('should render children when authenticated', () => {
    const mockAuthContext = {
      user: { id: 1, email: 'test@example.com' },
      loading: false
    };

    render(
      <AuthContext.Provider value={mockAuthContext}>
        <MemoryRouter>
          <ProtectedRoute>
            <div>Protected Content</div>
          </ProtectedRoute>
        </MemoryRouter>
      </AuthContext.Provider>
    );

    expect(screen.getByText(/protected content/i)).toBeInTheDocument();
  });
});
```

## Coverage Reports

### Pytest Coverage
```bash
# Generate HTML coverage report
pytest --cov=app --cov-report=html

# View report
open htmlcov/index.html

# Show missing lines in terminal
pytest --cov=app --cov-report=term-missing

# Fail if coverage below 80%
pytest --cov=app --cov-fail-under=80
```

### Jest Coverage
```bash
# Generate coverage report
npm test -- --coverage

# View report
open coverage/lcov-report/index.html

# Coverage summary in terminal
npm test -- --coverage --coverageReporters=text
```

## Best Practices Checklist

### TDD Workflow
- [ ] Always write test first (Red)
- [ ] Write minimal code to pass test (Green)
- [ ] Refactor while keeping tests green
- [ ] Run tests after every change
- [ ] Never skip the refactor step

### Pytest
- [ ] Use `pytest -q --cov` for quick feedback
- [ ] Configure coverage threshold (80%+)
- [ ] Use fixtures for database and client setup
- [ ] Mock external services (Gemini, Qdrant)
- [ ] Test auth-protected endpoints with fixtures
- [ ] Use `@pytest.mark.asyncio` for async tests

### Jest/React Testing Library
- [ ] Use `@testing-library/react` over Enzyme
- [ ] Query by accessibility labels (`getByLabelText`, `getByRole`)
- [ ] Use `userEvent` over `fireEvent`
- [ ] Mock API calls with `jest.fn()`
- [ ] Test loading and error states
- [ ] Clear mocks between tests (`beforeEach`)

### Coverage
- [ ] Aim for 80%+ coverage
- [ ] Focus on critical paths first
- [ ] Test edge cases and error handling
- [ ] Generate HTML reports for visibility
- [ ] Don't chase 100% (diminishing returns)

---

**Usage Note**: Apply strict TDD when developing features. Write failing tests first, implement minimal code to pass, then refactor. Use pytest for backend (FastAPI) and Jest with React Testing Library for frontend. Always mock external services and test auth-protected endpoints with proper fixtures.
