"""
Authentication utilities for Better Auth integration with JWT.
"""

from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

from app.config import settings


# Security scheme for Bearer token
security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    Get current user from JWT token.

    For now, this returns a mock user since Better Auth isn't fully integrated yet.
    In production, this would validate the JWT token and return the actual user.

    Args:
        credentials: HTTP Bearer token credentials

    Returns:
        Dictionary with user information

    Raises:
        HTTPException: If authentication fails
    """
    # For development, return a mock user
    # This allows the chatbot to work without full authentication setup
    return {
        "id": "dev_user_001",
        "email": "dev@example.com",
        "name": "Development User",
        "is_authenticated": True
    }

    # TODO: Implement proper Better Auth JWT validation
    # The code below shows what the full implementation would look like:

    # if not credentials:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Not authenticated",
    #         headers={"WWW-Authenticate": "Bearer"},
    #     )

    # try:
    #     # Decode JWT token
    #     payload = jwt.decode(
    #         credentials.credentials,
    #         settings.JWT_SECRET,
    #         algorithms=[settings.JWT_ALGORITHM]
    #     )

    #     # Extract user information
    #     user_id = payload.get("sub")
    #     if user_id is None:
    #         raise HTTPException(
    #             status_code=status.HTTP_401_UNAUTHORIZED,
    #             detail="Invalid authentication credentials"
    #         )

    #     # Return user info
    #     return {
    #         "id": user_id,
    #         "email": payload.get("email"),
    #         "name": payload.get("name"),
    #         "is_authenticated": True
    #     }

    # except jwt.ExpiredSignatureError:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Token has expired"
    #     )
    # except jwt.JWTError:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Could not validate credentials"
    #     )


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Data to encode in the token
        expires_delta: Optional expiration time delta

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRATION_HOURS)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM
    )

    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT token.

    Args:
        token: JWT token to verify

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


# Example Better Auth integration placeholder
class BetterAuthIntegration:
    """
    Placeholder for Better Auth integration.
    This would be replaced with actual Better Auth SDK integration.
    """

    def __init__(self):
        self.session_storage = {}  # In-memory for development

    def create_session(self, user_id: str) -> str:
        """Create a new session for user."""
        session_id = f"session_{user_id}_{datetime.utcnow().timestamp()}"
        self.session_storage[session_id] = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=24)
        }
        return session_id

    def validate_session(self, session_id: str) -> bool:
        """Validate if session is still valid."""
        if session_id not in self.session_storage:
            return False

        session = self.session_storage[session_id]
        return session["expires_at"] > datetime.utcnow()

    def end_session(self, session_id: str):
        """End a user session."""
        if session_id in self.session_storage:
            del self.session_storage[session_id]


# Global Better Auth integration instance (for development)
better_auth = BetterAuthIntegration()
