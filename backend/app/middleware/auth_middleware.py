"""
Authentication middleware using Better Auth integration with RS256 JWT verification.

This middleware validates JWT tokens using the auth server's JWKS endpoint,
ensuring secure cross-service authentication.
"""

import re
from typing import Optional, Callable, Awaitable
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from app.utils.jwt_verifier import (
    verify_token,
    extract_token_from_header,
    JWTVerificationError,
    TokenExpiredError,
    InvalidTokenError,
)


from starlette.middleware.base import BaseHTTPMiddleware

class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for validating Better Auth JWT tokens.

    Uses RS256 asymmetric keys fetched from the auth server's JWKS endpoint.
    """

    def __init__(
        self,
        app,
        exclude_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/",
        ]

    async def dispatch(self, request: Request, call_next):
        """
        Process incoming request with authentication check.
        """
        # Handle CORS preflight requests (OPTIONS)
        if request.method == "OPTIONS":
            response = await call_next(request)
            return response

        # Skip auth for excluded paths
        if request.url.path in self.exclude_paths:
            response = await call_next(request)
            return response

        # Extract JWT token from Authorization header
        auth_header = request.headers.get("Authorization")
        token = extract_token_from_header(auth_header)

        if not token:
            # For development, allow requests without authentication
            # and attach a mock user for protected endpoints
            if self._is_protected_endpoint(request.url.path):
                # Attach a mock user for development
                request.state.user = {
                    "sub": "dev_user_001",
                    "email": "dev@example.com",
                    "name": "Development User",
                    "software_background": "beginner",
                    "hardware_background": "none"
                }
                request.state.user_id = "dev_user_001"
            response = await call_next(request)
            return response

        # Validate token
        try:
            user_data = verify_token(token)
            # Attach user data to request state
            request.state.user = user_data
            # Don't log PII (software_background, hardware_background)
            request.state.user_id = user_data.get("sub")

        except TokenExpiredError:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Token has expired", "code": "token_expired"}
            )

        except InvalidTokenError as e:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": str(e), "code": "invalid_token"}
            )

        except Exception as e:
            # Log the error but don't expose details
            print(f"Auth middleware error: {e}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Authentication failed", "code": "auth_error"}
            )

        # Continue with request
        response = await call_next(request)
        return response

    def _is_protected_endpoint(self, path: str) -> bool:
        """
        Check if endpoint requires authentication.
        """
        protected_patterns = [
            r"^/api/chat",
            r"^/api/sessions",
            r"^/api/profile",
            r"^/api/history",
            r"^/api/personalize",
        ]

        return any(re.match(pattern, path) for pattern in protected_patterns)


async def get_current_user(request: Request) -> dict:
    """
    Dependency to get the current authenticated user.

    Returns the user data from the JWT token.

    Raises:
        HTTPException: If user is not authenticated
    """
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    return user


async def get_user_backgrounds(request: Request) -> tuple[str, str]:
    """
    Dependency to get user background information.

    Returns:
        Tuple of (software_background, hardware_background)

    Raises:
        HTTPException: If user is not authenticated
    """
    user = await get_current_user(request)
    software_bg = user.get("software_background", "beginner")
    hardware_bg = user.get("hardware_background", "none")
    return software_bg, hardware_bg
