"""
Authentication middleware using Better Auth integration.
"""

import json
from typing import Optional, Callable, Awaitable
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from better_abc import ABC, abstractmethod
import jwt
import time


class AuthMiddleware:
    """
    Authentication middleware for validating Better Auth JWT tokens.
    """

    def __init__(
        self,
        jwt_secret: str,
        jwt_algorithm: str = "HS256",
        exclude_paths: Optional[list] = None
    ):
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json"]

    async def __call__(self, request: Request, call_next: Callable[[Request], Awaitable[any]]) -> any:
        """
        Process incoming request with authentication check.
        """
        # Skip auth for excluded paths
        if request.url.path in self.exclude_paths:
            response = await call_next(request)
            return response

        # Extract JWT token from HTTP-only cookie
        token = self._extract_token_from_cookie(request)

        if not token:
            # For public endpoints, allow without auth
            # For protected endpoints, return 401
            if self._is_protected_endpoint(request.url.path):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"error": "Authentication required"}
                )
            else:
                response = await call_next(request)
                return response

        # Validate token
        try:
            user_data = self._validate_token(token)
            request.state.user = user_data
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Token has expired"}
            )
        except jwt.InvalidTokenError:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Invalid token"}
            )

        # Continue with request
        response = await call_next(request)
        return response

    def _extract_token_from_cookie(self, request: Request) -> Optional[str]:
        """
        Extract JWT token from HTTP-only cookie.
        """
        token_cookie = request.cookies.get("better-auth.session-token")
        if not token_cookie:
            return None

        # Token format: "session_token=<token>; HttpOnly; Secure"
        if "=" in token_cookie:
            return token_cookie.split("=")[1].split(";")[0].strip()
        return token_cookie

    def _validate_token(self, token: str) -> dict:
        """
        Validate JWT token and return user data.
        """
        payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])

        # Check if token is expired
        exp = payload.get("exp")
        if exp and time.time() > exp:
            raise jwt.ExpiredSignatureError("Token has expired")

        return payload

    def _is_protected_endpoint(self, path: str) -> bool:
        """
        Check if endpoint requires authentication.
        """
        protected_patterns = [
            "/api/chat",
            "/api/sessions",
            "/api/profile",
            "/api/history"
        ]

        return any(path.startswith(pattern) for pattern in protected_patterns)


# Example usage in main app
def setup_auth_middleware(app, settings):
    """
    Set up authentication middleware for the application.
    """
    auth_middleware = AuthMiddleware(
        jwt_secret=settings.JWT_SECRET,
        jwt_algorithm=settings.JWT_ALGORITHM,
        exclude_paths=["/", "/health", "/docs", "/openapi.json"]
    )

    # Add to middleware stack
    app.add_middleware(AuthMiddleware, auth_middleware)
