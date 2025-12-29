"""
JWT Verification utility using PyJWT with PyJWKClient for RS256 verification.

This module provides JWT token verification for the backend, fetching public keys
from the auth server's JWKS endpoint and caching them for performance.
"""

import jwt
from jwks_rsa import PyJWKClient, PyJWKClientError
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import logging
import requests
import json
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from app.config import settings

logger = logging.getLogger(__name__)

# Global JWKS client instance (lazy initialization)
_jwks_client: Optional[PyJWKClient] = None


def get_jwks_client() -> PyJWKClient:
    """
    Get or create the JWKS client instance.

    The client automatically caches fetched keys and refreshes them based on
    the configured lifespan (default 24 hours per constitution).
    """
    global _jwks_client

    if _jwks_client is None:
        try:
            _jwks_client = PyJWKClient(
                settings.JWKS_URL,
                cache_keys=True,
                lifespan=settings.JWKS_CACHE_LIFETIME,
            )
            logger.info(f"JWKS client initialized with URL: {settings.JWKS_URL}")
        except PyJWKClientError as e:
            logger.error(f"Failed to initialize JWKS client: {e}")
            raise

    return _jwks_client


class JWTVerificationError(Exception):
    """Base exception for JWT verification errors."""

    pass


class TokenExpiredError(JWTVerificationError):
    """Raised when the token has expired."""

    pass


class InvalidTokenError(JWTVerificationError):
    """Raised when the token is invalid."""

    pass


class JWKSFetchError(JWTVerificationError):
    """Raised when the JWKS endpoint cannot be reached."""

    pass


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify a JWT token and return its payload.

    This function:
    1. Fetches the signing key from the auth server's JWKS endpoint
    2. Verifies the token signature using RS256
    3. Validates the issuer claim
    4. Returns the decoded payload

    Args:
        token: The JWT token to verify

    Returns:
        Dict containing the decoded token payload

    Raises:
        TokenExpiredError: If the token has expired
        InvalidTokenError: If the token is invalid
        JWKSFetchError: If the JWKS endpoint cannot be reached
    """
    if not token:
        raise InvalidTokenError("No token provided")

    try:
        logger.info(f"Attempting to verify token, JWKS URL: {settings.JWKS_URL}")

        # Get the signing key from JWKS
        jwks_client = get_jwks_client()
        logger.info("JWKS client obtained successfully")

        signing_key = jwks_client.get_signing_key_from_jwt(token)
        logger.info("Signing key retrieved from JWKS")

        # Decode and verify the token
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer="physical-ai-auth-server",
            options={
                "verify_aud": False,  # No audience claim in our tokens
                "verify_iat": True,   # Verify issued at time
                "verify_exp": True,   # Verify expiration
            }
        )

        logger.info(f"Token verified successfully for user: {payload.get('sub')}")
        return payload

    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired")
        raise TokenExpiredError("Token has expired")

    except jwt.InvalidIssuerError as e:
        logger.warning(f"JWT token has invalid issuer: {e}")
        raise InvalidTokenError(f"Invalid token issuer: {str(e)}")

    except jwt.InvalidTokenError as e:
        logger.warning(f"JWT token validation failed: {e}")
        raise InvalidTokenError(f"Invalid token: {str(e)}")

    except PyJWKClientError as e:
        logger.error(f"JWKS client error: {e}")
        raise JWKSFetchError(f"Failed to fetch signing key: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error during JWT verification: {e}")
        raise InvalidTokenError(f"Verification failed: {str(e)}")


def extract_token_from_header(authorization: Optional[str]) -> Optional[str]:
    """
    Extract JWT token from Authorization header.

    Supports both "Bearer <token>" format and raw token.

    Args:
        authorization: The Authorization header value

    Returns:
        The extracted token or None if not found
    """
    if not authorization:
        return None

    if authorization.startswith("Bearer "):
        return authorization[7:]

    return authorization


def get_user_from_token(token: str) -> Dict[str, Any]:
    """
    Extract user information from a verified JWT token.

    Returns a sanitized user object without sensitive claims.

    Args:
        token: The JWT token to extract user from

    Returns:
        Dict containing user_id, email, software_background, hardware_background
    """
    payload = verify_token(token)

    # Extract relevant claims (exclude internal JWT claims)
    return {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
        "software_background": payload.get("software_background"),
        "hardware_background": payload.get("hardware_background"),
        "exp": payload.get("exp"),  # Include for client-side expiry check
    }


def is_token_expired(token: str) -> bool:
    """
    Check if a token is expired without full verification.

    Useful for client-side checks to trigger token refresh.

    Args:
        token: The JWT token to check

    Returns:
        True if the token is expired or invalid, False otherwise
    """
    try:
        # Decode without verification to check expiry
        # Note: We don't verify signature here for performance
        # Full verification should be done before using the token
        payload = jwt.decode(
            token,
            options={"verify_signature": False}
        )
        exp = payload.get("exp")
        if exp:
            return datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(tz=timezone.utc)
        return False
    except jwt.DecodeError:
        return True
