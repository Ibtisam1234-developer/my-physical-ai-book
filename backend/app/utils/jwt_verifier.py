"""
JWT Verification utility using PyJWT with PyJWKClient for RS256 verification.

This module provides JWT token verification for the backend, fetching public keys
from the auth server's JWKS endpoint and caching them for performance.
"""

import jwt  # This imports PyJWT (not the jwt package)
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import logging
import requests
import json
from jose import jwk
from jose.utils import base64url_decode
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from app.config import settings

logger = logging.getLogger(__name__)

# Global JWKS cache (simple implementation)
_jwks_cache: Optional[Dict] = None
_last_jwks_refresh: Optional[datetime] = None


def get_jwks_client():
    """
    Get JWKS from the auth server with simple caching.
    """
    global _jwks_cache, _last_jwks_refresh

    now = datetime.now(timezone.utc)
    cache_duration = getattr(settings, 'JWKS_CACHE_LIFETIME', 86400)  # Default 24 hours in seconds

    # Refresh cache if empty or expired
    if (_jwks_cache is None or
        _last_jwks_refresh is None or
        (now - _last_jwks_refresh).total_seconds() > cache_duration):

        try:
            response = requests.get(settings.JWKS_URL)
            response.raise_for_status()
            _jwks_cache = response.json()
            _last_jwks_refresh = now
            logger.info(f"JWKS refreshed from URL: {settings.JWKS_URL}")
        except requests.RequestException as e:
            logger.error(f"Failed to fetch JWKS: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JWKS response: {e}")
            raise

    return _jwks_cache


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


def get_signing_key_from_jwt(token: str):
    """
    Extract the signing key from JWKS based on the token's kid header.
    """
    try:
        import base64

        # Decode header without verification to get kid
        header_data = token.split('.')[0]
        # Add padding if needed
        header_data += '=' * (4 - len(header_data) % 4)
        # Use standard base64 urlsafe decode
        header_json = base64.urlsafe_b64decode(header_data)
        header = json.loads(header_json.decode('utf-8'))

        kid = header.get('kid')
        if not kid:
            raise InvalidTokenError("Token header missing kid")

        # Get JWKS
        jwks = get_jwks_client()

        # Find the key with matching kid
        key = None
        for jwk_data in jwks['keys']:
            if jwk_data['kid'] == kid:
                key = jwk_data
                break

        if not key:
            raise InvalidTokenError(f"Signing key with kid {kid} not found")

        # Parse the key components from the JWKS key data
        # Properly handle base64url decoding
        # Add proper padding for base64 decoding
        n_b64 = key['n'] + '=' * (4 - len(key['n']) % 4)
        e_b64 = key['e'] + '=' * (4 - len(key['e']) % 4)

        n_bytes = base64.urlsafe_b64decode(n_b64)
        e_bytes = base64.urlsafe_b64decode(e_b64)

        n = int.from_bytes(n_bytes, 'big')
        e = int.from_bytes(e_bytes, 'big')

        # Create RSA public key using cryptography
        public_numbers = rsa.RSAPublicNumbers(e, n)
        public_key_crypto = public_numbers.public_key()

        # For the jwt package (1.4.0), we need to create a proper JWK
        # Use the jwt.jwk.RSAJWK class if available, otherwise use the generic approach
        try:
            from jwt.jwk import RSAJWK
            public_key = RSAJWK(public_key_crypto)
        except AttributeError:
            # If RSAJWK doesn't exist, use the generic JWK approach
            # Create the JWK manually with the required parameters
            from jwt.jwk import JWK
            public_key = JWK.from_pyca(public_key_crypto)

        return public_key
    except Exception as e:
        logger.error(f"Error getting signing key: {e}")
        raise InvalidTokenError(f"Error getting signing key: {str(e)}")


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
        signing_key = get_signing_key_from_jwt(token)
        logger.info("Signing key retrieved from JWKS")

        # Decode and verify the token using jwt.JWT class
        jwt_instance = jwt.JWT()
        # Verify the token and get the payload
        payload = jwt_instance.decode(token, signing_key)

        # Verify issuer claim manually since jwt.JWT doesn't handle it automatically
        if payload.get('iss') != "physical-ai-auth-server":
            logger.warning(f"Invalid issuer: {payload.get('iss')}")
            raise InvalidTokenError(f"Invalid token issuer: {payload.get('iss')}")

        logger.info(f"Token verified successfully for user: {payload.get('sub')}")
        return payload

    except jwt.exceptions.JWTDecodeError as e:
        # Check if it's an expiration error by examining the message
        if "expired" in str(e).lower():
            logger.warning("JWT token has expired")
            raise TokenExpiredError("Token has expired")
        else:
            logger.warning(f"JWT token validation failed: {e}")
            raise InvalidTokenError(f"Invalid token: {str(e)}")

    except requests.RequestException as e:
        logger.error(f"JWKS fetch error: {e}")
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
