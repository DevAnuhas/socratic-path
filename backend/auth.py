"""
JWT authentication for Supabase-issued tokens.

Validates JWTs locally using the shared HMAC secret (sub-millisecond,
no network round-trip). The ``get_current_user`` dependency extracts
the user ID from the JWT ``sub`` claim for use in route handlers.
"""

import logging
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

_jwt_secret: str | None = None
_security = HTTPBearer()


def init_auth() -> None:
    """Load the Supabase JWT secret from environment. Call once at startup."""
    global _jwt_secret
    _jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
    if _jwt_secret:
        logger.info("Auth initialised (JWT secret loaded)")
    else:
        logger.warning(
            "SUPABASE_JWT_SECRET not set — auth will reject all requests"
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> str:
    """
    FastAPI dependency that validates the Supabase JWT and returns the user ID.

    Returns:
        The user's UUID (from the JWT ``sub`` claim).

    Raises:
        HTTPException 401: Invalid or expired token.
        HTTPException 503: Auth not configured (missing JWT secret).
    """
    if not _jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication not configured",
        )

    # Import here to fail fast at call time if python-jose is missing,
    # rather than at module import (keeps startup cleaner).
    from jose import JWTError, jwt

    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            _jwt_secret,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
    except JWTError as exc:
        logger.debug("JWT validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: str | None = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_id
