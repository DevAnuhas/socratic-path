import logging
import os
from typing import Any

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

_jwks: dict[str, Any] | None = None
_security = HTTPBearer()


def init_auth() -> None:
    global _jwks
    supabase_url = os.getenv("SUPABASE_URL")
    if not supabase_url:
        logger.warning("SUPABASE_URL not set — auth disabled")
        return

    jwks_url = f"{supabase_url}/auth/v1/.well-known/jwks.json"
    resp = httpx.get(jwks_url, timeout=10)
    resp.raise_for_status()
    _jwks = resp.json()
    logger.info("Auth initialised (JWKS loaded)")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> str:
    if not _jwks:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Auth not configured")

    from jose import JWTError, jwt

    try:
        payload = jwt.decode(
            credentials.credentials,
            _jwks,
            algorithms=["ES256"],
            options={"verify_aud": False},
        )
    except JWTError:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token missing subject")

    return user_id
