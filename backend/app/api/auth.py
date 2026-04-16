from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


def create_token(
    user_id: str,
    expires_delta: timedelta | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create a signed JWT for *user_id*."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=settings.jwt_expire_minutes))
    payload: dict[str, Any] = {
        "sub": user_id,
        "iat": now,
        "exp": expire,
        **(extra_claims or {}),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT. Raises ``HTTPException`` on failure."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        if payload.get("sub") is None:
            raise JWTError("Missing subject claim")
        return payload
    except JWTError as exc:
        settings = get_settings()
        detail = (
            f"Invalid or expired token: {exc}"
            if settings.env == "development"
            else "Invalid or expired token"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> dict[str, Any]:
    """FastAPI dependency that extracts the authenticated user from a JWT.

    In development mode (``env == "development"`` and no token supplied) a
    default anonymous user is returned so that auth can be bypassed for
    local testing.
    """
    settings = get_settings()

    if credentials is None or credentials.credentials == "":
        if settings.env == "development":
            return {"sub": "dev-user", "role": "admin"}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return verify_token(credentials.credentials)
