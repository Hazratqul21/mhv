"""Client-facing error messages (avoid leaking internals outside development)."""

from __future__ import annotations


def safe_error_detail(env: str, exc: BaseException) -> str:
    """Return exception text in development; generic message otherwise."""
    if (env or "").strip().lower() == "development":
        return str(exc)
    return "An internal error occurred. Check server logs."
