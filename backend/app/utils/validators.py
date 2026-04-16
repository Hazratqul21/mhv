from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

ALLOWED_UPLOAD_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".csv", ".json",
    ".py", ".js", ".ts", ".html", ".css",
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".mp3", ".wav", ".ogg", ".mp4", ".webm",
}

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{8,64}$")


def validate_session_id(session_id: str) -> bool:
    return bool(SESSION_ID_PATTERN.match(session_id))


def validate_upload_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_UPLOAD_EXTENSIONS


def validate_upload_size(size_bytes: int) -> bool:
    return 0 < size_bytes <= MAX_UPLOAD_SIZE


def sanitize_text(text: str, max_length: int = 32_000) -> str:
    """Strip control characters and enforce length."""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return cleaned[:max_length]


def validate_model_path(path: Path) -> Optional[str]:
    """Return an error message if the model path is invalid, else None."""
    if not path.exists():
        return f"Model file not found: {path}"
    if not path.suffix == ".gguf":
        return f"Expected .gguf file, got: {path.suffix}"
    if path.stat().st_size < 1_000_000:
        return f"Model file suspiciously small: {path.stat().st_size} bytes"
    return None
