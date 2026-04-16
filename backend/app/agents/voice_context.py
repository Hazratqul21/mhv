"""Shared voice-session hints for orchestrator stream + specialist agents."""

from __future__ import annotations

from typing import Any

VOICE_FULL_ACCESS_SYSTEM = (
    "Desktop **voice** session — **full MIYA** access. Use and delegate the same "
    "capabilities as the text UI: code, terminal/shell when host policy allows, web search, "
    "file/path operations, packages, devops. Do not refuse only because input was spoken; "
    "be decisive and use tools when available. Prefer concise wording for TTS unless the user "
    "asks for detail."
)


def voice_system_blocks(context: dict[str, Any] | None) -> list[dict[str, str]]:
    """Return extra system messages for voice / full-access clients (may be empty)."""
    if not context:
        return []
    out: list[dict[str, str]] = []
    if context.get("voice_full_access") or context.get("miya_client") == "voice":
        out.append({"role": "system", "content": VOICE_FULL_ACCESS_SYSTEM})
    ins = context.get("instruction")
    if isinstance(ins, str) and ins.strip():
        out.append({"role": "system", "content": f"Session instruction: {ins.strip()}"})
    return out


def voice_prepend_after_first_system(
    messages: list[dict[str, str]],
    context: dict[str, Any] | None,
) -> None:
    """Insert :func:`voice_system_blocks` immediately after ``messages[0]``."""
    for i, block in enumerate(voice_system_blocks(context)):
        messages.insert(1 + i, block)
