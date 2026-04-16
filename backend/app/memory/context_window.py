from __future__ import annotations

from typing import Any

from app.utils.logger import get_logger

log = get_logger(__name__)

CHARS_PER_TOKEN = 3.5


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


class ContextWindow:
    """Assembles an LLM prompt that fits within a token budget.

    Priority order (highest first):
        1. System prompt
        2. User query
        3. Recent conversation history (newest kept, oldest dropped first)
        4. Relevant retrieved documents
    """

    def __init__(self, default_max_tokens: int = 4096) -> None:
        self._default_max = default_max_tokens

    def build_context(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        relevant_docs: list[dict[str, Any]],
        query: str,
        max_tokens: int | None = None,
    ) -> str:
        budget = max_tokens or self._default_max

        sys_block = f"<|system|>\n{system_prompt}\n<|end|>\n"
        sys_tokens = _estimate_tokens(sys_block)
        if sys_tokens > budget:
            sys_block = _truncate_to_tokens(sys_block, budget)
            return sys_block

        query_block = f"<|user|>\n{query}\n<|end|>\n<|assistant|>\n"
        query_tokens = _estimate_tokens(query_block)

        remaining = budget - sys_tokens - query_tokens
        if remaining <= 0:
            return sys_block + _truncate_to_tokens(query_block, budget - sys_tokens)

        history_block, remaining = self._fit_history(history, remaining)
        docs_block = self._fit_docs(relevant_docs, remaining)

        parts = [sys_block]
        if docs_block:
            parts.append(docs_block)
        if history_block:
            parts.append(history_block)
        parts.append(query_block)
        return "".join(parts)

    def _fit_history(
        self,
        history: list[dict[str, Any]],
        budget: int,
    ) -> tuple[str, int]:
        """Pack as many recent messages as possible into *budget* tokens."""
        if not history:
            return "", budget

        lines: list[str] = []
        used = 0
        for msg in reversed(history):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            line = f"<|{role}|>\n{content}\n<|end|>\n"
            cost = _estimate_tokens(line)
            if used + cost > budget:
                break
            lines.append(line)
            used += cost

        lines.reverse()
        return "".join(lines), budget - used

    def _fit_docs(
        self,
        docs: list[dict[str, Any]],
        budget: int,
    ) -> str:
        """Pack relevant docs into the remaining budget."""
        if not docs or budget <= 0:
            return ""

        header = "<|context|>\nRelevant information:\n"
        used = _estimate_tokens(header)
        fragments: list[str] = []

        for doc in docs:
            text = doc.get("document", doc.get("content", ""))
            if not text:
                continue
            entry = f"- {text}\n"
            cost = _estimate_tokens(entry)
            if used + cost > budget:
                break
            fragments.append(entry)
            used += cost

        if not fragments:
            return ""
        return header + "".join(fragments) + "<|end|>\n"

    def estimate_tokens(self, text: str) -> int:
        """Public helper exposing the token estimator."""
        return _estimate_tokens(text)
