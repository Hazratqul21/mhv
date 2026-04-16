from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


class DataFormatter:
    """Converts raw training samples into ChatML / ShareGPT format
    suitable for Unsloth SFT training.

    Supported input formats:
    - {"instruction": "...", "output": "..."}                (Alpaca)
    - {"instruction": "...", "input": "...", "output": "..."}  (Alpaca+input)
    - {"conversations": [{"from": "human", ...}, ...]}       (ShareGPT)
    - {"messages": [{"role": "user", ...}, ...]}             (ChatML)

    Output: ChatML format with optional <think> tags for reasoning.
    """

    def __init__(self, system_prompt: str = "") -> None:
        self._settings = get_settings()
        self._system_prompt = system_prompt or (
            "You are Miya, an autonomous AI assistant. "
            "Think step by step, then provide a clear answer."
        )

    def format_dataset(
        self,
        samples: list[dict[str, Any]],
        include_thinking: bool = True,
    ) -> list[dict[str, Any]]:
        formatted = []
        for sample in samples:
            try:
                messages = self._to_chatml(sample, include_thinking)
                if messages and len(messages) >= 2:
                    formatted.append({"messages": messages})
            except Exception:
                continue

        log.info("formatted_dataset", input=len(samples), output=len(formatted))
        return formatted

    def _to_chatml(
        self, sample: dict[str, Any], include_thinking: bool
    ) -> list[dict[str, str]]:
        if "messages" in sample:
            return self._normalize_messages(sample["messages"])

        if "conversations" in sample:
            return self._from_sharegpt(sample["conversations"])

        if "instruction" in sample:
            return self._from_alpaca(sample, include_thinking)

        return []

    def _from_alpaca(
        self, sample: dict[str, Any], include_thinking: bool
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
        ]

        user_content = sample["instruction"]
        extra_input = sample.get("input", "")
        if extra_input:
            user_content = f"{user_content}\n\n{extra_input}"
        messages.append({"role": "user", "content": user_content})

        output = sample.get("output", "")
        thinking = sample.get("thinking", "")

        if include_thinking and thinking:
            assistant_content = f"<think>\n{thinking}\n</think>\n\n{output}"
        else:
            assistant_content = output

        messages.append({"role": "assistant", "content": assistant_content})
        return messages

    def _from_sharegpt(
        self, conversations: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        role_map = {
            "human": "user",
            "gpt": "assistant",
            "system": "system",
            "user": "user",
            "assistant": "assistant",
        }
        messages: list[dict[str, str]] = []
        for turn in conversations:
            role = role_map.get(turn.get("from", ""), turn.get("role", "user"))
            content = turn.get("value", "") or turn.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
        return messages

    def _normalize_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        normalized = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("system", "user", "assistant") and content:
                normalized.append({"role": role, "content": content})
        return normalized

    def save_formatted(
        self,
        formatted: list[dict[str, Any]],
        output_path: str | None = None,
    ) -> Path:
        out = Path(
            output_path
            or str(Path(self._settings.finetune_output_dir) / "train_data.jsonl")
        )
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            for entry in formatted:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        log.info("saved_formatted", path=str(out), count=len(formatted))
        return out

    def split_train_eval(
        self,
        formatted: list[dict[str, Any]],
        eval_ratio: float = 0.1,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        import random
        shuffled = formatted[:]
        random.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * (1 - eval_ratio)))
        return shuffled[:split_idx], shuffled[split_idx:]
