from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


class DataCollector:
    """Collects training data from MIYA's own interactions.

    Sources:
    - Chat history (SQLite conversation store)
    - User feedback (thumbs-up/down with corrections)
    - Custom datasets (JSONL files uploaded by user)
    - Agent execution logs (high-quality responses)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._data_dir = Path(self._settings.finetune_output_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    async def collect_from_history(
        self,
        db_path: str | None = None,
        min_quality: float = 0.7,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        db = db_path or str(
            self._settings.data_dir / "sqlite" / "conversations.db"
        )
        samples: list[dict[str, Any]] = []

        try:
            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, session_id, role, content, metadata_json
                FROM conversations
                ORDER BY id ASC
                LIMIT ?
                """,
                (limit * 2,),
            )
            rows = cursor.fetchall()
            conn.close()

            by_session: dict[str, list[dict]] = {}
            for row in rows:
                sid = row["session_id"]
                by_session.setdefault(sid, []).append({
                    "row_id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                })

            for sid, msgs in by_session.items():
                msgs.sort(key=lambda m: m["row_id"])
                for i, msg in enumerate(msgs):
                    if msg["role"] == "user" and i + 1 < len(msgs):
                        assistant = msgs[i + 1]
                        if assistant["role"] != "assistant":
                            continue
                        if len(assistant["content"]) < 20:
                            continue

                        score = assistant["metadata"].get("quality_score", 5.0)
                        if isinstance(score, (int, float)) and score >= min_quality * 10:
                            samples.append({
                                "instruction": msg["content"],
                                "output": assistant["content"],
                                "score": score,
                                "messages": [
                                    {"role": "user", "content": msg["content"]},
                                    {"role": "assistant", "content": assistant["content"]},
                                ],
                            })

                    if len(samples) >= limit:
                        break

        except Exception as exc:
            log.warning("history_collection_failed", error=str(exc))

        log.info("collected_from_history", count=len(samples))
        return samples

    async def collect_from_feedback(
        self,
        feedback_path: str | None = None,
    ) -> list[dict[str, Any]]:
        path = Path(feedback_path or str(self._settings.data_dir / "feedback.jsonl"))
        samples: list[dict[str, Any]] = []

        if not path.exists():
            return samples

        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("rating", 0) >= 4:
                        samples.append({
                            "instruction": entry["query"],
                            "output": entry.get("corrected_response") or entry["response"],
                            "source": "feedback",
                        })
        except Exception as exc:
            log.warning("feedback_collection_failed", error=str(exc))

        log.info("collected_from_feedback", count=len(samples))
        return samples

    async def collect_from_dataset(
        self,
        dataset_path: str,
    ) -> list[dict[str, Any]]:
        path = Path(dataset_path)
        samples: list[dict[str, Any]] = []

        if not path.exists():
            log.warning("dataset_not_found", path=dataset_path)
            return samples

        try:
            if path.suffix == ".jsonl":
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            samples.append(json.loads(line))
            elif path.suffix == ".json":
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples.extend(data)
            else:
                log.warning("unsupported_format", suffix=path.suffix)
        except Exception as exc:
            log.warning("dataset_load_failed", error=str(exc))

        log.info("collected_from_dataset", path=dataset_path, count=len(samples))
        return samples

    async def collect_all(
        self,
        custom_datasets: list[str] | None = None,
        min_quality: float = 0.7,
    ) -> list[dict[str, Any]]:
        all_samples: list[dict[str, Any]] = []

        history = await self.collect_from_history(min_quality=min_quality)
        all_samples.extend(history)

        feedback = await self.collect_from_feedback()
        all_samples.extend(feedback)

        for ds_path in (custom_datasets or []):
            ds = await self.collect_from_dataset(ds_path)
            all_samples.extend(ds)

        seen = set()
        unique: list[dict[str, Any]] = []
        for s in all_samples:
            key = s.get("instruction", "")[:200]
            if key and key not in seen:
                seen.add(key)
                unique.append(s)

        output = self._data_dir / f"collected_{int(time.time())}.jsonl"
        with open(output, "w", encoding="utf-8") as f:
            for s in unique:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        log.info("collection_complete", total=len(unique), output=str(output))
        return unique
