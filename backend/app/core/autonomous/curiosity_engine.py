"""MIYA Curiosity Engine — Autonomous self-improvement loop.

Runs daily and weekly cycles:
- Daily: analyze conversations, find weak areas, research online, store knowledge
- Weekly: collect training data, fine-tune model, evaluate, deploy if improved
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


class CuriosityEngine:
    """Orchestrates MIYA's autonomous self-improvement."""

    def __init__(
        self,
        llm_engine: Any,
        conversation_db: Any,
        vector_memory: Any | None = None,
    ) -> None:
        self._engine = llm_engine
        self._conv_db = conversation_db
        self._vector = vector_memory
        self._settings = get_settings()
        self._data_dir = Path(self._settings.data_dir)
        self._running = False
        self._stats = {
            "daily_runs": 0,
            "weekly_runs": 0,
            "topics_researched": 0,
            "knowledge_stored": 0,
            "training_runs": 0,
            "models_deployed": 0,
        }

    # ------------------------------------------------------------------
    # Daily cycle
    # ------------------------------------------------------------------

    async def run_daily_cycle(self) -> dict[str, Any]:
        """Full daily self-improvement cycle."""
        log.info("curiosity_daily_start")
        results: dict[str, Any] = {"timestamp": time.time()}

        try:
            weak_areas = await self._analyze_conversations()
            results["weak_areas"] = weak_areas
            log.info("curiosity_analysis_done", weak_count=len(weak_areas))

            if weak_areas:
                knowledge = await self._research_topics(weak_areas)
                results["knowledge_gained"] = len(knowledge)
                log.info("curiosity_research_done", items=len(knowledge))

                stored = await self._store_knowledge(knowledge)
                results["stored"] = stored
                self._stats["knowledge_stored"] += stored

            collected = await self._collect_training_data()
            results["training_samples"] = collected

            self._stats["daily_runs"] += 1
            results["status"] = "success"

        except Exception as exc:
            log.error("curiosity_daily_failed", error=str(exc))
            results["status"] = "error"
            results["error"] = str(exc)

        self._save_cycle_log("daily", results)
        log.info("curiosity_daily_complete", results=results)
        return results

    # ------------------------------------------------------------------
    # Weekly cycle
    # ------------------------------------------------------------------

    async def run_weekly_cycle(self) -> dict[str, Any]:
        """Weekly fine-tuning and model improvement cycle."""
        log.info("curiosity_weekly_start")
        results: dict[str, Any] = {"timestamp": time.time()}

        try:
            data_count = await self._count_training_data()
            results["available_samples"] = data_count

            if data_count < 50:
                results["status"] = "skipped"
                results["reason"] = f"Not enough training data ({data_count} < 50 minimum)"
                log.info("curiosity_weekly_skipped", reason=results["reason"])
                self._save_cycle_log("weekly", results)
                return results

            train_result = await self._train_model()
            results["training"] = train_result

            if train_result.get("success"):
                improved = await self._evaluate_new_model(train_result.get("model_path", ""))
                results["evaluation"] = improved

                if improved.get("is_better"):
                    deploy = await self._deploy_and_reload(train_result.get("model_path", ""))
                    results["deployed"] = deploy
                    self._stats["models_deployed"] += 1
                else:
                    results["deployed"] = False
                    log.info("curiosity_old_model_better")

            self._stats["weekly_runs"] += 1
            results["status"] = "success"

        except Exception as exc:
            log.error("curiosity_weekly_failed", error=str(exc))
            results["status"] = "error"
            results["error"] = str(exc)

        self._save_cycle_log("weekly", results)
        return results

    # ------------------------------------------------------------------
    # Step 1: Analyze conversations
    # ------------------------------------------------------------------

    async def _analyze_conversations(self) -> list[dict[str, Any]]:
        """Find weak responses from the last 24 hours."""
        cutoff = time.time() - 86400
        weak_areas: list[dict[str, Any]] = []

        try:
            sessions = await self._conv_db.get_sessions(limit=100)
        except Exception as exc:
            log.warning("curiosity_sessions_failed", error=str(exc))
            return []

        model_name = self._settings.chat_model
        try:
            self._engine.get_model(model_name)
        except KeyError:
            try:
                self._engine.swap_model(model_name, n_ctx=self._settings.chat_ctx)
            except Exception:
                log.warning("curiosity_model_unavailable")
                return []

        for session in sessions:
            if session.get("last_active", 0) < cutoff:
                continue

            history = await self._conv_db.get_history(session["session_id"], limit=20)

            pairs = self._extract_qa_pairs(history)
            for pair in pairs:
                score = await self._evaluate_response(pair, model_name)
                if score < 6.0:
                    weak_areas.append({
                        "question": pair["question"][:300],
                        "answer": pair["answer"][:300],
                        "score": score,
                        "topic": await self._extract_topic(pair["question"], model_name),
                        "session_id": session["session_id"],
                    })

                    await self._save_quality_score(
                        session["session_id"], pair, score
                    )

        weak_areas.sort(key=lambda x: x["score"])
        return weak_areas[:20]

    def _extract_qa_pairs(self, history: list[dict]) -> list[dict[str, str]]:
        pairs: list[dict[str, str]] = []
        for i, msg in enumerate(history):
            if msg["role"] == "user" and i + 1 < len(history):
                next_msg = history[i + 1]
                if next_msg["role"] == "assistant" and len(next_msg["content"]) > 10:
                    pairs.append({
                        "question": msg["content"],
                        "answer": next_msg["content"],
                    })
        return pairs

    async def _evaluate_response(self, pair: dict, model_name: str) -> float:
        prompt = (
            "Rate this AI response on a scale of 1-10. Consider accuracy, "
            "completeness, and helpfulness. Reply with ONLY a number.\n\n"
            f"Question: {pair['question'][:200]}\n"
            f"Answer: {pair['answer'][:400]}\n\n"
            "Score (1-10):"
        )
        try:
            result = await self._engine.generate(
                model_name, prompt=prompt, max_tokens=10, temperature=0.1,
            )
            text = result.get("text", "").strip()
            for word in text.split():
                try:
                    score = float(word)
                    if 1 <= score <= 10:
                        return score
                except ValueError:
                    continue
        except Exception:
            pass
        return 5.0

    async def _extract_topic(self, question: str, model_name: str) -> str:
        prompt = (
            "What is the main topic of this question? Reply with 2-4 words only.\n\n"
            f"Question: {question[:200]}\n\nTopic:"
        )
        try:
            result = await self._engine.generate(
                model_name, prompt=prompt, max_tokens=20, temperature=0.1,
            )
            return result.get("text", "general").strip()[:50]
        except Exception:
            return "general"

    async def _save_quality_score(
        self, session_id: str, pair: dict, score: float
    ) -> None:
        """Save quality score to conversations metadata."""
        try:
            db = await self._conv_db._conn()
            await db.execute(
                "UPDATE conversations SET metadata_json = json_set("
                "COALESCE(metadata_json, '{}'), '$.quality_score', ?) "
                "WHERE session_id = ? AND role = 'assistant' AND content = ?",
                (score, session_id, pair["answer"][:500]),
            )
            await db.commit()
        except Exception as exc:
            log.debug("quality_score_save_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Step 2: Research topics
    # ------------------------------------------------------------------

    async def _research_topics(
        self, weak_areas: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Search the web for topics where MIYA performed poorly."""
        topics = set()
        for area in weak_areas:
            topic = area.get("topic", "")
            if topic and topic != "general":
                topics.add(topic)

        if not topics:
            return []

        knowledge: list[dict[str, Any]] = []

        try:
            from duckduckgo_search import DDGS
        except ImportError:
            log.warning("curiosity_duckduckgo_not_installed")
            return []

        for topic in list(topics)[:5]:
            search_query = f"{topic} explained tutorial guide"
            try:
                results = await asyncio.to_thread(
                    DDGS().text, search_query, max_results=5,
                )
                for r in (results or []):
                    knowledge.append({
                        "topic": topic,
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "url": r.get("href", ""),
                        "source": "web_research",
                        "timestamp": time.time(),
                    })
                    self._stats["topics_researched"] += 1
            except Exception as exc:
                log.warning("curiosity_search_failed", topic=topic, error=str(exc))

        return knowledge

    # ------------------------------------------------------------------
    # Step 3: Store knowledge
    # ------------------------------------------------------------------

    async def _store_knowledge(self, knowledge: list[dict[str, Any]]) -> int:
        """Store researched knowledge in ChromaDB for future RAG retrieval."""
        if not self._vector or not knowledge:
            return 0

        stored = 0
        for item in knowledge:
            text = f"Topic: {item['topic']}\nTitle: {item['title']}\n{item['body']}"
            metadata = {
                "source": "curiosity_research",
                "topic": item["topic"],
                "url": item.get("url", ""),
                "timestamp": str(int(item.get("timestamp", time.time()))),
            }
            try:
                self._vector.add(text=text, metadata=metadata)
                stored += 1
            except Exception as exc:
                log.debug("curiosity_store_failed", error=str(exc))

        log.info("curiosity_knowledge_stored", count=stored)
        return stored

    # ------------------------------------------------------------------
    # Step 4: Collect training data
    # ------------------------------------------------------------------

    async def _collect_training_data(self) -> int:
        """Collect high-quality conversations for fine-tuning."""
        output_dir = self._data_dir / "finetune"
        output_dir.mkdir(parents=True, exist_ok=True)

        samples: list[dict[str, Any]] = []

        try:
            sessions = await self._conv_db.get_sessions(limit=200)
        except Exception:
            return 0

        for session in sessions:
            history = await self._conv_db.get_history(session["session_id"], limit=50)
            pairs = self._extract_qa_pairs(history)

            for pair in pairs:
                if len(pair["answer"]) < 20:
                    continue

                meta = {}
                try:
                    db = await self._conv_db._conn()
                    rows = await db.execute_fetchall(
                        "SELECT metadata_json FROM conversations "
                        "WHERE session_id = ? AND role = 'assistant' AND content = ? LIMIT 1",
                        (session["session_id"], pair["answer"][:500]),
                    )
                    if rows:
                        meta = json.loads(rows[0][0]) if rows[0][0] else {}
                except Exception:
                    pass

                score = meta.get("quality_score", 5.0)
                if score >= 7.0:
                    samples.append({
                        "messages": [
                            {"role": "user", "content": pair["question"]},
                            {"role": "assistant", "content": pair["answer"]},
                        ],
                        "score": score,
                    })

        if samples:
            output_path = output_dir / f"auto_collected_{int(time.time())}.jsonl"
            with open(output_path, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            log.info("curiosity_training_data_collected", count=len(samples), path=str(output_path))

        return len(samples)

    async def _count_training_data(self) -> int:
        """Count available training samples across all collected files."""
        finetune_dir = self._data_dir / "finetune"
        if not finetune_dir.exists():
            return 0

        total = 0
        for f in finetune_dir.glob("*.jsonl"):
            try:
                with open(f) as fh:
                    total += sum(1 for line in fh if line.strip())
            except Exception:
                pass
        return total

    # ------------------------------------------------------------------
    # Step 5: Train model (weekly)
    # ------------------------------------------------------------------

    async def _train_model(self) -> dict[str, Any]:
        """Run LoRA fine-tuning on collected data."""
        try:
            from app.core.finetune.trainer import FineTuneTrainer
            trainer = FineTuneTrainer()

            finetune_dir = self._data_dir / "finetune"
            data_files = sorted(finetune_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)

            if not data_files:
                return {"success": False, "error": "No training data files"}

            latest = str(data_files[-1])
            result = await trainer.start_training(
                dataset_path=latest,
                base_model=self._settings.chat_model,
            )

            self._stats["training_runs"] += 1
            return result

        except Exception as exc:
            log.error("curiosity_training_failed", error=str(exc))
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Step 6: Evaluate new model
    # ------------------------------------------------------------------

    async def _evaluate_new_model(self, model_path: str) -> dict[str, Any]:
        """Compare new fine-tuned model against current model."""
        try:
            from app.core.finetune.evaluator import ModelEvaluator
            evaluator = ModelEvaluator()
            result = await evaluator.compare_models(
                model_a=self._settings.chat_model,
                model_b=model_path,
            )
            is_better = result.get("recommendation", "") == "model_b"
            return {"is_better": is_better, **result}
        except Exception as exc:
            log.error("curiosity_eval_failed", error=str(exc))
            return {"is_better": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Step 7: Deploy and reload
    # ------------------------------------------------------------------

    async def _deploy_and_reload(self, model_path: str) -> dict[str, Any]:
        """Deploy new model and hot-reload the engine."""
        try:
            from app.core.finetune.deployer import ModelDeployer
            deployer = ModelDeployer()
            deploy_result = await deployer.deploy(
                model_path=model_path,
                role="chat",
            )

            new_model = deploy_result.get("deployed_as", "")
            if new_model:
                self._engine.hot_reload("chat", new_model)
                log.info("curiosity_model_deployed", model=new_model)

            return {"success": True, "model": new_model}
        except Exception as exc:
            log.error("curiosity_deploy_failed", error=str(exc))
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_cycle_log(self, cycle_type: str, results: dict) -> None:
        log_dir = self._data_dir / "curiosity_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{cycle_type}_{int(time.time())}.json"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pass

    def get_stats(self) -> dict[str, Any]:
        return {**self._stats}
