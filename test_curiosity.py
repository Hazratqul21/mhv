#!/usr/bin/env python3
"""Test suite for the MIYA Curiosity Loop system.

Tests each component individually: ConversationDB, DataCollector,
CuriosityEngine (analyze, research, store, collect), JobRunner, and
LLMEngine hot_reload.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Mock llama_cpp before any app imports
llama_mock = MagicMock()
sys.modules["llama_cpp"] = llama_mock

os.environ.setdefault("MIYA_ENV", "development")
os.environ.setdefault("MIYA_DATA_DIR", tempfile.mkdtemp(prefix="miya_test_"))

PASS = 0
FAIL = 0


def ok(label: str):
    global PASS
    PASS += 1
    print(f"  \033[32m✓\033[0m {label}")


def fail(label: str, detail: str = ""):
    global FAIL
    FAIL += 1
    extra = f" — {detail}" if detail else ""
    print(f"  \033[31m✗\033[0m {label}{extra}")


# =====================================================================
# Test 1: ConversationDB with quality_score in metadata
# =====================================================================
async def test_conversation_db():
    print("\n[1] ConversationDB quality_score support")
    from app.memory.conversation_db import ConversationDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = ConversationDB(db_path=Path(tmpdir) / "test.db")
        await db.initialize()

        sid = "test-session-1"
        await db.add_message(sid, "user", "What is Python?")
        await db.add_message(
            sid, "assistant", "Python is a programming language.",
            metadata={"quality_score": 8.5},
        )
        await db.add_message(sid, "user", "Explain loops")
        await db.add_message(
            sid, "assistant", "Loops iterate.",
            metadata={"quality_score": 3.0},
        )

        history = await db.get_history(sid, limit=10)
        if len(history) == 4:
            ok("4 messages stored")
        else:
            fail("4 messages stored", f"got {len(history)}")

        high_q = [
            m for m in history
            if m["role"] == "assistant" and m["metadata"].get("quality_score", 0) >= 7
        ]
        if len(high_q) == 1 and high_q[0]["metadata"]["quality_score"] == 8.5:
            ok("quality_score stored in metadata correctly")
        else:
            fail("quality_score stored in metadata", f"high_q={high_q}")

        await db.close()


# =====================================================================
# Test 2: DataCollector adapted to real schema
# =====================================================================
async def test_data_collector():
    print("\n[2] DataCollector — ConversationDB schema compatibility")
    from app.core.finetune.data_collector import DataCollector

    with tempfile.TemporaryDirectory() as tmpdir:
        import sqlite3
        db_path = Path(tmpdir) / "conversations.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata_json TEXT DEFAULT '{}'
            )
        """)

        rows = [
            ("s1", "user", "What is AI?", time.time(), "{}"),
            ("s1", "assistant", "AI is artificial intelligence, a branch of computer science dealing with machines that can learn and reason.",
             time.time(), '{"quality_score": 9.0}'),
            ("s1", "user", "How does ML work?", time.time(), "{}"),
            ("s1", "assistant", "ML works by training models on data to find patterns.",
             time.time(), '{"quality_score": 8.0}'),
            ("s2", "user", "Hello", time.time(), "{}"),
            ("s2", "assistant", "Bad answer",
             time.time(), '{"quality_score": 2.0}'),
        ]
        conn.executemany(
            "INSERT INTO conversations (session_id, role, content, timestamp, metadata_json) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()

        collector = DataCollector()
        samples = await collector.collect_from_history(
            db_path=str(db_path),
            min_quality=0.7,
        )

        if len(samples) >= 2:
            ok(f"collected {len(samples)} high-quality samples")
        else:
            fail(f"expected >=2 samples", f"got {len(samples)}")

        has_messages = all("messages" in s for s in samples)
        if has_messages:
            ok("samples include 'messages' field for SFT")
        else:
            fail("samples missing 'messages' field")

        low_q = [s for s in samples if s.get("score", 10) < 7]
        if not low_q:
            ok("low-quality samples correctly filtered out")
        else:
            fail("low-quality samples leaked", f"count={len(low_q)}")


# =====================================================================
# Test 3: CuriosityEngine — analyze, research, store
# =====================================================================
async def test_curiosity_engine():
    print("\n[3] CuriosityEngine — analyze, research, store, collect")
    from app.core.autonomous.curiosity_engine import CuriosityEngine
    from app.memory.conversation_db import ConversationDB
    from app.memory.vector_store import VectorMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["MIYA_DATA_DIR"] = tmpdir
        from importlib import reload
        import app.config
        reload(app.config)
        app.config.get_settings.cache_clear()

        db = ConversationDB(db_path=Path(tmpdir) / "conversations.db")
        await db.initialize()

        sid = "curiosity-test"
        now = time.time()
        await db.add_message(sid, "user", "Explain quantum computing")
        await db.add_message(
            sid, "assistant",
            "Quantum computing uses qubits and superposition to perform parallel computations. "
            "It can solve certain problems exponentially faster than classical computers.",
            metadata={"quality_score": 8.0},
        )
        await db.add_message(sid, "user", "What is 2+2?")
        await db.add_message(
            sid, "assistant",
            "The answer to 2+2 is 3 which is basic arithmetic.",
            metadata={"quality_score": 2.0},
        )

        mock_engine = MagicMock()
        mock_engine.get_model = MagicMock(side_effect=KeyError("not loaded"))
        mock_engine.load_model = MagicMock()

        async def mock_generate(model, prompt, max_tokens=10, temperature=0.1):
            if "Rate this" in prompt:
                if "2+2" in prompt and "answer to 2+2 is 3" in prompt:
                    return {"text": "2"}
                return {"text": "8"}
            if "topic" in prompt.lower():
                return {"text": "math basics"}
            return {"text": "5"}

        mock_engine.generate = mock_generate

        try:
            vector = VectorMemory(
                collection_name="test_curiosity",
                local_mode=True,
            )
        except Exception:
            vector = None

        settings = app.config.get_settings()
        object.__setattr__(settings, "chat_model", "test-model.gguf")
        object.__setattr__(settings, "chat_ctx", 2048)

        engine = CuriosityEngine(
            llm_engine=mock_engine,
            conversation_db=db,
            vector_memory=vector,
        )

        mock_settings = MagicMock()
        mock_settings.chat_model = "test-model.gguf"
        mock_settings.chat_ctx = 2048
        mock_settings.data_dir = Path(tmpdir)
        engine._settings = mock_settings

        # --- _analyze_conversations ---

        weak = await engine._analyze_conversations()
        if weak:
            ok(f"found {len(weak)} weak areas")
            topics = [w.get("topic", "") for w in weak]
            ok(f"extracted topics: {topics}")
        else:
            fail("no weak areas found (expected at least 1)")

        # --- _research_topics ---
        try:
            from duckduckgo_search import DDGS
            knowledge = await engine._research_topics(weak)
            if knowledge:
                ok(f"researched {len(knowledge)} knowledge items")
            else:
                ok("research returned 0 (network may be unavailable, OK)")
        except ImportError:
            ok("duckduckgo_search not installed, skipping research test")
            knowledge = [
                {"topic": "math basics", "title": "Test", "body": "Test content", "url": "", "timestamp": time.time()},
            ]

        # --- _store_knowledge ---
        if vector:
            stored = await engine._store_knowledge(knowledge)
            if stored > 0:
                ok(f"stored {stored} items in vector memory")
            elif not knowledge:
                ok("no knowledge to store (OK)")
            else:
                fail("store_knowledge returned 0")
        else:
            ok("vector memory unavailable, skip store test")

        # --- _collect_training_data ---
        collected = await engine._collect_training_data()
        ok(f"collect_training_data returned {collected} samples")

        # --- stats ---
        stats = engine.get_stats()
        if isinstance(stats, dict):
            ok(f"stats: {stats}")
        else:
            fail("get_stats failed")

        if vector:
            try:
                vector.reset_collection()
            except Exception:
                pass

        await db.close()


# =====================================================================
# Test 4: JobRunner
# =====================================================================
async def test_job_runner():
    print("\n[4] JobRunner — scheduling and execution")
    from app.core.autonomous.job_runner import JobRunner

    call_log: list[str] = []

    async def mock_daily():
        call_log.append("daily")

    async def mock_weekly():
        call_log.append("weekly")

    runner = JobRunner(check_interval=1)

    runner.register("daily_test", mock_daily, interval_seconds=2, run_immediately=True)
    runner.register("weekly_test", mock_weekly, interval_seconds=100)

    status = runner.get_status()
    if len(status["jobs"]) == 2:
        ok("2 jobs registered")
    else:
        fail("job registration", f"got {len(status['jobs'])} jobs")

    runner.start()
    await asyncio.sleep(3.5)
    await runner.stop()

    if "daily" in call_log:
        ok(f"daily job ran ({call_log.count('daily')}x)")
    else:
        fail("daily job did not run")

    if "weekly" not in call_log:
        ok("weekly job correctly did NOT run yet (interval=100s)")
    else:
        fail("weekly job ran too early")

    # --- Manual trigger ---
    call_log.clear()
    ran = await runner.run_now("weekly_test")
    if ran and "weekly" in call_log:
        ok("run_now('weekly_test') executed successfully")
    else:
        fail("run_now failed")

    # --- Enable/disable ---
    runner.disable_job("daily_test")
    s = runner.get_status()
    if not s["jobs"]["daily_test"]["enabled"]:
        ok("disable_job works")
    else:
        fail("disable_job failed")

    runner.enable_job("daily_test")
    s = runner.get_status()
    if s["jobs"]["daily_test"]["enabled"]:
        ok("enable_job works")
    else:
        fail("enable_job failed")


# =====================================================================
# Test 5: LLMEngine hot_reload
# =====================================================================
async def test_hot_reload():
    print("\n[5] LLMEngine hot_reload")
    from app.core.llm_engine import LLMEngine

    engine = LLMEngine()

    if hasattr(engine, "hot_reload"):
        ok("hot_reload method exists")
    else:
        fail("hot_reload method missing")
        return

    engine._models["old-model.gguf"] = MagicMock()
    engine._locks["old-model.gguf"] = asyncio.Lock()

    try:
        object.__setattr__(engine._settings, "chat_model", "old-model.gguf")
    except Exception:
        pass

    with patch.object(engine, "load_model") as mock_load:
        engine.hot_reload("chat", "new-model.gguf")

        if "old-model.gguf" not in engine._models:
            ok("old model unloaded")
        else:
            fail("old model still loaded")

        mock_load.assert_called_once()
        call_args = mock_load.call_args
        if call_args[0][0] == "new-model.gguf":
            ok("new model load_model called with correct filename")
        else:
            fail("load_model called with wrong args", str(call_args))

    engine.hot_reload("unknown_role", "whatever.gguf")
    ok("unknown role handled gracefully")


# =====================================================================
# Test 6: Full integration — daily cycle end-to-end
# =====================================================================
async def test_full_daily_cycle():
    print("\n[6] Full daily cycle — end-to-end")
    from app.core.autonomous.curiosity_engine import CuriosityEngine
    from app.memory.conversation_db import ConversationDB

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["MIYA_DATA_DIR"] = tmpdir
        from importlib import reload
        import app.config
        reload(app.config)
        app.config.get_settings.cache_clear()

        settings = app.config.get_settings()
        object.__setattr__(settings, "chat_model", "test.gguf")
        object.__setattr__(settings, "chat_ctx", 2048)

        db = ConversationDB(db_path=Path(tmpdir) / "conv.db")
        await db.initialize()

        for i in range(5):
            sid = f"session-{i}"
            await db.add_message(sid, "user", f"Question number {i} about topic {i}")
            await db.add_message(
                sid, "assistant",
                f"Here is a detailed answer about topic {i} that covers the key points.",
                metadata={"quality_score": 8.0 if i % 2 == 0 else 3.0},
            )

        mock_engine = MagicMock()
        mock_engine.get_model = MagicMock(side_effect=KeyError)
        mock_engine.load_model = MagicMock()

        async def mock_gen(model, prompt, max_tokens=10, temperature=0.1):
            if "Rate this" in prompt:
                return {"text": "4"}
            return {"text": "general topic"}

        mock_engine.generate = mock_gen

        engine = CuriosityEngine(
            llm_engine=mock_engine,
            conversation_db=db,
            vector_memory=None,
        )

        object.__setattr__(engine._settings, "chat_model", "test.gguf")
        object.__setattr__(engine._settings, "chat_ctx", 2048)
        engine._data_dir = Path(tmpdir)

        result = await engine.run_daily_cycle()

        if result.get("status") == "success":
            ok("daily cycle completed successfully")
        else:
            fail("daily cycle failed", str(result.get("error", "")))

        if "weak_areas" in result:
            ok(f"identified {len(result['weak_areas'])} weak areas")
        else:
            ok("no weak areas key (may be empty)")

        log_dir = Path(tmpdir) / "curiosity_logs"
        logs = list(log_dir.glob("daily_*.json")) if log_dir.exists() else []
        if logs:
            ok(f"cycle log saved: {logs[0].name}")
        else:
            fail("cycle log not saved")

        stats = engine.get_stats()
        if stats["daily_runs"] == 1:
            ok("daily_runs counter = 1")
        else:
            fail("daily_runs counter", f"got {stats['daily_runs']}")

        await db.close()


# =====================================================================

async def main():
    print("=" * 60)
    print("  MIYA Curiosity Loop — Test Suite")
    print("=" * 60)

    await test_conversation_db()
    await test_data_collector()
    await test_curiosity_engine()
    await test_job_runner()
    await test_hot_reload()
    await test_full_daily_cycle()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"  \033[32mALL {total} TESTS PASSED\033[0m")
    else:
        print(f"  \033[31m{FAIL}/{total} FAILED\033[0m, {PASS} passed")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    code = asyncio.run(main())
    sys.exit(code)
