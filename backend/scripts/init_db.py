"""Initialize MIYA databases and storage buckets."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.memory.conversation_db import ConversationDB
from app.memory.vector_store import VectorMemory


async def main() -> None:
    settings = get_settings()
    print("=" * 50)
    print(" MIYA Database Initialization")
    print("=" * 50)

    # Ensure data directories exist
    for subdir in ("sqlite", "chroma", "cache", "uploads", "logs"):
        path = settings.data_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        print(f"  [✓] Directory: {path}")

    # Initialize SQLite
    print("\n── SQLite ──")
    db_path = settings.data_dir / "sqlite" / "conversations.db"
    conv_db = ConversationDB(str(db_path))
    await conv_db.initialize()
    print(f"  [✓] Conversations database: {db_path}")

    # Initialize ChromaDB collection
    print("\n── ChromaDB ──")
    try:
        vm = VectorMemory(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        count = vm.count()
        print(f"  [✓] Connected. Collection 'miya_memory' has {count} documents.")
    except Exception as exc:
        print(f"  [!] ChromaDB not reachable ({exc}). Will retry when service starts.")

    print("\n" + "=" * 50)
    print(" Initialization complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
