from __future__ import annotations

import hashlib
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


class VectorMemory:
    """Semantic vector store backed by ChromaDB.

    Connects to a remote ChromaDB instance in production and falls back to
    a local PersistentClient when ``local_mode=True`` or the remote service
    is unreachable.
    """

    def __init__(
        self,
        collection_name: str = "miya_memory",
        local_mode: bool = False,
    ) -> None:
        self._settings = get_settings()
        self._collection_name = collection_name
        self._local_mode = local_mode
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    def _connect(self) -> chromadb.ClientAPI:
        if self._client is not None:
            return self._client

        if self._local_mode:
            persist_dir = str(self._settings.data_dir / "chromadb")
            log.info("chroma_local_mode", path=persist_dir)
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        else:
            try:
                self._client = chromadb.HttpClient(
                    host=self._settings.chroma_host,
                    port=self._settings.chroma_port,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._client.heartbeat()
                log.info(
                    "chroma_connected",
                    host=self._settings.chroma_host,
                    port=self._settings.chroma_port,
                )
            except Exception as exc:
                log.warning("chroma_remote_failed, falling back to local", error=str(exc))
                persist_dir = str(self._settings.data_dir / "chromadb")
                self._client = chromadb.PersistentClient(
                    path=persist_dir,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )

        return self._client

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is not None:
            return self._collection

        client = self._connect()
        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("chroma_collection_ready", name=self._collection_name)
        return self._collection

    @staticmethod
    def _make_id(text: str, doc_id: str | None = None) -> str:
        if doc_id:
            return doc_id
        return hashlib.sha256(text.encode()).hexdigest()[:20]

    def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Add a document to the vector store. Returns the document id."""
        collection = self._get_collection()
        final_id = self._make_id(text, doc_id)
        collection.upsert(
            ids=[final_id],
            documents=[text],
            metadatas=[metadata or {}],
        )
        log.debug("vector_added", doc_id=final_id)
        return final_id

    def add_batch(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[str]:
        """Add multiple documents in a single call."""
        collection = self._get_collection()
        ids = [
            self._make_id(t, (doc_ids[i] if doc_ids else None))
            for i, t in enumerate(texts)
        ]
        metas = metadatas or [{} for _ in texts]
        collection.upsert(ids=ids, documents=texts, metadatas=metas)
        log.debug("vector_batch_added", count=len(ids))
        return ids

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search. Returns list of dicts with id, document, metadata, distance."""
        collection = self._get_collection()
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(n_results, collection.count() or n_results),
        }
        if filter_dict:
            kwargs["where"] = filter_dict

        if collection.count() == 0:
            return []

        results = collection.query(**kwargs)

        items: list[dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i] if results["documents"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0.0,
            })
        return items

    def delete(self, doc_id: str) -> None:
        """Remove a document by its id."""
        collection = self._get_collection()
        collection.delete(ids=[doc_id])
        log.debug("vector_deleted", doc_id=doc_id)

    def delete_where(self, filter_dict: dict[str, Any]) -> None:
        """Delete all documents matching the metadata filter."""
        collection = self._get_collection()
        collection.delete(where=filter_dict)

    def count(self) -> int:
        """Return the total number of stored documents."""
        return self._get_collection().count()

    def reset_collection(self) -> None:
        """Drop and recreate the collection."""
        client = self._connect()
        try:
            client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = None
        self._get_collection()
        log.info("vector_collection_reset", name=self._collection_name)
