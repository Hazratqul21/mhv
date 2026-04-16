from __future__ import annotations

from typing import Any

import chromadb

from app.config import get_settings
from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaTool(BaseTool):
    name = "chroma"
    description = "Vector database operations: add, search, and delete documents in ChromaDB"
    category = "memory"
    parameters = {
        "action": {
            "type": "string",
            "enum": ["add_documents", "search", "delete"],
            "description": "Operation to perform",
        },
        "collection": {
            "type": "string",
            "description": "Target collection name",
        },
        "documents": {
            "type": "array",
            "description": "Documents to add (for add_documents)",
        },
        "ids": {
            "type": "array",
            "description": "Document IDs",
        },
        "query": {
            "type": "string",
            "description": "Search query text (for search)",
        },
        "n_results": {
            "type": "integer",
            "description": "Number of results to return",
            "default": 5,
        },
        "metadatas": {
            "type": "array",
            "description": "Metadata dicts per document (for add_documents)",
        },
    }

    def __init__(self) -> None:
        settings = get_settings()
        self._client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )

    def _get_collection(self, name: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(name)

    async def execute(self, input_data: dict[str, Any]) -> Any:
        action = input_data.get("action", "")
        collection_name = input_data.get("collection", "default")

        try:
            col = self._get_collection(collection_name)

            if action == "add_documents":
                documents = input_data.get("documents", [])
                ids = input_data.get("ids", [])
                metadatas = input_data.get("metadatas")
                if not documents or not ids:
                    return {"error": "Both 'documents' and 'ids' are required"}
                kwargs: dict[str, Any] = {
                    "documents": documents,
                    "ids": ids,
                }
                if metadatas:
                    kwargs["metadatas"] = metadatas
                col.add(**kwargs)
                return {"status": "ok", "added": len(documents)}

            if action == "search":
                query = input_data.get("query", "")
                n_results = int(input_data.get("n_results", 5))
                if not query:
                    return {"error": "'query' is required for search"}
                results = col.query(query_texts=[query], n_results=n_results)
                return {
                    "ids": results.get("ids", []),
                    "documents": results.get("documents", []),
                    "distances": results.get("distances", []),
                    "metadatas": results.get("metadatas", []),
                }

            if action == "delete":
                ids = input_data.get("ids", [])
                if not ids:
                    return {"error": "'ids' required for delete"}
                col.delete(ids=ids)
                return {"status": "ok", "deleted": len(ids)}

            return {"error": f"Unknown action '{action}'"}

        except Exception as exc:
            logger.error("chroma_error", action=action, error=str(exc))
            return {"error": str(exc)}
