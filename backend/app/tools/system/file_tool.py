from __future__ import annotations

from pathlib import Path
from typing import Any

import aiofiles

from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)

_MAX_READ_SIZE = 1_000_000  # 1 MB guard


class FileTool(BaseTool):
    name = "file"
    description = "Local filesystem operations: read, write, list directories, and search for files"
    category = "system"
    parameters = {
        "action": {
            "type": "string",
            "enum": ["read", "write", "list_dir", "search", "info", "mkdir"],
            "description": "File operation to perform",
        },
        "path": {
            "type": "string",
            "description": "File or directory path",
        },
        "content": {
            "type": "string",
            "description": "Content to write (for write action)",
        },
        "pattern": {
            "type": "string",
            "description": "Glob pattern for search (e.g. '*.py')",
        },
        "recursive": {
            "type": "boolean",
            "description": "Recurse into subdirectories",
            "default": False,
        },
        "encoding": {
            "type": "string",
            "description": "File encoding",
            "default": "utf-8",
        },
    }

    async def execute(self, input_data: dict[str, Any]) -> Any:
        action = input_data.get("action", "")
        target = input_data.get("path", "")

        if not target:
            return {"error": "'path' is required"}

        p = Path(target)

        try:
            if action == "read":
                if not p.exists():
                    return {"error": f"File not found: {target}"}
                if not p.is_file():
                    return {"error": f"Not a file: {target}"}
                size = p.stat().st_size
                if size > _MAX_READ_SIZE:
                    return {"error": f"File too large ({size} bytes). Max is {_MAX_READ_SIZE}."}
                encoding = input_data.get("encoding", "utf-8")
                async with aiofiles.open(p, mode="r", encoding=encoding) as f:
                    content = await f.read()
                return {"path": target, "content": content, "size": size}

            if action == "write":
                content = input_data.get("content", "")
                encoding = input_data.get("encoding", "utf-8")
                p.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(p, mode="w", encoding=encoding) as f:
                    await f.write(content)
                return {"status": "ok", "path": target, "size": len(content.encode(encoding))}

            if action == "list_dir":
                if not p.is_dir():
                    return {"error": f"Not a directory: {target}"}
                recursive = input_data.get("recursive", False)
                entries = []
                iterator = p.rglob("*") if recursive else p.iterdir()
                for item in iterator:
                    entries.append({
                        "name": str(item.relative_to(p)),
                        "type": "dir" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None,
                    })
                    if len(entries) >= 500:
                        break
                return {"path": target, "entries": entries, "count": len(entries)}

            if action == "search":
                pattern = input_data.get("pattern", "*")
                if not p.is_dir():
                    return {"error": f"Not a directory: {target}"}
                recursive = input_data.get("recursive", False)
                iterator = p.rglob(pattern) if recursive else p.glob(pattern)
                matches = []
                for item in iterator:
                    matches.append(str(item))
                    if len(matches) >= 200:
                        break
                return {"pattern": pattern, "matches": matches, "count": len(matches)}

            if action == "info":
                if not p.exists():
                    return {"error": f"Path not found: {target}"}
                stat = p.stat()
                return {
                    "path": target,
                    "exists": True,
                    "type": "dir" if p.is_dir() else "file",
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                }

            if action == "mkdir":
                p.mkdir(parents=True, exist_ok=True)
                return {"status": "ok", "path": target}

            return {"error": f"Unknown action '{action}'"}

        except PermissionError:
            return {"error": f"Permission denied: {target}"}
        except Exception as exc:
            logger.error("file_error", action=action, error=str(exc))
            return {"error": str(exc)}
