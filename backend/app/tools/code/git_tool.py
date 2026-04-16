from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GitTool(BaseTool):
    name = "git"
    description = "Git repository inspection: status, diff, log, and branch info"
    category = "code"
    parameters = {
        "action": {
            "type": "string",
            "enum": ["status", "diff", "log", "branch", "show"],
            "description": "Git command to run",
        },
        "repo_path": {
            "type": "string",
            "description": "Path to the git repository",
        },
        "args": {
            "type": "array",
            "description": "Additional arguments for the git command",
        },
        "max_lines": {
            "type": "integer",
            "description": "Maximum output lines to return",
            "default": 200,
        },
    }

    _ALLOWED_COMMANDS = {"status", "diff", "log", "branch", "show"}

    async def _run_git(self, repo_path: str, args: list[str], max_lines: int) -> dict[str, Any]:
        cmd = ["git", "-C", repo_path, *args]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

        if proc.returncode != 0:
            return {"error": stderr.decode(errors="replace").strip(), "exit_code": proc.returncode}

        lines = stdout.decode(errors="replace").splitlines()
        truncated = len(lines) > max_lines
        return {
            "output": "\n".join(lines[:max_lines]),
            "lines": min(len(lines), max_lines),
            "truncated": truncated,
        }

    async def execute(self, input_data: dict[str, Any]) -> Any:
        action = input_data.get("action", "")
        repo_path = input_data.get("repo_path", ".")
        extra_args = input_data.get("args", [])
        max_lines = int(input_data.get("max_lines", 200))

        if action not in self._ALLOWED_COMMANDS:
            return {"error": f"Action '{action}' not allowed. Use one of: {self._ALLOWED_COMMANDS}"}

        if not Path(repo_path).exists():
            return {"error": f"Path not found: {repo_path}"}

        try:
            if action == "status":
                return await self._run_git(repo_path, ["status", "--porcelain", *extra_args], max_lines)

            if action == "diff":
                return await self._run_git(repo_path, ["diff", *extra_args], max_lines)

            if action == "log":
                default_args = ["log", "--oneline", "-20"]
                return await self._run_git(repo_path, [*default_args, *extra_args], max_lines)

            if action == "branch":
                return await self._run_git(repo_path, ["branch", "-a", *extra_args], max_lines)

            if action == "show":
                return await self._run_git(repo_path, ["show", *extra_args], max_lines)

            return {"error": f"Unknown action '{action}'"}

        except asyncio.TimeoutError:
            return {"error": "Git command timed out"}
        except Exception as exc:
            logger.error("git_error", action=action, error=str(exc))
            return {"error": str(exc)}
