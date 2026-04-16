from __future__ import annotations

import asyncio
import os
from typing import Any

from app.config import get_settings
from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)

MAX_OUTPUT = 50_000


class HostShellTool(BaseTool):
    """Execute shell commands directly on the host system.

    Outside ``development`` this tool is **disabled** unless
    ``MIYA_ALLOW_HOST_SHELL=1`` is set (intentional safety gate).
    """

    name = "host_shell"
    description = (
        "Execute bash commands on the host system. "
        "Disabled in non-development unless MIYA_ALLOW_HOST_SHELL=1."
    )
    category = "system"
    parameters: dict[str, Any] = {
        "command": {
            "type": "string",
            "required": True,
            "description": "Shell command to execute",
        },
        "cwd": {
            "type": "string",
            "required": False,
            "description": "Working directory for the command",
        },
        "timeout": {
            "type": "integer",
            "required": False,
            "default": 30,
            "description": "Timeout in seconds",
        },
        "language": {
            "type": "string",
            "required": False,
            "default": "bash",
            "description": "Shell language (bash/python)",
        },
    }

    async def execute(self, input_data: dict[str, Any]) -> Any:
        command = input_data.get("command") or input_data.get("code", "")
        if not command:
            return {"success": False, "error": "'command' is required"}

        settings = get_settings()
        if settings.env != "development":
            allowed = os.getenv("MIYA_ALLOW_HOST_SHELL", "").strip().lower()
            if allowed not in ("1", "true", "yes", "on"):
                return {
                    "success": False,
                    "error": (
                        "host_shell is disabled when MIYA_ENV is not development. "
                        "Set MIYA_ALLOW_HOST_SHELL=1 only if you accept full host access."
                    ),
                }

        language = input_data.get("language", "bash")
        cwd = input_data.get("cwd")
        timeout = min(max(int(input_data.get("timeout", 30)), 1), 120)

        if language == "python":
            cmd = ["python3", "-c", command]
        else:
            cmd = ["bash", "-c", command]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            out = stdout.decode("utf-8", errors="replace")[:MAX_OUTPUT]
            err = stderr.decode("utf-8", errors="replace")[:MAX_OUTPUT]

            return {
                "success": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": out,
                "stderr": err,
                "output": out if proc.returncode == 0 else err,
            }

        except asyncio.TimeoutError:
            return {"success": False, "error": f"Command timed out after {timeout}s"}
        except FileNotFoundError:
            return {"success": False, "error": "Shell not found"}
        except Exception as exc:
            logger.error("host_shell_error", command=command[:100], error=str(exc))
            return {"success": False, "error": str(exc)}
