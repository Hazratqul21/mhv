from __future__ import annotations

from typing import Any

import asyncssh

from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SSHTool(BaseTool):
    name = "ssh"
    description = "Execute commands on remote hosts via SSH"
    category = "system"
    parameters = {
        "host": {
            "type": "string",
            "description": "Remote hostname or IP",
        },
        "command": {
            "type": "string",
            "description": "Command to execute on the remote host",
        },
        "username": {
            "type": "string",
            "description": "SSH username",
            "default": "root",
        },
        "port": {
            "type": "integer",
            "description": "SSH port",
            "default": 22,
        },
        "key_path": {
            "type": "string",
            "description": "Path to private key file",
        },
        "password": {
            "type": "string",
            "description": "SSH password (key_path preferred)",
        },
        "timeout": {
            "type": "integer",
            "description": "Connection timeout in seconds",
            "default": 30,
        },
    }

    async def execute(self, input_data: dict[str, Any]) -> Any:
        host = input_data.get("host", "")
        command = input_data.get("command", "")

        if not host:
            return {"error": "'host' is required"}
        if not command:
            return {"error": "'command' is required"}

        username = input_data.get("username", "root")
        port = int(input_data.get("port", 22))
        key_path = input_data.get("key_path")
        password = input_data.get("password")
        timeout = int(input_data.get("timeout", 30))

        connect_kwargs: dict[str, Any] = {
            "host": host,
            "port": port,
            "username": username,
            "known_hosts": None,
            "connect_timeout": timeout,
        }

        if key_path:
            connect_kwargs["client_keys"] = [key_path]
        if password:
            connect_kwargs["password"] = password

        try:
            async with asyncssh.connect(**connect_kwargs) as conn:
                result = await conn.run(command, check=False, timeout=timeout)

                return {
                    "host": host,
                    "command": command,
                    "stdout": result.stdout.strip() if result.stdout else "",
                    "stderr": result.stderr.strip() if result.stderr else "",
                    "exit_code": result.exit_status,
                }

        except asyncssh.DisconnectError as exc:
            return {"error": f"SSH disconnected: {exc}"}
        except asyncssh.PermissionDenied:
            return {"error": "SSH permission denied — check credentials"}
        except TimeoutError:
            return {"error": f"SSH connection to {host}:{port} timed out"}
        except Exception as exc:
            logger.error("ssh_error", host=host, error=str(exc))
            return {"error": str(exc)}
