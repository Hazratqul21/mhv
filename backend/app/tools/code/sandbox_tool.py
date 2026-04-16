from __future__ import annotations

from typing import Any

import docker
from docker.errors import ContainerError, ImageNotFound, APIError

from app.config import get_settings
from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SandboxTool(BaseTool):
    name = "sandbox"
    description = "Execute Python code safely inside a Docker container with resource limits"
    category = "code"
    parameters = {
        "code": {
            "type": "string",
            "description": "Python code to execute",
        },
        "timeout": {
            "type": "integer",
            "description": "Execution timeout in seconds",
        },
        "memory_limit": {
            "type": "string",
            "description": "Memory limit (e.g. 256m, 512m)",
        },
        "packages": {
            "type": "array",
            "description": "Pip packages to install before execution",
        },
    }

    _IMAGE = "python:3.11-slim"

    def __init__(self) -> None:
        settings = get_settings()
        self._timeout = settings.sandbox_timeout
        self._memory_limit = settings.sandbox_memory_limit
        self._network_disabled = not settings.sandbox_network
        self._client: docker.DockerClient | None = None

    def _get_client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    async def execute(self, input_data: dict[str, Any]) -> Any:
        code = input_data.get("code", "")
        if not code:
            return {"error": "'code' is required"}

        timeout = int(input_data.get("timeout", self._timeout))
        mem_limit = input_data.get("memory_limit", self._memory_limit)
        packages = input_data.get("packages", [])

        install_cmd = ""
        if packages:
            pkg_str = " ".join(packages)
            install_cmd = f"pip install --quiet {pkg_str} && "

        full_cmd = f"bash -c '{install_cmd}python -c \"{_escape_for_shell(code)}\"'"

        try:
            client = self._get_client()
            container = client.containers.run(
                self._IMAGE,
                command=["bash", "-c", f"{install_cmd}python -c {_quote(code)}"],
                mem_limit=mem_limit,
                nano_cpus=1_000_000_000,
                network_disabled=self._network_disabled,
                remove=True,
                stdout=True,
                stderr=True,
                detach=False,
                timeout=timeout,
            )

            output = container.decode("utf-8") if isinstance(container, bytes) else str(container)
            return {
                "status": "ok",
                "output": output.strip(),
                "timeout": timeout,
            }

        except ContainerError as exc:
            stderr = exc.stderr.decode("utf-8") if exc.stderr else ""
            return {"error": "execution_failed", "stderr": stderr, "exit_code": exc.exit_status}
        except ImageNotFound:
            return {"error": f"Docker image '{self._IMAGE}' not found. Pull it first."}
        except APIError as exc:
            logger.error("sandbox_docker_api_error", error=str(exc))
            return {"error": str(exc)}
        except Exception as exc:
            logger.error("sandbox_error", error=str(exc))
            return {"error": str(exc)}


def _quote(code: str) -> str:
    """Wrap code for bash -c execution."""
    import shlex
    return shlex.quote(code)


def _escape_for_shell(code: str) -> str:
    return code.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
