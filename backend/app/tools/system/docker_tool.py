from __future__ import annotations

from typing import Any

import docker
from docker.errors import NotFound, APIError

from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DockerTool(BaseTool):
    name = "docker"
    description = "Docker management: list, run, stop, and inspect containers"
    category = "system"
    parameters = {
        "action": {
            "type": "string",
            "enum": ["list_containers", "run", "stop", "remove", "logs", "inspect"],
            "description": "Docker operation",
        },
        "image": {
            "type": "string",
            "description": "Image name for run",
        },
        "container_id": {
            "type": "string",
            "description": "Container ID or name",
        },
        "command": {
            "type": "string",
            "description": "Command to run in the container",
        },
        "environment": {
            "type": "object",
            "description": "Environment variables for run",
        },
        "detach": {
            "type": "boolean",
            "description": "Run container in background",
            "default": True,
        },
        "all": {
            "type": "boolean",
            "description": "Include stopped containers in list",
            "default": False,
        },
        "tail": {
            "type": "integer",
            "description": "Number of log lines to return",
            "default": 100,
        },
    }

    def __init__(self) -> None:
        self._client: docker.DockerClient | None = None

    def _get_client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    async def execute(self, input_data: dict[str, Any]) -> Any:
        action = input_data.get("action", "")

        try:
            client = self._get_client()

            if action == "list_containers":
                show_all = input_data.get("all", False)
                containers = client.containers.list(all=show_all)
                return {
                    "containers": [
                        {
                            "id": c.short_id,
                            "name": c.name,
                            "image": str(c.image.tags[0]) if c.image.tags else str(c.image.id[:12]),
                            "status": c.status,
                            "ports": c.ports,
                        }
                        for c in containers
                    ],
                    "count": len(containers),
                }

            if action == "run":
                image = input_data.get("image", "")
                if not image:
                    return {"error": "'image' is required"}
                command = input_data.get("command")
                env = input_data.get("environment", {})
                detach = input_data.get("detach", True)

                container = client.containers.run(
                    image,
                    command=command,
                    environment=env,
                    detach=detach,
                    remove=not detach,
                )
                if detach:
                    return {
                        "status": "running",
                        "container_id": container.short_id,
                        "name": container.name,
                    }
                output = container.decode("utf-8") if isinstance(container, bytes) else str(container)
                return {"status": "completed", "output": output}

            if action == "stop":
                cid = input_data.get("container_id", "")
                if not cid:
                    return {"error": "'container_id' is required"}
                container = client.containers.get(cid)
                container.stop()
                return {"status": "stopped", "container_id": cid}

            if action == "remove":
                cid = input_data.get("container_id", "")
                if not cid:
                    return {"error": "'container_id' is required"}
                container = client.containers.get(cid)
                container.remove(force=True)
                return {"status": "removed", "container_id": cid}

            if action == "logs":
                cid = input_data.get("container_id", "")
                if not cid:
                    return {"error": "'container_id' is required"}
                tail = int(input_data.get("tail", 100))
                container = client.containers.get(cid)
                logs = container.logs(tail=tail).decode("utf-8", errors="replace")
                return {"container_id": cid, "logs": logs}

            if action == "inspect":
                cid = input_data.get("container_id", "")
                if not cid:
                    return {"error": "'container_id' is required"}
                container = client.containers.get(cid)
                return {
                    "id": container.id,
                    "name": container.name,
                    "status": container.status,
                    "image": str(container.image.tags),
                    "created": str(container.attrs.get("Created", "")),
                    "ports": container.ports,
                    "labels": container.labels,
                }

            return {"error": f"Unknown action '{action}'"}

        except NotFound as exc:
            return {"error": f"Container not found: {exc}"}
        except APIError as exc:
            logger.error("docker_api_error", error=str(exc))
            return {"error": str(exc)}
        except Exception as exc:
            logger.error("docker_error", action=action, error=str(exc))
            return {"error": str(exc)}
