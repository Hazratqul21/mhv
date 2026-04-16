from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)


class BaseTool(ABC):
    """Abstract base for every tool in the Miya tool system."""

    name: str = ""
    description: str = ""
    category: str = ""
    parameters: dict[str, Any] = {}

    @abstractmethod
    async def execute(self, input_data: dict[str, Any]) -> Any:
        ...

    def tool_descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Central registry that owns every tool instance."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            logger.warning("tool_overwritten", name=tool.name)
        self._tools[tool.name] = tool
        logger.info("tool_registered", name=tool.name, category=tool.category)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def get_by_category(self, category: str) -> list[BaseTool]:
        return [t for t in self._tools.values() if t.category == category]

    def list_all(self) -> list[BaseTool]:
        return list(self._tools.values())

    async def execute_chain(
        self,
        tool_names: list[str],
        input_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run tools in sequence, threading output → next input."""
        results: list[dict[str, Any]] = []
        current = input_data

        for name in tool_names:
            tool = self.get(name)
            if tool is None:
                entry = {"tool": name, "error": f"Tool '{name}' not found"}
                results.append(entry)
                break

            try:
                output = await tool.execute(current)
                results.append({"tool": name, "result": output})
                if isinstance(output, dict):
                    current = {**current, **output}
            except Exception as exc:
                logger.error("tool_chain_error", tool=name, error=str(exc))
                results.append({"tool": name, "error": str(exc)})
                break

        return results

    def get_tool_descriptions(self) -> list[dict[str, Any]]:
        """Return lightweight descriptors suitable for LLM system prompts."""
        return [t.tool_descriptor() for t in self._tools.values()]
