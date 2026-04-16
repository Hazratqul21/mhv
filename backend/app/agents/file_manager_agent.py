from __future__ import annotations

from typing import Any, Callable, Coroutine

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.helpers import Timer
from app.utils.logger import get_logger

from .base_agent import AgentResult, BaseAgent
from .voice_context import voice_prepend_after_first_system

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Miya's file management specialist. You perform advanced file "
    "operations including organization, renaming, conversion, compression, "
    "and directory analysis.\n\n"
    "Capabilities:\n"
    "- Organize files by type, date, size, or custom rules\n"
    "- Batch rename with patterns (regex, sequential, date-based)\n"
    "- File format conversion (text encodings, line endings, etc.)\n"
    "- Compress/decompress archives (zip, tar.gz, 7z)\n"
    "- Search files with glob patterns, regex, and content matching\n"
    "- Directory structure analysis and visualization\n"
    "- Duplicate file detection\n"
    "- Disk usage analysis and cleanup recommendations\n\n"
    "Rules:\n"
    "- ALWAYS confirm before destructive operations (delete, overwrite)\n"
    "- Show a dry-run plan before batch operations\n"
    "- Preserve original files when converting (create copies)\n"
    "- Report summary statistics after batch operations\n"
    "- Use safe move/copy — never overwrite without warning\n\n"
    "Available tools:\n"
    "- file: File operations. "
    'Args: {"action": "read|write|list|move|copy|delete|stat", "path": "...", ...}\n'
    "- sandbox: Execute file management scripts. "
    'Args: {"language": "python", "code": "..."}\n\n'
    "To use a tool, output a JSON block:\n"
    '```json\n{"tool": "<name>", "args": {…}}\n```'
)

FILE_MANAGER_TOOLS = ["file", "sandbox"]


class FileManagerAgent(BaseAgent):
    """Advanced file operations agent with batch processing support."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="file_manager",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=FILE_MANAGER_TOOLS,
        )
        self._engine = llm_engine
        self._model_name = settings.chat_model

    def ensure_loaded(self) -> None:
        settings = get_settings()
        try:
            self._engine.get_model(self._model_name)
        except KeyError:
            self._engine.swap_model(self._model_name, n_ctx=settings.chat_ctx)

    async def execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        tool_executor: Callable[..., Coroutine[Any, Any, Any]] | None = None,
    ) -> AgentResult:
        logger.info("[FileManagerAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}

        all_tool_calls: list[dict[str, Any]] = []

        with Timer() as t:
            messages = self._build_messages(query, context)

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.2,
                )
            except Exception as exc:
                logger.exception("[FileManagerAgent] generation failed")
                return AgentResult(
                    success=False, output="", execution_time_ms=t.elapsed_ms, error=str(exc),
                )

            output = response.get("text", "") if isinstance(response, dict) else str(response)

            tool_calls = self._parse_tool_calls(output)
            if tool_calls and tool_executor:
                output, extra = await self._tool_loop(
                    messages, output, tool_calls, tool_executor
                )
                all_tool_calls.extend(extra)

        return AgentResult(
            success=True,
            output=output,
            tool_calls=all_tool_calls,
            token_usage={
                "prompt_tokens": response.get("prompt_tokens", 0),
                "completion_tokens": response.get("completion_tokens", 0),
            },
            execution_time_ms=t.elapsed_ms,
        )

    async def _tool_loop(
        self,
        messages: list[dict[str, str]],
        initial_output: str,
        tool_calls: list[dict[str, Any]],
        tool_executor: Any,
        max_rounds: int = 6,
    ) -> tuple[str, list[dict[str, Any]]]:
        all_calls: list[dict[str, Any]] = []
        output = initial_output

        for _ in range(max_rounds):
            if not tool_calls:
                break

            results: list[str] = []
            for call in tool_calls:
                name = call.get("tool") or call.get("name", "file")
                args = call.get("args") or call.get("arguments", {})
                all_calls.append({"tool": name, "args": args})
                try:
                    result = await tool_executor(name, args)
                    results.append(f"[{name}] Output:\n{result}")
                except Exception as exc:
                    logger.warning("[FileManagerAgent] tool %s failed: %s", name, exc)
                    results.append(f"[{name}] Error: {exc}")

            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": "File operation results:\n" + "\n\n".join(results)})

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.2,
                )
                output = response.get("text", "") if isinstance(response, dict) else str(response)
                tool_calls = self._parse_tool_calls(output)
            except Exception:
                logger.exception("[FileManagerAgent] follow-up generation failed")
                break

        return output, all_calls

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        voice_prepend_after_first_system(messages, context)

        working_dir = context.get("working_directory")
        if working_dir:
            messages.append({"role": "system", "content": f"Working directory: {working_dir}"})

        file_list = context.get("file_list")
        if file_list:
            listing = "\n".join(str(f) for f in file_list[:50])
            messages.append({"role": "system", "content": f"Current files:\n{listing}"})

        operation = context.get("operation")
        if operation:
            messages.append({"role": "system", "content": f"Requested operation: {operation}"})

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        return messages
