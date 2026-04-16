from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.logger import get_logger

log = get_logger(__name__)

AGENT_TEMPLATE = '''\
from __future__ import annotations

from typing import Any, Callable, Coroutine

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.helpers import Timer
from app.utils.logger import get_logger

from .base_agent import AgentResult, BaseAgent

logger = get_logger(__name__)

SYSTEM_PROMPT = """{system_prompt}"""


class {class_name}(BaseAgent):
    """{description}"""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="{agent_name}",
            model_path=settings.{model_field},
            system_prompt=SYSTEM_PROMPT,
            available_tools={tools},
        )
        self._engine = llm_engine
        self._model_name = settings.{model_field}

    def ensure_loaded(self) -> None:
        settings = get_settings()
        try:
            self._engine.get_model(self._model_name)
        except KeyError:
            self._engine.swap_model(self._model_name, n_ctx=settings.{ctx_field})

    async def execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        tool_executor: Callable[..., Coroutine[Any, Any, Any]] | None = None,
    ) -> AgentResult:
        logger.info("[{class_name}] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {{}}

        with Timer() as t:
            messages = [
                {{"role": "system", "content": self.system_prompt}},
            ]
            history = context.get("history", [])
            for msg in history[-6:]:
                messages.append({{"role": msg["role"], "content": msg["content"]}})
            messages.append({{"role": "user", "content": query}})

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.3,
                )
            except Exception as exc:
                logger.exception("[{class_name}] generation failed")
                return AgentResult(
                    success=False, output="", execution_time_ms=t.elapsed_ms, error=str(exc),
                )

        output = response.get("text", "") if isinstance(response, dict) else str(response)

        return AgentResult(
            success=True,
            output=output,
            token_usage={{
                "prompt_tokens": response.get("prompt_tokens", 0),
                "completion_tokens": response.get("completion_tokens", 0),
            }},
            execution_time_ms=t.elapsed_ms,
        )
'''

TOOL_TEMPLATE = '''\
from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


class {class_name}:
    """BaseTool: {description}"""

    name = "{tool_name}"
    description = "{description}"
    category = "{category}"
    parameters: dict[str, Any] = {parameters}

    async def execute(self, input_data: dict[str, Any]) -> Any:
        action = input_data.get("action", "")
        log.info("%s executing action=%s", self.name, action)
        try:
            # Generated tool logic
            return {{"success": True, "action": action, "result": "Not yet implemented"}}
        except Exception as exc:
            log.error("%s failed: %s", self.name, exc)
            return {{"success": False, "error": str(exc)}}
'''


class CodeWriter:
    """Generates Python source code for new agents and tools.

    The generated code is validated via syntax check and optionally
    linted before being written to disk.
    """

    def __init__(self, llm_engine: LLMEngine) -> None:
        self._engine = llm_engine
        self._settings = get_settings()
        self._agents_dir = self._settings.base_dir / "app" / "agents"
        self._tools_dir = self._settings.base_dir / "app" / "tools"

    async def create_agent(
        self,
        agent_name: str,
        class_name: str,
        description: str,
        system_prompt: str,
        model_field: str = "chat_model",
        ctx_field: str = "chat_ctx",
        tools: list[str] | None = None,
    ) -> dict[str, Any]:
        code = AGENT_TEMPLATE.format(
            agent_name=agent_name,
            class_name=class_name,
            description=description,
            system_prompt=system_prompt,
            model_field=model_field,
            ctx_field=ctx_field,
            tools=tools or [],
        )

        validation = await self._validate_syntax(code)
        if not validation["valid"]:
            enhanced = await self._fix_code(code, validation["error"])
            if enhanced:
                code = enhanced
                validation = await self._validate_syntax(code)

        if not validation["valid"]:
            return {"success": False, "error": validation["error"]}

        filepath = self._agents_dir / f"{agent_name}_agent.py"
        filepath.write_text(code, encoding="utf-8")

        await self._update_agents_init(agent_name, class_name)

        log.info("agent_created", name=agent_name, path=str(filepath))
        return {"success": True, "path": str(filepath), "class": class_name}

    async def create_tool(
        self,
        tool_name: str,
        class_name: str,
        description: str,
        category: str,
        parameters: dict | None = None,
    ) -> dict[str, Any]:
        code = TOOL_TEMPLATE.format(
            tool_name=tool_name,
            class_name=class_name,
            description=description,
            category=category,
            parameters=parameters or {},
        )

        validation = await self._validate_syntax(code)
        if not validation["valid"]:
            return {"success": False, "error": validation["error"]}

        cat_dir = self._tools_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        (cat_dir / "__init__.py").touch()

        filepath = cat_dir / f"{tool_name}_tool.py"
        filepath.write_text(code, encoding="utf-8")

        log.info("tool_created", name=tool_name, path=str(filepath))
        return {"success": True, "path": str(filepath), "class": class_name}

    async def generate_code_with_llm(self, spec: str) -> str:
        """Use the LLM to generate arbitrary Python code from a specification."""
        prompt = (
            "Write production-quality Python code for the following specification. "
            "Return ONLY the Python code, no explanations.\n\n"
            f"Specification:\n{spec}"
        )
        model = self._settings.code_model
        try:
            self._engine.get_model(model)
        except KeyError:
            self._engine.swap_model(model, n_ctx=self._settings.code_ctx)
        result = await self._engine.generate(
            model,
            prompt=prompt,
            max_tokens=4096,
            temperature=0.2,
        )
        text = result["text"].strip()
        if text.startswith("```python"):
            text = text[len("```python"):].strip()
        if text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text

    async def _validate_syntax(self, code: str) -> dict[str, Any]:
        try:
            compile(code, "<generated>", "exec")
            return {"valid": True, "error": None}
        except SyntaxError as exc:
            return {"valid": False, "error": f"Line {exc.lineno}: {exc.msg}"}

    async def _fix_code(self, code: str, error: str) -> str | None:
        prompt = (
            f"Fix this Python syntax error: {error}\n\n"
            f"Code:\n```python\n{code}\n```\n\n"
            "Return ONLY the fixed Python code."
        )
        try:
            result = await self._engine.generate(
                self._settings.code_model,
                prompt=prompt,
                max_tokens=4096,
                temperature=0.1,
            )
            fixed = result["text"].strip()
            if fixed.startswith("```"):
                fixed = fixed.split("\n", 1)[1] if "\n" in fixed else fixed[3:]
            if fixed.endswith("```"):
                fixed = fixed[:-3]
            return fixed.strip()
        except Exception:
            return None

    async def _update_agents_init(self, agent_name: str, class_name: str) -> None:
        init_path = self._agents_dir / "__init__.py"
        if not init_path.exists():
            return

        content = init_path.read_text(encoding="utf-8")
        if class_name in content:
            return

        import_line = f"from .{agent_name}_agent import {class_name}"
        lines = content.split("\n")

        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("from .") or line.startswith("import "):
                insert_idx = i + 1

        lines.insert(insert_idx, import_line)

        content_new = "\n".join(lines)

        if "__all__" in content_new:
            import re
            match = re.search(r"(__all__\s*=\s*\[)", content_new)
            if match:
                all_start = content_new.find("[", match.start())
                depth = 0
                all_end = -1
                for i in range(all_start, len(content_new)):
                    if content_new[i] == "[":
                        depth += 1
                    elif content_new[i] == "]":
                        depth -= 1
                        if depth == 0:
                            all_end = i
                            break
                if all_end > 0:
                    content_new = (
                        content_new[:all_end]
                        + f'    "{class_name}",\n'
                        + content_new[all_end:]
                    )

        init_path.write_text(content_new, encoding="utf-8")
