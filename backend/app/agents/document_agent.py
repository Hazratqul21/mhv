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
    "You are Miya's document generation specialist. You create structured, "
    "professional documents in various formats.\n\n"
    "Capabilities:\n"
    "- Create reports, proposals, specifications, and resumes\n"
    "- Output in markdown and HTML formats\n"
    "- Generate tables of contents automatically\n"
    "- Describe charts and data visualizations in text/markdown\n"
    "- Apply document templates for consistent formatting\n"
    "- Include headers, footers, numbered sections\n"
    "- Cross-reference sections and appendices\n\n"
    "Rules:\n"
    "- Follow the selected template structure precisely\n"
    "- Use professional, clear language appropriate to the audience\n"
    "- Number all sections and sub-sections hierarchically\n"
    "- Include an executive summary for long documents\n"
    "- Tables must be properly aligned and labeled\n"
    "- Cite sources when including external data or claims\n\n"
    "Available tools:\n"
    "- file: Save generated documents. "
    'Args: {"action": "write", "path": "...", "content": "..."}\n'
    "- sandbox: Execute code for data processing or chart descriptions. "
    'Args: {"language": "python", "code": "..."}\n\n'
    "To use a tool, output a JSON block:\n"
    '```json\n{"tool": "<name>", "args": {…}}\n```'
)

DOCUMENT_TOOLS = ["file", "sandbox"]

TEMPLATES = {
    "report": (
        "# {title}\n\n"
        "## Executive Summary\n\n"
        "## 1. Introduction\n\n"
        "## 2. Methodology\n\n"
        "## 3. Findings\n\n"
        "## 4. Analysis\n\n"
        "## 5. Recommendations\n\n"
        "## 6. Conclusion\n\n"
        "## Appendix\n"
    ),
    "proposal": (
        "# {title}\n\n"
        "## Executive Summary\n\n"
        "## 1. Problem Statement\n\n"
        "## 2. Proposed Solution\n\n"
        "## 3. Scope & Deliverables\n\n"
        "## 4. Timeline\n\n"
        "## 5. Budget\n\n"
        "## 6. Team & Qualifications\n\n"
        "## 7. Risk Assessment\n\n"
        "## 8. Conclusion\n"
    ),
    "spec": (
        "# {title}\n\n"
        "## 1. Overview\n\n"
        "## 2. Requirements\n\n"
        "### 2.1 Functional Requirements\n\n"
        "### 2.2 Non-Functional Requirements\n\n"
        "## 3. Architecture\n\n"
        "## 4. Data Model\n\n"
        "## 5. API Specification\n\n"
        "## 6. Security Considerations\n\n"
        "## 7. Testing Strategy\n\n"
        "## 8. Glossary\n"
    ),
    "resume": (
        "# {title}\n\n"
        "## Contact Information\n\n"
        "## Professional Summary\n\n"
        "## Experience\n\n"
        "## Education\n\n"
        "## Skills\n\n"
        "## Certifications\n\n"
        "## Projects\n"
    ),
}


class DocumentAgent(BaseAgent):
    """Structured document generation agent with template support."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="document",
            model_path=settings.chat_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=DOCUMENT_TOOLS,
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
        logger.info("[DocumentAgent] processing query (len=%d)", len(query))
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
                    temperature=0.4,
                )
            except Exception as exc:
                logger.exception("[DocumentAgent] generation failed")
                return AgentResult(
                    success=False, output="", execution_time_ms=t.elapsed_ms, error=str(exc),
                )

            output = response.get("text", "") if isinstance(response, dict) else str(response)

            tool_calls = self._parse_tool_calls(output)
            if tool_calls and tool_executor:
                for call in tool_calls:
                    name = call.get("tool") or call.get("name", "file")
                    args = call.get("args") or call.get("arguments", {})
                    all_tool_calls.append({"tool": name, "args": args})
                    try:
                        await tool_executor(name, args)
                    except Exception as exc:
                        logger.warning("[DocumentAgent] tool %s failed: %s", name, exc)

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

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        voice_prepend_after_first_system(messages, context)

        doc_type = context.get("document_type", "report")
        template = TEMPLATES.get(doc_type)
        if template:
            title = context.get("title", "Untitled Document")
            skeleton = template.format(title=title)
            messages.append({
                "role": "system",
                "content": f"Document template ({doc_type}):\n{skeleton}",
            })

        output_format = context.get("output_format", "markdown")
        messages.append({"role": "system", "content": f"Output format: {output_format}"})

        audience = context.get("audience")
        if audience:
            messages.append({"role": "system", "content": f"Target audience: {audience}"})

        data = context.get("data") or context.get("source_data")
        if data:
            data_text = str(data)[:4000]
            messages.append({"role": "system", "content": f"Source data:\n{data_text}"})

        outline = context.get("outline")
        if outline:
            messages.append({"role": "system", "content": f"Document outline:\n{outline}"})

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        return messages
