from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Optional

from app.agents.voice_context import voice_prepend_after_first_system
from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.core.event_bus import EventBus
from app.utils.logger import get_logger
from app.utils.helpers import Timer, generate_session_id

log = get_logger(__name__)

AGENT_SELECTION_PROMPT = """\
You are Miya, an autonomous AI orchestrator. Given the user's query and context, \
select the best agent. For complex multi-step goals, select "autopilot".

=== CORE AGENTS ===
- chat: General conversation, Q&A, casual questions
- code: Programming, debugging, code review, implementation
- vision: Image analysis, OCR, visual Q&A
- voice: Speech-to-text, text-to-speech
- search: Quick web search and info retrieval
- tool: Dynamic tool selection and chaining

=== KNOWLEDGE & RESEARCH ===
- rag: Document Q&A from knowledge base
- research: Deep multi-source research with citations
- summarizer: Summarize text, documents, conversations

=== LANGUAGE & COMMUNICATION ===
- translator: Translate between 20+ languages
- email: Professional email drafting and replies
- creative: Stories, poems, lyrics, marketing copy

=== TECHNICAL ===
- math: Math problems, equations, statistics, calculus
- data: Data analysis, CSV/JSON, pandas, ML
- sql: SQL queries, schema design, optimization
- shell: Shell commands, system administration
- security: Security audit, vulnerability scanning
- devops: Docker, K8s, CI/CD, infrastructure
- planner: Task planning, project decomposition

=== AUTONOMOUS / META ===
- autopilot: Complex multi-step goals (auto-decomposes and executes)
- meta: Create/configure new agents dynamically
- reflection: Evaluate and improve response quality
- memory_mgr: Long-term memory management
- workflow: Multi-step automated workflows
- scheduler: Scheduled/recurring tasks
- monitor: System health monitoring
- learning: Learn from feedback, improve over time
- collaboration: Coordinate multiple agents on one task

=== MEDIA ===
- video: Video generation (CogVideoX, AnimateDiff)
- music: Music/audio generation (MusicGen, AudioCraft)
- voice_clone: Voice cloning (RVC) and multi-voice TTS (Bark)
- art_director: Multi-media coordinator (image+video+audio projects)
- image_gen: Image generation via ComfyUI/Flux/SDXL

=== PC HOST / SYSTEM ===
- system_admin: PC system administration (packages, services, disk, network)
- gpu_manager: GPU/VRAM resource management and model loading optimization

=== APP BUILDER ===
- app_builder: Full-stack application generation (web/desktop/API)
- frontend: Frontend development (React, Vue, Svelte, Tailwind)
- backend: Backend development (FastAPI, Flask, Express, databases)

=== FINE-TUNING ===
- finetune: Model fine-tuning lifecycle (collect data, train, evaluate, convert GGUF, deploy)

=== SPECIALIZED ===
- api_agent: External REST/GraphQL API calls
- scraping: Intelligent web scraping
- notification: Alerts and notifications
- file_manager: File organization and batch operations
- testing: Generate and run tests (pytest)
- document: Create reports, proposals, specs

Available tool categories: memory, search, browser, media, code, data, system, ai

Respond ONLY with JSON:
{{"agent": "<name>", "tools": ["tool1", "tool2"], "confidence": 0.0-1.0, "reasoning": "..."}}

User query: {query}
Context: {context}
"""


class MiyaOrchestrator:
    """Central controller that routes queries to specialized agents.

    Supports two modes:
    - **Standard**: Single agent handles the query
    - **AutoPilot**: Complex goals decomposed into multi-step plans
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        event_bus: EventBus,
        agents: Optional[dict] = None,
        tool_registry: Optional[Any] = None,
        memory: Optional[Any] = None,
    ) -> None:
        self._engine = llm_engine
        self._event_bus = event_bus
        self._agents = agents or {}
        self._tools = tool_registry
        self._memory = memory
        self._settings = get_settings()
        self._autopilot: Optional[Any] = None
        self._evolution: Optional[Any] = None
        self._healer: Optional[Any] = None

    def set_autonomous(self, evolution=None, healer=None) -> None:
        """Wire autonomous subsystems after initialization."""
        self._evolution = evolution
        self._healer = healer

    async def initialize(self) -> None:
        log.info("orchestrator_initializing")
        try:
            self._engine.load_model(
                self._settings.orchestrator_model,
                n_ctx=self._settings.orchestrator_ctx,
            )
        except Exception as exc:
            log.warning("orchestrator_model_load_skipped",
                        model=self._settings.orchestrator_model,
                        error=str(exc),
                        note="keyword routing will be used as primary selector")

        try:
            from app.core.autonomous.autopilot import AutoPilot
            from app.core.autonomous.agent_protocol import AgentProtocol

            protocol = AgentProtocol()
            for name in self._agents:
                protocol.register(name)

            self._autopilot = AutoPilot(
                llm_engine=self._engine,
                event_bus=self._event_bus,
                agents=self._agents,
                tool_registry=self._tools,
                protocol=protocol,
            )
            log.info("autopilot_ready")
        except Exception as exc:
            log.warning("autopilot_init_skipped", error=str(exc))

        await self._event_bus.emit("orchestrator.ready")
        log.info("orchestrator_ready", agents=len(self._agents))

    def register_agent(self, name: str, agent: Any) -> None:
        self._agents[name] = agent
        log.info("agent_registered", name=name)

    async def process(
        self,
        query: str,
        session_id: Optional[str] = None,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> dict:
        session_id = session_id or generate_session_id()
        timer = Timer()

        with timer:
            await self._event_bus.emit(
                "query.received", query=query, session_id=session_id
            )

            context = await self._get_context(session_id, query)
            if extra_context:
                for key, val in extra_context.items():
                    if val is not None:
                        context[key] = val
            selection = await self._select_agent(query, context)
            agent_name = selection.get("agent", "chat")
            requested_tools = selection.get("tools", [])

            log.info(
                "agent_selected",
                agent=agent_name,
                tools=requested_tools,
                confidence=selection.get("confidence"),
            )

            if agent_name == "autopilot" and self._autopilot:
                result = await self._autopilot.execute_goal(
                    goal=query, session_id=session_id,
                    context=json.dumps(context.get("history", [])[:3], default=str),
                )
            else:
                result = await self._execute_agent(
                    agent_name, query, context, requested_tools
                )

            if self._memory:
                try:
                    await self._memory.store(session_id, query, result)
                except Exception:
                    pass

        response = {
            "response": result.get("output", result.get("response", "I could not process that request.")),
            "agent_used": result.get("agent_used", agent_name),
            "tools_used": result.get("tools_used", []),
            "confidence": selection.get("confidence", 0.0),
            "session_id": session_id,
            "execution_time_ms": timer.elapsed_ms,
        }

        asyncio.ensure_future(self._post_response_eval(
            agent_name, query, response["response"]
        ))

        await self._event_bus.emit("query.completed", **response)
        return response

    async def _post_response_eval(
        self, agent_name: str, query: str, response_text: str,
    ) -> None:
        """Background evaluation — feeds SelfEvolution and reports quality gaps."""
        if not self._evolution:
            return
        try:
            eval_result = await self._evolution.evaluate_and_improve(
                agent_name=agent_name,
                query=query,
                response=response_text,
            )
            if eval_result.get("score", 1.0) < 0.4:
                log.info("low_quality_detected", agent=agent_name,
                         score=eval_result["score"])
        except Exception:
            pass

    async def _get_context(self, session_id: str, query: str) -> dict:
        context: dict[str, Any] = {"session_id": session_id}
        if self._memory:
            try:
                history = await self._memory.get_recent(session_id, limit=10)
                context["history"] = history
                relevant = await self._memory.search_relevant(query, limit=5)
                context["relevant"] = relevant
            except Exception as exc:
                log.warning("context_retrieval_failed", error=str(exc))
        return context

    def _keyword_route(self, query: str) -> dict | None:
        """Fast keyword-based agent selection — no LLM needed."""
        q = query.lower()

        ROUTES: list[tuple[str, list[str], list[str]]] = [
            ("code", ["write code", "debug", "implement", "function", "class ", "refactor",
                       "python", "javascript", "typescript", "rust ", "java ", "golang",
                       "fix bug", "code review", "programming", "algorithm", "kod yoz",
                       "dasturlash", "xato tuzat"], ["code_sandbox"]),
            ("shell", ["run command", "terminal", "bash", "shell", "apt ", "pip ",
                        "systemctl", "docker run", "sudo ", "mkdir", "chmod",
                        "buyruq", "terminal"], ["shell_exec"]),
            ("search", ["search", "find info", "google", "look up", "what is", "who is",
                          "when was", "qidir", "izla", "axborot"], ["web_search"]),
            ("research", ["research", "analyze", "deep dive", "investigate", "compare",
                            "tadqiqot", "tahlil"], ["web_search"]),
            ("math", ["calculate", "math", "equation", "integral", "derivative",
                       "statistics", "hisoblash", "matematik", "tengla"], []),
            ("sql", ["sql", "query", "database", "table", "select ", "insert ",
                      "join "], ["sql_exec"]),
            ("translator", ["translate", "tarjima", "перевод", "tercüme"], []),
            ("creative", ["write a story", "poem", "lyrics", "creative writing",
                            "hikoya", "she'r", "qo'shiq"], []),
            ("summarizer", ["summarize", "summary", "tldr", "qisqacha", "xulosa"], []),
            ("email", ["write email", "draft email", "reply to email", "xat yoz"], []),
            ("data", ["analyze data", "csv", "dataframe", "pandas", "dataset",
                        "ma'lumot tahlil"], ["pandas_tool"]),
            ("vision", ["image", "photo", "picture", "rasm", "surat", "what do you see",
                          "describe image", "ocr"], []),
            ("planner", ["plan", "roadmap", "break down", "decompose", "reja",
                           "loyiha reja"], []),
            ("devops", ["deploy", "kubernetes", "k8s", "ci/cd", "pipeline",
                          "nginx", "infrastructure"], []),
            ("security", ["security", "vulnerability", "pentest", "xavfsizlik",
                            "scan", "audit security"], []),
            ("finetune", ["fine-tune", "finetune", "train model", "lora",
                            "model o'rgatish"], []),
            ("system_admin", ["system info", "disk space", "memory usage", "cpu",
                                "install package", "tizim", "o'rnatish"], ["shell_exec"]),
            ("gpu_manager", ["gpu", "vram", "cuda", "model load", "unload model"], []),
            ("video", ["generate video", "video yarat", "animate"], []),
            ("music", ["generate music", "musiqa", "audio generate"], []),
            ("image_gen", ["generate image", "rasm yarat", "create image"], []),
            ("file_manager", ["file", "rename", "move file", "copy file",
                                "fayl", "papka"], ["file_ops"]),
            ("app_builder", ["build app", "create app", "ilova yarat",
                               "web app", "full-stack"], []),
            ("document", ["write document", "report", "proposal", "hujjat yoz",
                            "hisobot"], []),
            ("testing", ["write test", "pytest", "unit test", "test yoz"], []),
            ("voice", ["voice", "speech", "ovoz", "gapir", "say aloud", "speak"], []),
            ("rag", ["from my documents", "knowledge base", "hujjatlardan", "bilim bazasi"], []),
            ("monitor", ["system health", "monitoring", "tizim salomatligi"], []),
            ("autopilot", ["multi-step", "complex goal", "murakkab vazifa",
                             "step by step do", "autonomous"], []),
            ("scraping", ["scrape", "web scraping", "crawl", "skreyping"], []),
            ("notification", ["notify", "alert", "bildirishnoma", "ogohlantir"], []),
            ("frontend", ["react", "vue", "svelte", "tailwind", "css", "html"], []),
            ("backend", ["fastapi", "flask", "express", "backend develop"], []),
            ("voice_clone", ["clone voice", "rvc", "ovoz klonlash"], []),
        ]

        for agent, keywords, tools in ROUTES:
            if any(kw in q for kw in keywords):
                if agent in self._agents:
                    return {"agent": agent, "tools": tools,
                            "confidence": 0.85, "reasoning": "keyword_match"}
        return None

    def _simple_chat_fast_path(self, query: str) -> bool:
        """Skip orchestrator LLM routing for short, non-specialist lines.

        Saves VRAM/time (one fewer completion + model swap) — important when
        the voice client and backend share one GPU.
        """
        q = (query or "").strip()
        if not q or len(q) > 160:
            return False
        words = q.split()
        if len(words) > 10:
            return False
        low = q.lower()
        if "http://" in low or "https://" in low or "```" in q:
            return False
        # Obvious specialist intents — still use orchestrator if no keyword hit
        agentish = (
            "fix my", "deploy ", "refactor", "write a function", "write a class",
            "database", "kubernetes", "dockerfile", "nginx", "terraform",
            "fine-tune", "train the model", "pytest", "unit test",
        )
        if any(p in low for p in agentish):
            return False
        return True

    async def _select_agent(self, query: str, context: dict) -> dict:
        keyword_result = self._keyword_route(query)
        if keyword_result:
            log.info("agent_routed_by_keyword", agent=keyword_result["agent"])
            return keyword_result

        if (
            context.get("miya_client") != "voice"
            and self._simple_chat_fast_path(query)
            and "chat" in self._agents
        ):
            log.info(
                "agent_routed_simple_chat",
                words=len(query.split()),
                reason="skip_orchestrator_llm",
            )
            return {
                "agent": "chat",
                "tools": [],
                "confidence": 0.55,
                "reasoning": "simple_query_fast_path",
            }

        sel_ctx = json.dumps(context.get("history", [])[:3], default=str)
        if context.get("miya_client") == "voice":
            sel_ctx += (
                "\n[Client: desktop VOICE on the user's PC — they may ask for code, "
                "shell commands, web search, opening paths, file tasks; pick the best "
                "matching specialist agent (code, shell, search, …), not generic chat, "
                "when the query implies tools or implementation.]"
            )

        try:
            result = await self._engine.generate(
                self._settings.orchestrator_model,
                prompt=AGENT_SELECTION_PROMPT.format(
                    query=query,
                    context=sel_ctx,
                ),
                max_tokens=256,
                temperature=0.1,
            )
            text = result["text"].strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except (json.JSONDecodeError, Exception) as exc:
            log.warning("agent_selection_fallback", error=str(exc))

        return {"agent": "chat", "tools": [], "confidence": 0.5, "reasoning": "fallback"}

    async def _call_tool(self, tool_name: str, args: dict) -> Any:
        """Callable wrapper around ToolRegistry for agent tool execution."""
        if not self._tools:
            return f"Error: No tool registry available"
        tool = self._tools.get(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found"
        try:
            result = await tool.execute(args)
            if isinstance(result, dict):
                return json.dumps(result, default=str)
            return str(result)
        except Exception as exc:
            return f"Error executing {tool_name}: {exc}"

    async def _execute_agent(
        self,
        agent_name: str,
        query: str,
        context: dict,
        tools: list[str],
    ) -> dict:
        agent = self._agents.get(agent_name)
        if not agent:
            log.warning("agent_not_found", name=agent_name)
            agent = self._agents.get("chat")
            if not agent:
                return {"output": "No agents available.", "tools_used": []}

        if hasattr(agent, "ensure_loaded"):
            try:
                agent.ensure_loaded()
            except Exception as exc:
                log.warning("agent_model_load_failed", agent=agent_name, error=str(exc))

        try:
            result = await asyncio.wait_for(
                agent.execute(
                    query=query,
                    context=context,
                    tool_executor=self._call_tool,
                ),
                timeout=300,
            )
            return {
                "output": result.output,
                "tools_used": [tc.get("tool", "") for tc in result.tool_calls],
                "token_usage": result.token_usage,
            }
        except asyncio.TimeoutError:
            log.error("agent_timeout", agent=agent_name, timeout_s=300)
            return {
                "output": f"Agent '{agent_name}' timed out after 5 minutes. Try a simpler query.",
                "tools_used": [],
            }
        except Exception as exc:
            log.error("agent_execution_failed", agent=agent_name, error=str(exc))
            if self._healer:
                asyncio.ensure_future(self._healer.handle_error(
                    source=f"agent:{agent_name}",
                    error=str(exc),
                    context={"query": query[:200], "agent": agent_name},
                ))
            return {
                "output": f"An error occurred while processing your request: {exc}",
                "tools_used": [],
            }

    async def process_stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """Streaming variant of process — yields tokens as they arrive."""
        session_id = session_id or generate_session_id()

        await self._event_bus.emit(
            "query.received", query=query, session_id=session_id
        )

        context = await self._get_context(session_id, query)
        if extra_context:
            for key, val in extra_context.items():
                if val is not None:
                    context[key] = val
        selection = await self._select_agent(query, context)
        agent_name = selection.get("agent", "chat")

        agent = self._agents.get(agent_name) or self._agents.get("chat")
        if not agent:
            yield "No agents available."
            return

        model_file = getattr(agent, "model_path", self._settings.chat_model)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": getattr(agent, "system_prompt", "You are Miya, a helpful AI assistant.")},
        ]
        voice_prepend_after_first_system(messages, context)
        if context.get("reply_language"):
            lang = str(context["reply_language"]).strip()
            if lang:
                messages.append({
                    "role": "system",
                    "content": (
                        f"User preference (voice/session): reply **only in {lang}** for this turn, "
                        "clearly and naturally. Ignore the general 'same language as user message' rule "
                        "if it conflicts — this directive wins."
                    ),
                })
        for msg in context.get("history", [])[-10:]:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})

        try:
            self._engine.swap_model(model_file, n_ctx=self._settings.chat_ctx)
        except Exception as exc:
            yield f"Model load failed: {exc}"
            return

        collected: list[str] = []
        async for token in self._engine.chat_stream(
            model_file,
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
        ):
            collected.append(token)
            yield token

        full_response = "".join(collected)
        if self._memory:
            try:
                await self._memory.store(session_id, query, {"output": full_response})
            except Exception:
                pass

        await self._event_bus.emit(
            "query.completed",
            response=full_response,
            agent_used=agent_name,
            session_id=session_id,
        )

    async def shutdown(self) -> None:
        log.info("orchestrator_shutting_down")
        self._engine.unload_all()
        await self._event_bus.emit("orchestrator.shutdown")
