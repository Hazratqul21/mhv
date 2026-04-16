from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def _register_agents(orchestrator, engine) -> None:
    """Instantiate and register every agent with the orchestrator."""
    from app.agents import (
        ChatAgent, CodeAgent, VisionAgent, VoiceAgent, SearchAgent, ToolAgent,
        RAGAgent, ResearchAgent, SummarizerAgent,
        TranslatorAgent, EmailAgent, CreativeAgent,
        MathAgent, DataAgent, SQLAgent, ShellAgent, SecurityAgent, DevOpsAgent, PlannerAgent,
        MetaAgent, ReflectionAgent, MemoryManagerAgent, WorkflowAgent,
        SchedulerAgent, MonitorAgent, LearningAgent, CollaborationAgent,
        VideoAgent, MusicAgent, VoiceCloneAgent, ArtDirectorAgent,
        SystemAdminAgent, GPUManagerAgent,
        AppBuilderAgent, FrontendAgent, BackendAgent,
        FineTuneAgent,
        APIAgent, ScrapingAgent, NotificationAgent, FileManagerAgent,
        ImageGenAgent, TestingAgent, DocumentAgent,
    )

    agent_classes = [
        ChatAgent, CodeAgent, VisionAgent, VoiceAgent, SearchAgent, ToolAgent,
        RAGAgent, ResearchAgent, SummarizerAgent,
        TranslatorAgent, EmailAgent, CreativeAgent,
        MathAgent, DataAgent, SQLAgent, ShellAgent, SecurityAgent, DevOpsAgent, PlannerAgent,
        MetaAgent, ReflectionAgent, MemoryManagerAgent, WorkflowAgent,
        SchedulerAgent, MonitorAgent, LearningAgent, CollaborationAgent,
        VideoAgent, MusicAgent, VoiceCloneAgent, ArtDirectorAgent,
        SystemAdminAgent, GPUManagerAgent,
        AppBuilderAgent, FrontendAgent, BackendAgent,
        FineTuneAgent,
        APIAgent, ScrapingAgent, NotificationAgent, FileManagerAgent,
        ImageGenAgent, TestingAgent, DocumentAgent,
    ]

    for cls in agent_classes:
        try:
            agent = cls(llm_engine=engine)
            orchestrator.register_agent(agent.name, agent)
        except Exception as exc:
            log.warning("agent_register_failed", agent=cls.__name__, error=str(exc))


def _register_tools(registry, settings) -> None:
    """Auto-discover and register tool classes from app.tools subpackages."""
    import importlib
    import pkgutil
    from app.tools.registry import BaseTool

    tools_pkg_path = Path(__file__).parent / "tools"
    for category_dir in sorted(tools_pkg_path.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("_"):
            continue
        pkg_name = f"app.tools.{category_dir.name}"
        try:
            pkg = importlib.import_module(pkg_name)
        except ImportError:
            continue

        for importer, modname, ispkg in pkgutil.iter_modules([str(category_dir)]):
            fqn = f"{pkg_name}.{modname}"
            try:
                mod = importlib.import_module(fqn)
                for attr_name in dir(mod):
                    obj = getattr(mod, attr_name)
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, BaseTool)
                        and obj is not BaseTool
                        and hasattr(obj, "name")
                        and obj.name
                    ):
                        try:
                            instance = obj()
                            registry.register(instance)
                        except Exception as exc:
                            log.debug("tool_instantiation_failed", tool=attr_name, error=str(exc))
            except Exception as exc:
                log.debug("tool_module_import_failed", module=fqn, error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle hook."""
    settings = get_settings()
    setup_logging(settings.log_level, json_output=settings.env != "development")
    log.info("miya_starting", env=settings.env)

    app.state.start_time = time.time()
    app.state.orchestrator = None
    app.state.memory = {}

    # ── Memory layer ────────────────────────────────────────────────────
    try:
        from app.memory.conversation_db import ConversationDB
        conv_db = ConversationDB()
        await conv_db.initialize()
        app.state.memory["conversation_db"] = conv_db
        log.info("conversation_db_ready")
    except Exception as exc:
        log.warning("conversation_db_failed", error=str(exc))

    try:
        from app.memory.vector_store import VectorMemory
        vector = VectorMemory(local_mode=(settings.env == "development"))
        app.state.memory["vector"] = vector
        log.info("vector_store_ready")
    except Exception as exc:
        log.warning("vector_store_failed", error=str(exc))

    try:
        from app.memory.cache_manager import CacheManager
        cache = CacheManager()
        await cache.connect()
        app.state.memory["cache"] = cache
        log.info("cache_ready")
    except Exception as exc:
        log.warning("cache_failed", error=str(exc))

    # ── Memory adapter ────────────────────────────────────────────────────
    try:
        from app.memory.adapter import MemoryAdapter
        memory_adapter = MemoryAdapter(app.state.memory)
        app.state.memory_adapter = memory_adapter
    except Exception as exc:
        log.warning("memory_adapter_failed", error=str(exc))
        memory_adapter = None

    # ── Core engine + orchestrator ──────────────────────────────────────
    try:
        from app.core.llm_engine import LLMEngine
        from app.core.orchestrator import MiyaOrchestrator
        from app.core.event_bus import EventBus
        from app.tools.registry import ToolRegistry

        engine = LLMEngine(settings)
        event_bus = EventBus()
        tool_registry = ToolRegistry()

        _register_tools(tool_registry, settings)

        orchestrator = MiyaOrchestrator(
            llm_engine=engine,
            event_bus=event_bus,
            tool_registry=tool_registry,
            memory=memory_adapter,
        )

        _register_agents(orchestrator, engine)

        try:
            await orchestrator.initialize()
        except Exception as exc:
            log.warning("orchestrator_initialize_skipped", error=str(exc))

        app.state.orchestrator = orchestrator
        app.state.engine = engine
        app.state.event_bus = event_bus
        log.info("orchestrator_ready", agents=len(orchestrator._agents))
    except Exception as exc:
        log.error("orchestrator_init_failed", error=str(exc))

    # ── LLM Router ─────────────────────────────────────────────────────
    try:
        from app.core.llm_router import LLMRouter
        app.state.llm_router = LLMRouter(engine)
        log.info("llm_router_ready")
    except Exception as exc:
        log.warning("llm_router_failed", error=str(exc))

    # ── Autonomous Systems ─────────────────────────────────────────────
    try:
        from app.core.autonomous.curiosity_engine import CuriosityEngine
        from app.core.autonomous.job_runner import JobRunner
        from app.core.autonomous.self_evolution import SelfEvolutionEngine
        from app.core.autonomous.self_healer import SelfHealer
        from app.core.autonomous.feedback_loop import FeedbackLoop

        conv_db = app.state.memory.get("conversation_db")
        vector = app.state.memory.get("vector")

        self_healer = SelfHealer(engine)
        app.state.self_healer = self_healer

        evolution = SelfEvolutionEngine(engine)
        app.state.evolution_engine = evolution

        feedback = FeedbackLoop(engine)
        app.state.feedback_loop = feedback

        job_runner = JobRunner(check_interval=60)

        if conv_db and engine:
            curiosity = CuriosityEngine(
                llm_engine=engine,
                conversation_db=conv_db,
                vector_memory=vector,
            )
            app.state.curiosity_engine = curiosity

            job_runner.register(
                name="curiosity_daily",
                coro_factory=curiosity.run_daily_cycle,
                interval_seconds=86400,
            )
            job_runner.register(
                name="curiosity_weekly",
                coro_factory=curiosity.run_weekly_cycle,
                interval_seconds=604800,
            )
            log.info("curiosity_engine_ready")

        job_runner.register(
            name="self_healer_check",
            coro_factory=self_healer._check_all,
            interval_seconds=60,
        )

        job_runner.register(
            name="evolution_cycle",
            coro_factory=evolution._run_cycle,
            interval_seconds=300,
        )

        job_runner.start()
        app.state.job_runner = job_runner
        if app.state.orchestrator:
            app.state.orchestrator.set_autonomous(
                evolution=evolution, healer=self_healer,
            )

        log.info("autonomous_systems_ready",
                 modules=["curiosity", "self_healer", "evolution", "feedback"])
    except Exception as exc:
        log.warning("autonomous_systems_failed", error=str(exc))

    # ── Static uploads directory ────────────────────────────────────────
    uploads_dir = settings.data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    yield

    # ── Shutdown ────────────────────────────────────────────────────────
    log.info("miya_shutting_down")

    job_runner = getattr(app.state, "job_runner", None)
    if job_runner:
        await job_runner.stop()

    healer = getattr(app.state, "self_healer", None)
    if healer:
        await healer.stop()

    if app.state.orchestrator:
        await app.state.orchestrator.shutdown()

    conv_db = app.state.memory.get("conversation_db")
    if conv_db:
        await conv_db.close()

    cache = app.state.memory.get("cache")
    if cache:
        await cache.close()

    log.info("miya_stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Miya AI",
        description="Self-hosted multi-agent AI assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── CORS ────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request-ID middleware ───────────────────────────────────────────
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # ── Exception handlers ─────────────────────────────────────────────
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return JSONResponse(
            status_code=404,
            content={"error": "Not Found", "detail": str(exc.detail if hasattr(exc, 'detail') else exc)},
        )

    @app.exception_handler(500)
    async def server_error_handler(request: Request, exc):
        rid = getattr(request.state, "request_id", "unknown")
        log.error("unhandled_error", request_id=rid, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "request_id": rid},
        )

    # ── Routes ─────────────────────────────────────────────────────────
    from app.api.routes import router as api_router
    from app.api.websocket import ws_router

    app.include_router(api_router, prefix="/api/v1", tags=["api"])
    app.include_router(ws_router, tags=["websocket"])

    # ── Convenience health on root ─────────────────────────────────────
    @app.get("/health")
    async def root_health():
        return {"status": "ok", "version": "0.1.0"}

    # ── Serve web frontend at root ─────────────────────────────────────
    web_dir = settings.project_root / "frontend" / "web"

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        from fastapi.responses import FileResponse
        ico = web_dir / "favicon.png"
        if ico.exists():
            return FileResponse(str(ico), media_type="image/png")
        return Response(status_code=204)

    @app.get("/")
    async def root_page():
        index = web_dir / "index.html"
        if index.exists():
            from fastapi.responses import HTMLResponse
            return HTMLResponse(index.read_text(encoding="utf-8"))
        return JSONResponse(
            content={
                "name": "Miya AI",
                "version": "0.1.0",
                "docs": "/docs",
                "health": "/health",
                "api": "/api/v1",
            }
        )

    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="web-static")

    # ── Mount uploads as static files ──────────────────────────────────
    uploads_path = settings.data_dir / "uploads"
    uploads_path.mkdir(parents=True, exist_ok=True)
    try:
        app.mount("/uploads", StaticFiles(directory=str(uploads_path)), name="uploads")
    except Exception:
        pass

    return app


app = create_app()
