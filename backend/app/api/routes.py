from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, status

from app.api.auth import get_current_user
from app.api.models import (
    AgentInfo,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    FineTuneCollectRequest,
    FineTuneConvertRequest,
    FineTuneDeployRequest,
    FineTuneEvalRequest,
    FineTuneJobResponse,
    FineTuneStartRequest,
    HealthResponse,
    SessionInfo,
    ToolInfo,
    UploadResponse,
)
from app.config import get_settings
from app.utils.logger import get_logger
from app.utils.public_errors import safe_error_detail
from app.version import MIYA_VERSION

log = get_logger(__name__)

router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
CHAT_PROCESS_TIMEOUT = 360.0  # seconds (WebSocket bilan bir xil)


def _get_orchestrator(request: Request):
    orch = request.app.state.orchestrator
    if orch is None:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")
    return orch


def _get_memory(request: Request):
    return getattr(request.app.state, "memory", None)


# ── Chat ────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, responses={500: {"model": ErrorResponse}})
async def chat(
    body: ChatRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    orchestrator = _get_orchestrator(request)

    try:
        extra: dict = {}
        if body.reply_language:
            extra["reply_language"] = body.reply_language.strip()
        if body.context:
            for k, v in body.context.items():
                if v is not None:
                    extra[str(k)] = v
        result = await asyncio.wait_for(
            orchestrator.process(
                query=body.message,
                session_id=body.session_id,
                extra_context=extra or None,
            ),
            timeout=CHAT_PROCESS_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.error("chat_timeout", session_id=body.session_id)
        raise HTTPException(
            status_code=504,
            detail="Chat processing timed out. Try a shorter message or a simpler task.",
        ) from None
    except Exception as exc:
        log.error("chat_error", error=str(exc))
        settings = get_settings()
        raise HTTPException(
            status_code=500,
            detail=safe_error_detail(settings.env, exc),
        ) from exc

    return ChatResponse(**result)


# ── Health ──────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    start_time: float = getattr(request.app.state, "start_time", time.time())
    services: dict[str, str] = {}

    memory = _get_memory(request)
    if memory:
        services["vector_store"] = "ok" if memory.get("vector") else "unavailable"
        services["conversation_db"] = "ok" if memory.get("conversation_db") else "unavailable"
        services["cache"] = "ok" if memory.get("cache") else "unavailable"

    return HealthResponse(
        status="ok",
        version=MIYA_VERSION,
        uptime_seconds=round(time.time() - start_time, 2),
        services=services,
    )


# ── Agents ──────────────────────────────────────────────────────────────────

@router.get("/agents", response_model=list[AgentInfo])
async def list_agents(
    request: Request,
    user: dict = Depends(get_current_user),
):
    orchestrator = _get_orchestrator(request)
    agents = getattr(orchestrator, "_agents", {})
    results: list[AgentInfo] = []
    for name, agent in agents.items():
        results.append(AgentInfo(
            name=name,
            model=getattr(agent, "model_path", ""),
            status="ready",
            description=getattr(agent, "system_prompt", "")[:120],
        ))
    return results


# ── Tools ───────────────────────────────────────────────────────────────────

@router.get("/tools", response_model=list[ToolInfo])
async def list_tools(
    request: Request,
    user: dict = Depends(get_current_user),
):
    orchestrator = _get_orchestrator(request)
    registry = getattr(orchestrator, "_tools", None)
    if registry is None:
        return []
    return [
        ToolInfo(**desc)
        for desc in registry.get_tool_descriptions()
    ]


# ── Sessions ────────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    memory = _get_memory(request)
    if not memory or not memory.get("conversation_db"):
        raise HTTPException(status_code=404, detail="Session tracking unavailable")

    conv_db = memory["conversation_db"]
    history = await conv_db.get_history(session_id, limit=1)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")

    sessions = await conv_db.get_sessions(limit=500)
    for s in sessions:
        if s["session_id"] == session_id:
            return SessionInfo(**s)

    return SessionInfo(session_id=session_id, message_count=len(history))


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    memory = _get_memory(request)
    if not memory or not memory.get("conversation_db"):
        raise HTTPException(status_code=404, detail="Session tracking unavailable")

    await memory["conversation_db"].delete_session(session_id)


# ── File Upload ─────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    if file.size and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    settings = get_settings()
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename or "upload").name
    dest = upload_dir / f"{int(time.time())}_{safe_name}"

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    dest.write_bytes(content)
    log.info("file_uploaded", filename=safe_name, size=len(content))

    return UploadResponse(
        filename=safe_name,
        size_bytes=len(content),
        content_type=file.content_type or "application/octet-stream",
        path=str(dest),
    )


# ── Metrics ─────────────────────────────────────────────────────────────────

@router.get("/metrics")
async def metrics(request: Request):
    """Prometheus-style plaintext metrics."""
    start_time: float = getattr(request.app.state, "start_time", time.time())
    uptime = time.time() - start_time

    orchestrator = _get_orchestrator(request)
    agents = getattr(orchestrator, "_agents", {})
    event_bus = getattr(orchestrator, "_event_bus", None)

    lines = [
        "# HELP miya_uptime_seconds Time since application start",
        "# TYPE miya_uptime_seconds gauge",
        f"miya_uptime_seconds {uptime:.2f}",
        "",
        "# HELP miya_agents_total Number of registered agents",
        "# TYPE miya_agents_total gauge",
        f"miya_agents_total {len(agents)}",
    ]

    if event_bus:
        history = event_bus.get_history("query.completed", limit=10_000)
        lines += [
            "",
            "# HELP miya_queries_total Total processed queries",
            "# TYPE miya_queries_total counter",
            f"miya_queries_total {len(history)}",
        ]

    return "\n".join(lines) + "\n"


# ── Fine-Tuning ──────────────────────────────────────────────────────────────

def _get_finetune_components(request: Request):
    from app.core.finetune import (
        DataCollector,
        DataFormatter,
        FineTuneTrainer,
        GGUFConverter,
        ModelDeployer,
        ModelEvaluator,
    )
    ft = getattr(request.app.state, "_finetune", None)
    if ft is None:
        ft = {
            "collector": DataCollector(),
            "formatter": DataFormatter(),
            "trainer": FineTuneTrainer(),
            "evaluator": ModelEvaluator(),
            "converter": GGUFConverter(),
            "deployer": ModelDeployer(),
        }
        request.app.state._finetune = ft
    return ft


@router.post("/finetune/collect")
async def finetune_collect(
    body: FineTuneCollectRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    try:
        ft = _get_finetune_components(request)
        samples = await ft["collector"].collect_all(
            custom_datasets=body.custom_datasets or None,
            min_quality=body.min_quality,
        )
        return {"collected": len(samples)}
    except Exception as exc:
        log.error("finetune_collect_error", error=str(exc))
        settings = get_settings()
        raise HTTPException(
            status_code=500,
            detail=safe_error_detail(settings.env, exc),
        ) from exc


@router.post("/finetune/train", response_model=FineTuneJobResponse)
async def finetune_train(
    body: FineTuneStartRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    ft = _get_finetune_components(request)
    trainer = ft["trainer"]
    if trainer.is_training:
        raise HTTPException(status_code=409, detail="A training job is already running")

    job = await trainer.start_training(
        base_model_hf=body.base_model,
        dataset_path=body.dataset_path,
        output_name=body.output_name,
        lora_rank=body.lora_rank,
        lora_alpha=body.lora_alpha,
        batch_size=body.batch_size,
        epochs=body.epochs,
        learning_rate=body.learning_rate,
        max_seq_length=body.max_seq_length,
        load_in_4bit=body.load_in_4bit,
    )
    return FineTuneJobResponse(
        job_id=job.job_id,
        status=job.status,
        base_model=job.base_model,
        started_at=job.started_at,
    )


@router.get("/finetune/jobs")
async def finetune_list_jobs(
    request: Request,
    user: dict = Depends(get_current_user),
):
    ft = _get_finetune_components(request)
    return ft["trainer"].list_jobs()


@router.get("/finetune/jobs/{job_id}", response_model=FineTuneJobResponse)
async def finetune_job_status(
    job_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    ft = _get_finetune_components(request)
    job = ft["trainer"].get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return FineTuneJobResponse(
        job_id=job.job_id,
        status=job.status,
        base_model=job.base_model,
        started_at=job.started_at,
        completed_at=job.completed_at,
        current_loss=job.current_loss,
        error=job.error,
    )


@router.post("/finetune/evaluate")
async def finetune_evaluate(
    body: FineTuneEvalRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    ft = _get_finetune_components(request)
    if body.baseline_path:
        result = await ft["evaluator"].compare_models(body.baseline_path, body.model_path)
    else:
        result = await ft["evaluator"].evaluate_model(body.model_path)
        result = {
            "model": result.model_name,
            "avg_score": result.avg_score,
            "scores": result.scores,
        }
    return result


@router.post("/finetune/convert")
async def finetune_convert(
    body: FineTuneConvertRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    ft = _get_finetune_components(request)
    result = await ft["converter"].convert(
        merged_model_dir=body.merged_model_dir,
        output_name=body.output_name,
        quant_type=body.quant_type,
    )
    return result


@router.get("/finetune/models")
async def finetune_list_models(
    request: Request,
    user: dict = Depends(get_current_user),
):
    ft = _get_finetune_components(request)
    return await ft["converter"].list_converted()


@router.post("/finetune/deploy")
async def finetune_deploy(
    body: FineTuneDeployRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    ft = _get_finetune_components(request)
    result = await ft["deployer"].deploy(
        gguf_path=body.gguf_path,
        target_role=body.target_role,
        backup_previous=body.backup_previous,
    )
    return result


@router.post("/finetune/rollback/{role}")
async def finetune_rollback(
    role: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    ft = _get_finetune_components(request)
    return await ft["deployer"].rollback(role)


@router.get("/finetune/deployments")
async def finetune_deployments(
    request: Request,
    user: dict = Depends(get_current_user),
):
    ft = _get_finetune_components(request)
    return await ft["deployer"].list_deployments()


# ── Autonomous System Endpoints ──────────────────────────────────────


@router.get("/autonomous/status")
async def autonomous_status(request: Request, user: dict = Depends(get_current_user)):
    """Full status of all autonomous subsystems."""
    healer = getattr(request.app.state, "self_healer", None)
    evolution = getattr(request.app.state, "evolution_engine", None)
    curiosity = getattr(request.app.state, "curiosity_engine", None)
    job_runner = getattr(request.app.state, "job_runner", None)

    return {
        "health": healer.get_health() if healer else {},
        "evolution": evolution.get_metrics() if evolution else {},
        "curiosity": curiosity.get_stats() if curiosity else {},
        "jobs": job_runner.get_status() if job_runner else {},
    }


@router.post("/autonomous/goal")
async def autonomous_goal(
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Submit an autonomous multi-step goal to AutoPilot."""
    body = await request.json()
    goal = body.get("goal", "")
    if not goal:
        raise HTTPException(400, "goal is required")

    orch = request.app.state.orchestrator
    if not orch or not orch._autopilot:
        raise HTTPException(503, "AutoPilot not initialized")

    import asyncio
    task = asyncio.ensure_future(
        orch._autopilot.execute_goal(
            goal=goal,
            session_id=body.get("session_id", ""),
        )
    )

    return {"status": "accepted", "goal": goal, "message": "Goal submitted to AutoPilot"}


@router.post("/autonomous/heal")
async def autonomous_heal(
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Trigger a manual healing check."""
    healer = getattr(request.app.state, "self_healer", None)
    if not healer:
        raise HTTPException(503, "SelfHealer not initialized")

    await healer._check_all()
    return {"status": "ok", "health": healer.get_health()}


@router.post("/autonomous/curiosity/daily")
async def trigger_curiosity_daily(
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Manually trigger a daily curiosity cycle."""
    curiosity = getattr(request.app.state, "curiosity_engine", None)
    if not curiosity:
        raise HTTPException(503, "CuriosityEngine not initialized")

    import asyncio
    asyncio.ensure_future(curiosity.run_daily_cycle())
    return {"status": "accepted", "message": "Daily curiosity cycle triggered"}
