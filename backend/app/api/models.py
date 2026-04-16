from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32_000)
    session_id: Optional[str] = None
    context: Optional[dict[str, Any]] = None
    stream: bool = False
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reply_language: Optional[str] = Field(
        default=None,
        description="Force assistant reply language, e.g. English, uz, ru (voice clients).",
    )


class ChatResponse(BaseModel):
    response: str
    agent_used: str
    tools_used: list[str] = Field(default_factory=list)
    session_id: str
    execution_time_ms: int = 0
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    uptime_seconds: float = 0.0
    services: dict[str, str] = Field(default_factory=dict)


class AgentInfo(BaseModel):
    name: str
    model: str
    status: str = "ready"
    description: str = ""


class ToolInfo(BaseModel):
    name: str
    category: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class SessionInfo(BaseModel):
    session_id: str
    message_count: int = 0
    created_at: Optional[float] = None
    last_active: Optional[float] = None


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
    request_id: Optional[str] = None


class UploadResponse(BaseModel):
    filename: str
    size_bytes: int
    content_type: str
    path: str


# ── Fine-Tuning ─────────────────────────────────────────────────────────

class FineTuneStartRequest(BaseModel):
    base_model: str = Field("mistralai/Mistral-7B-Instruct-v0.3", description="HuggingFace model ID")
    dataset_path: str = Field(..., description="Path to JSONL training data")
    output_name: str = "miya-finetuned"
    lora_rank: int = 16
    lora_alpha: int = 32
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-4
    max_seq_length: int = 4096
    load_in_4bit: bool = True


class FineTuneJobResponse(BaseModel):
    job_id: str
    status: str
    base_model: str
    started_at: float = 0.0
    completed_at: float = 0.0
    current_loss: float = 0.0
    error: str = ""


class FineTuneConvertRequest(BaseModel):
    merged_model_dir: str
    output_name: str = "miya-finetuned"
    quant_type: str = "q4_k_m"


class FineTuneDeployRequest(BaseModel):
    gguf_path: str
    target_role: str = "orchestrator"
    backup_previous: bool = True


class FineTuneEvalRequest(BaseModel):
    model_path: str
    baseline_path: str = ""


class FineTuneCollectRequest(BaseModel):
    min_quality: float = 0.7
    custom_datasets: list[str] = Field(default_factory=list)
