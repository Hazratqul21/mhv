from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Optional

import warnings

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── General ──────────────────────────────────────────────────────────
    env: str = Field("development", alias="MIYA_ENV")
    debug: bool = Field(False, alias="MIYA_DEBUG")
    log_level: str = Field("INFO", alias="MIYA_LOG_LEVEL")
    secret_key: str = Field("change-me", alias="MIYA_SECRET_KEY")
    base_dir: Path = Path(__file__).resolve().parent.parent

    # ── API ──────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    api_workers: int = Field(1, alias="API_WORKERS")
    cors_origins: str = Field(
        "http://localhost:7860,http://localhost:3000", alias="CORS_ORIGINS"
    )

    # ── GPU ──────────────────────────────────────────────────────────────
    cuda_devices: str = Field("0", alias="CUDA_VISIBLE_DEVICES")
    gpu_layers: int = Field(35, alias="GPU_LAYERS")

    # ── Model Paths ──────────────────────────────────────────────────────
    models_dir: Optional[Path] = Field(default=None)
    orchestrator_model: str = Field(
        "qwen3.5-27b-opus-q4_k_m.gguf", alias="ORCHESTRATOR_MODEL"
    )
    chat_model: str = Field(
        "mistral-7b-instruct-v0.3-q4_k_m.gguf", alias="CHAT_MODEL"
    )
    code_model: str = Field(
        "qwen2.5-coder-14b-instruct-q4_k_m.gguf", alias="CODE_MODEL"
    )
    vision_model: str = Field(
        "llava-v1.6-mistral-7b-q4_k_m.gguf", alias="VISION_MODEL"
    )
    embedding_model: str = Field(
        "nomic-embed-text-v1.5.Q6_K.gguf", alias="EMBEDDING_MODEL"
    )
    creative_model: str = Field(
        "mythos-prime-q4_k_m.gguf", alias="CREATIVE_MODEL"
    )
    uncensored_model: str = Field(
        "qwen3.5-27b-opus-abliterated-q4_k_m.gguf", alias="UNCENSORED_MODEL"
    )
    math_model: str = Field(
        "qwen3.5-27b-opus-abliterated-q4_k_m.gguf", alias="MATH_MODEL"
    )

    # ── Context Sizes ────────────────────────────────────────────────────
    orchestrator_ctx: int = Field(16384, alias="ORCHESTRATOR_CTX")
    chat_ctx: int = Field(4096, alias="CHAT_CTX")
    code_ctx: int = Field(8192, alias="CODE_CTX")
    vision_ctx: int = Field(4096, alias="VISION_CTX")
    creative_ctx: int = Field(4096, alias="CREATIVE_CTX")
    math_ctx: int = Field(8192, alias="MATH_CTX")

    # ── ChromaDB ─────────────────────────────────────────────────────────
    chroma_host: str = Field("chroma", alias="CHROMA_HOST")
    chroma_port: int = Field(8000, alias="CHROMA_PORT")

    # ── Redis ────────────────────────────────────────────────────────────
    redis_host: str = Field("redis", alias="REDIS_HOST")
    redis_port: int = Field(6379, alias="REDIS_PORT")
    redis_db: int = Field(0, alias="REDIS_DB")

    # ── MinIO ────────────────────────────────────────────────────────────
    minio_host: str = Field("minio", alias="MINIO_HOST")
    minio_port: int = Field(9000, alias="MINIO_PORT")
    minio_root_user: str = Field("miya", alias="MINIO_ROOT_USER")
    minio_root_password: str = Field("changeme123", alias="MINIO_ROOT_PASSWORD")
    minio_bucket: str = Field("miya-storage", alias="MINIO_BUCKET")

    # ── External Services ────────────────────────────────────────────────
    searxng_host: str = Field("http://searxng:8080", alias="SEARXNG_HOST")
    comfyui_host: str = Field("http://comfyui:8188", alias="COMFYUI_HOST")
    cogvideo_host: str = Field("http://localhost:8190", alias="COGVIDEO_HOST")
    musicgen_host: str = Field("http://localhost:8191", alias="MUSICGEN_HOST")
    bark_host: str = Field("http://localhost:8194", alias="BARK_HOST")
    rvc_host: str = Field("http://localhost:8192", alias="RVC_HOST")
    triposr_host: str = Field("http://localhost:8193", alias="TRIPOSR_HOST")

    # ── Cloud AI (optional, set keys to enable) ─────────────────────────
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    google_api_key: str = Field("", alias="GOOGLE_API_KEY")

    # ── Sandbox ──────────────────────────────────────────────────────────
    sandbox_timeout: int = Field(30, alias="SANDBOX_TIMEOUT")
    sandbox_memory_limit: str = Field("512m", alias="SANDBOX_MEMORY_LIMIT")
    sandbox_network: bool = Field(False, alias="SANDBOX_NETWORK")

    # ── JWT ──────────────────────────────────────────────────────────────
    jwt_secret: str = Field("change-me", alias="JWT_SECRET")
    jwt_algorithm: str = Field("HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(1440, alias="JWT_EXPIRE_MINUTES")

    # ── Rate Limiting ────────────────────────────────────────────────────
    rate_limit_per_minute: int = Field(60, alias="RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(10, alias="RATE_LIMIT_BURST")

    # ── Data Directory ─────────────────────────────────────────────────
    data_dir_override: str = Field("", alias="MIYA_DATA_DIR")

    # ── Fine-Tuning ─────────────────────────────────────────────────────
    finetune_lora_rank: int = Field(16, alias="FINETUNE_LORA_RANK")
    finetune_lora_alpha: int = Field(32, alias="FINETUNE_LORA_ALPHA")
    finetune_batch_size: int = Field(2, alias="FINETUNE_BATCH_SIZE")
    finetune_epochs: int = Field(3, alias="FINETUNE_EPOCHS")
    finetune_lr: float = Field(2e-4, alias="FINETUNE_LR")
    finetune_output_dir: str = Field("data/finetune", alias="FINETUNE_OUTPUT_DIR")

    # ── Whisper / TTS ────────────────────────────────────────────────────
    whisper_model: str = Field("large-v3", alias="WHISPER_MODEL")
    whisper_device: str = Field("cuda", alias="WHISPER_DEVICE")
    tts_engine: str = Field("kokoro", alias="TTS_ENGINE")
    tts_voice: str = Field("af_heart", alias="TTS_VOICE")

    class Config:
        env_file = str(Path(__file__).resolve().parent.parent.parent / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"

    @model_validator(mode="after")
    def warn_default_secrets_in_production(self) -> "Settings":
        if (self.env or "").lower() not in ("development", "dev", "test"):
            if self.secret_key == "change-me" or self.jwt_secret == "change-me":
                warnings.warn(
                    "MIYA_SECRET_KEY and/or JWT_SECRET are still default 'change-me' "
                    f"while MIYA_ENV={self.env!r}. Set strong secrets before exposing the API.",
                    UserWarning,
                    stacklevel=2,
                )
        return self

    def model_post_init(self, __context) -> None:
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"

    @property
    def project_root(self) -> Path:
        return self.base_dir.parent

    @property
    def data_dir(self) -> Path:
        if self.data_dir_override:
            return Path(self.data_dir_override)
        return self.base_dir.parent / "data"

    @property
    def output_dir(self) -> Path:
        """Generated media (video, audio, voice, 3D) lives under data/outputs."""
        return self.data_dir / "outputs"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def chroma_url(self) -> str:
        return f"http://{self.chroma_host}:{self.chroma_port}"

    @property
    def minio_endpoint(self) -> str:
        return f"{self.minio_host}:{self.minio_port}"

    def model_path(self, model_filename: str) -> Path:
        return self.models_dir / model_filename

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
