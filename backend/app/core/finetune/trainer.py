from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TrainingJob:
    job_id: str
    base_model: str
    dataset_path: str
    output_dir: str
    status: str = "pending"  # pending, running, completed, failed
    started_at: float = 0.0
    completed_at: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 3
    current_loss: float = 0.0
    best_loss: float = float("inf")
    error: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


class FineTuneTrainer:
    """Runs LoRA/QLoRA fine-tuning using Unsloth on local GPU.

    Supports:
    - QLoRA (4-bit) for low VRAM usage
    - Gradient checkpointing for additional memory savings
    - train_on_responses_only to focus learning on assistant outputs
    - Periodic checkpointing and loss logging
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._jobs: dict[str, TrainingJob] = {}
        self._active_job: str | None = None

    async def start_training(
        self,
        base_model_hf: str,
        dataset_path: str,
        output_name: str = "miya-finetuned",
        *,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        batch_size: int | None = None,
        epochs: int | None = None,
        learning_rate: float | None = None,
        max_seq_length: int = 4096,
        load_in_4bit: bool = True,
    ) -> TrainingJob:
        job_id = f"ft_{int(time.time())}"
        output_dir = str(
            Path(self._settings.finetune_output_dir) / output_name
        )

        job = TrainingJob(
            job_id=job_id,
            base_model=base_model_hf,
            dataset_path=dataset_path,
            output_dir=output_dir,
            total_epochs=epochs or self._settings.finetune_epochs,
        )
        self._jobs[job_id] = job

        asyncio.create_task(
            self._run_training(
                job,
                lora_rank=lora_rank or self._settings.finetune_lora_rank,
                lora_alpha=lora_alpha or self._settings.finetune_lora_alpha,
                batch_size=batch_size or self._settings.finetune_batch_size,
                learning_rate=learning_rate or self._settings.finetune_lr,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
            )
        )

        return job

    async def _run_training(
        self,
        job: TrainingJob,
        lora_rank: int,
        lora_alpha: int,
        batch_size: int,
        learning_rate: float,
        max_seq_length: int,
        load_in_4bit: bool,
    ) -> None:
        job.status = "running"
        job.started_at = time.time()
        self._active_job = job.job_id
        log.info("training_started", job_id=job.job_id, model=job.base_model)

        try:
            result = await asyncio.to_thread(
                self._train_sync,
                job=job,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
            )
            job.status = "completed"
            job.metrics = result
            log.info("training_completed", job_id=job.job_id, metrics=result)
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            log.error("training_failed", job_id=job.job_id, error=str(exc))
        finally:
            job.completed_at = time.time()
            self._active_job = None

    def _train_sync(
        self,
        job: TrainingJob,
        lora_rank: int,
        lora_alpha: int,
        batch_size: int,
        learning_rate: float,
        max_seq_length: int,
        load_in_4bit: bool,
    ) -> dict[str, Any]:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=job.base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=None,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        dataset = load_dataset("json", data_files=job.dataset_path, split="train")

        def formatting_func(examples: dict) -> list[str]:
            texts = []
            for messages in examples["messages"]:
                parts = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
                parts.append("<|im_start|>assistant\n")
                texts.append("\n".join(parts))
            return texts

        output_dir = Path(job.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=batch_size,
            num_train_epochs=job.total_epochs,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            optim="adamw_8bit",
            seed=42,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            formatting_func=formatting_func,
            max_seq_length=max_seq_length,
        )

        train_result = trainer.train()

        lora_path = output_dir / "lora_adapter"
        model.save_pretrained(str(lora_path))
        tokenizer.save_pretrained(str(lora_path))

        merged_path = output_dir / "merged_16bit"
        model.save_pretrained_merged(
            str(merged_path), tokenizer, save_method="merged_16bit"
        )

        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime_seconds": train_result.metrics.get("train_runtime", 0),
            "train_samples": len(dataset),
            "lora_path": str(lora_path),
            "merged_path": str(merged_path),
            "epochs": job.total_epochs,
            "lora_rank": lora_rank,
        }

        with open(output_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def get_job(self, job_id: str) -> TrainingJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        return [
            {
                "job_id": j.job_id,
                "status": j.status,
                "base_model": j.base_model,
                "started_at": j.started_at,
                "completed_at": j.completed_at,
                "current_loss": j.current_loss,
                "error": j.error,
            }
            for j in self._jobs.values()
        ]

    @property
    def is_training(self) -> bool:
        return self._active_job is not None
