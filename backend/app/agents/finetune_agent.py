from __future__ import annotations

import json
from typing import Any, Callable, Coroutine

from app.config import get_settings
from app.core.finetune import (
    DataCollector,
    DataFormatter,
    FineTuneTrainer,
    GGUFConverter,
    ModelDeployer,
    ModelEvaluator,
)
from app.core.llm_engine import LLMEngine
from app.utils.helpers import Timer
from app.utils.logger import get_logger

from .base_agent import AgentResult, BaseAgent
from .voice_context import voice_prepend_after_first_system

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Miya's fine-tuning specialist. You manage the complete lifecycle "
    "of model training: data collection, formatting, training with Unsloth/QLoRA, "
    "evaluation, GGUF conversion, and deployment.\n\n"
    "Capabilities:\n"
    "- Collect training data from chat history, feedback, and custom datasets\n"
    "- Format data into ChatML for SFT training\n"
    "- Run QLoRA fine-tuning with Unsloth on local GPU\n"
    "- Evaluate fine-tuned vs baseline models\n"
    "- Convert to GGUF and deploy to production\n"
    "- Rollback if a new model underperforms\n\n"
    "Available tools:\n"
    "- finetune_collect: Gather training data. "
    'Args: {"action": "history|feedback|dataset|all", "path": "...", "min_quality": float}\n'
    "- finetune_format: Format collected data for training. "
    'Args: {"action": "format|split", "input_path": "...", "include_thinking": bool}\n'
    "- finetune_train: Start/monitor training. "
    'Args: {"action": "start|status|list", "base_model": "...", "dataset_path": "...", '
    '"output_name": "...", "epochs": int, "lora_rank": int}\n'
    "- finetune_eval: Evaluate a model. "
    'Args: {"action": "evaluate|compare", "model_path": "...", "baseline_path": "..."}\n'
    "- finetune_convert: Convert to GGUF. "
    'Args: {"action": "convert|list", "merged_dir": "...", "quant_type": "q4_k_m"}\n'
    "- finetune_deploy: Deploy or rollback. "
    'Args: {"action": "deploy|rollback|history", "gguf_path": "...", "role": "orchestrator"}\n\n'
    "To use a tool, output a JSON block:\n"
    '```json\n{"tool": "<name>", "args": {…}}\n```\n\n'
    "Guidelines:\n"
    "- Always collect & format data before training\n"
    "- Recommend QLoRA for 7B/14B models on RTX 5060\n"
    "- Always evaluate before deploying\n"
    "- Keep backups before deployment\n"
)

FINETUNE_TOOLS = [
    "finetune_collect",
    "finetune_format",
    "finetune_train",
    "finetune_eval",
    "finetune_convert",
    "finetune_deploy",
]

HF_BASE_MODELS = {
    "chat": "mistralai/Mistral-7B-Instruct-v0.3",
    "code": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "orchestrator": "Qwen/Qwen2.5-14B-Instruct",
    "creative": "ronniealfaro/mythos-prime",
}


class FineTuneAgent(BaseAgent):
    """Manages the full model fine-tuning lifecycle."""

    def __init__(self, llm_engine: LLMEngine) -> None:
        settings = get_settings()
        super().__init__(
            name="finetune",
            model_path=settings.orchestrator_model,
            system_prompt=SYSTEM_PROMPT,
            available_tools=FINETUNE_TOOLS,
        )
        self._engine = llm_engine
        self._model_name = settings.orchestrator_model
        self._collector = DataCollector()
        self._formatter = DataFormatter()
        self._trainer = FineTuneTrainer()
        self._evaluator = ModelEvaluator()
        self._converter = GGUFConverter()
        self._deployer = ModelDeployer()

    def ensure_loaded(self) -> None:
        settings = get_settings()
        try:
            self._engine.get_model(self._model_name)
        except KeyError:
            self._engine.swap_model(self._model_name, n_ctx=settings.orchestrator_ctx)

    async def execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        tool_executor: Callable[..., Coroutine[Any, Any, Any]] | None = None,
    ) -> AgentResult:
        logger.info("[FineTuneAgent] processing query (len=%d)", len(query))
        self.ensure_loaded()
        context = context or {}
        all_tool_calls: list[dict[str, Any]] = []

        with Timer() as t:
            messages = self._build_messages(query, context)

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.5,
                )
            except Exception as exc:
                logger.exception("[FineTuneAgent] generation failed")
                return AgentResult(
                    success=False,
                    output="",
                    execution_time_ms=t.elapsed_ms,
                    error=str(exc),
                )

            output = (
                response.get("text", "") if isinstance(response, dict) else str(response)
            )

            tool_calls = self._parse_tool_calls(output)
            if tool_calls:
                output, extra = await self._tool_loop(
                    messages, output, tool_calls, max_rounds=5
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

    async def _execute_finetune_tool(
        self, name: str, args: dict[str, Any]
    ) -> str:
        """Route tool calls to the appropriate pipeline component."""
        try:
            if name == "finetune_collect":
                return await self._handle_collect(args)
            elif name == "finetune_format":
                return await self._handle_format(args)
            elif name == "finetune_train":
                return await self._handle_train(args)
            elif name == "finetune_eval":
                return await self._handle_eval(args)
            elif name == "finetune_convert":
                return await self._handle_convert(args)
            elif name == "finetune_deploy":
                return await self._handle_deploy(args)
            else:
                return f"Unknown tool: {name}"
        except Exception as exc:
            logger.warning("[FineTuneAgent] tool %s failed: %s", name, exc)
            return f"Error: {exc}"

    async def _handle_collect(self, args: dict[str, Any]) -> str:
        action = args.get("action", "all")
        if action == "history":
            data = await self._collector.collect_from_history(
                min_quality=args.get("min_quality", 0.7)
            )
        elif action == "feedback":
            data = await self._collector.collect_from_feedback()
        elif action == "dataset":
            data = await self._collector.collect_from_dataset(args["path"])
        else:
            data = await self._collector.collect_all(
                custom_datasets=args.get("datasets"),
                min_quality=args.get("min_quality", 0.7),
            )
        return f"Collected {len(data)} samples"

    async def _handle_format(self, args: dict[str, Any]) -> str:
        action = args.get("action", "format")
        if action == "format":
            import json as _json
            from pathlib import Path

            path = args.get("input_path", "")
            samples = []
            if path:
                p = Path(path)
                if p.exists():
                    with open(p) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                samples.append(_json.loads(line))

            formatted = self._formatter.format_dataset(
                samples,
                include_thinking=args.get("include_thinking", True),
            )
            out = self._formatter.save_formatted(formatted)
            return f"Formatted {len(formatted)} samples -> {out}"
        elif action == "split":
            return "Split not yet run — format first, then call split"
        return "Unknown format action"

    async def _handle_train(self, args: dict[str, Any]) -> str:
        action = args.get("action", "start")
        if action == "start":
            base = args.get("base_model", HF_BASE_MODELS.get("chat", ""))
            dataset = args.get("dataset_path", "")
            if not dataset:
                return "Error: dataset_path required"
            job = await self._trainer.start_training(
                base_model_hf=base,
                dataset_path=dataset,
                output_name=args.get("output_name", "miya-finetuned"),
                epochs=args.get("epochs"),
                lora_rank=args.get("lora_rank"),
            )
            return f"Training started: job_id={job.job_id}, model={base}"
        elif action == "status":
            job = self._trainer.get_job(args.get("job_id", ""))
            if not job:
                return "Job not found"
            return json.dumps({
                "job_id": job.job_id,
                "status": job.status,
                "loss": job.current_loss,
                "epoch": f"{job.current_epoch}/{job.total_epochs}",
                "error": job.error,
            })
        elif action == "list":
            return json.dumps(self._trainer.list_jobs(), indent=2)
        return "Unknown train action"

    async def _handle_eval(self, args: dict[str, Any]) -> str:
        action = args.get("action", "evaluate")
        if action == "evaluate":
            result = await self._evaluator.evaluate_model(args.get("model_path", ""))
            return json.dumps({
                "model": result.model_name,
                "avg_score": result.avg_score,
                "scores": result.scores,
            }, indent=2)
        elif action == "compare":
            comp = await self._evaluator.compare_models(
                args.get("baseline_path", ""),
                args.get("model_path", ""),
            )
            return json.dumps(comp, indent=2)
        return "Unknown eval action"

    async def _handle_convert(self, args: dict[str, Any]) -> str:
        action = args.get("action", "convert")
        if action == "convert":
            result = await self._converter.convert(
                merged_model_dir=args.get("merged_dir", ""),
                output_name=args.get("output_name", "miya-finetuned"),
                quant_type=args.get("quant_type", "q4_k_m"),
            )
            return json.dumps(result, indent=2)
        elif action == "list":
            models = await self._converter.list_converted()
            return json.dumps(models, indent=2)
        return "Unknown convert action"

    async def _handle_deploy(self, args: dict[str, Any]) -> str:
        action = args.get("action", "deploy")
        if action == "deploy":
            result = await self._deployer.deploy(
                gguf_path=args.get("gguf_path", ""),
                target_role=args.get("role", "orchestrator"),
            )
            return json.dumps(result, indent=2)
        elif action == "rollback":
            result = await self._deployer.rollback(args.get("role", "orchestrator"))
            return json.dumps(result, indent=2)
        elif action == "history":
            history = await self._deployer.list_deployments()
            return json.dumps(history, indent=2)
        return "Unknown deploy action"

    async def _tool_loop(
        self,
        messages: list[dict[str, str]],
        initial_output: str,
        tool_calls: list[dict[str, Any]],
        max_rounds: int = 5,
    ) -> tuple[str, list[dict[str, Any]]]:
        all_calls: list[dict[str, Any]] = []
        output = initial_output

        for _ in range(max_rounds):
            if not tool_calls:
                break

            results: list[str] = []
            for call in tool_calls:
                name = call.get("tool") or call.get("name", "")
                args = call.get("args") or call.get("arguments", {})
                all_calls.append({"tool": name, "args": args})
                result = await self._execute_finetune_tool(name, args)
                results.append(f"[{name}] {result}")

            messages.append({"role": "assistant", "content": output})
            messages.append({
                "role": "user",
                "content": "Tool results:\n" + "\n\n".join(results),
            })

            try:
                response = await self._engine.chat(
                    model_filename=self._model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.5,
                )
                output = (
                    response.get("text", "")
                    if isinstance(response, dict)
                    else str(response)
                )
                tool_calls = self._parse_tool_calls(output)
            except Exception:
                logger.exception("[FineTuneAgent] follow-up generation failed")
                break

        return output, all_calls

    def _build_messages(
        self, query: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        if self._trainer.is_training:
            messages.append({
                "role": "system",
                "content": "NOTE: A training job is currently running.",
            })

        history = context.get("history", [])
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        voice_prepend_after_first_system(messages, context)
        return messages
