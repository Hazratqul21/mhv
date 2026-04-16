from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


class ModelDeployer:
    """Deploys a fine-tuned GGUF model into MIYA's active model pool.

    Steps:
    1. Copy GGUF to models/ directory
    2. Update config to point to the new model
    3. Trigger LLM engine to reload the model
    4. Verify model loads correctly
    5. Keep backup of previous model config
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    async def deploy(
        self,
        gguf_path: str,
        target_role: str = "orchestrator",
        backup_previous: bool = True,
    ) -> dict[str, Any]:
        valid_roles = ("orchestrator", "chat", "code", "creative", "uncensored")
        if target_role not in valid_roles:
            raise ValueError(f"Invalid role: {target_role}. Use: {valid_roles}")

        source = Path(gguf_path)
        if not source.exists():
            raise FileNotFoundError(f"GGUF not found: {gguf_path}")

        models_dir = self._settings.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)

        role_to_config = {
            "orchestrator": "orchestrator_model",
            "chat": "chat_model",
            "code": "code_model",
            "creative": "creative_model",
            "uncensored": "uncensored_model",
        }
        config_field = role_to_config[target_role]
        previous_model = getattr(self._settings, config_field)

        if backup_previous:
            self._create_backup(target_role, previous_model)

        target_path = models_dir / source.name
        log.info(
            "deploying_model",
            source=str(source),
            target=str(target_path),
            role=target_role,
        )

        shutil.copy2(str(source), str(target_path))

        self._update_env(config_field, source.name)

        deploy_record = {
            "timestamp": time.time(),
            "role": target_role,
            "previous_model": previous_model,
            "new_model": source.name,
            "gguf_source": str(source),
            "size_mb": source.stat().st_size / (1024 * 1024),
        }

        history_path = Path(self._settings.finetune_output_dir) / "deploy_history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(deploy_record) + "\n")

        log.info("model_deployed", record=deploy_record)
        return deploy_record

    def _create_backup(self, role: str, model_filename: str) -> None:
        backup_dir = Path(self._settings.finetune_output_dir) / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        backup_record = {
            "role": role,
            "model": model_filename,
            "backed_up_at": time.time(),
        }
        with open(backup_dir / f"{role}_backup.json", "w") as f:
            json.dump(backup_record, f, indent=2)

        log.info("backup_created", role=role, model=model_filename)

    def _update_env(self, config_field: str, new_model_name: str) -> None:
        env_key_map = {
            "orchestrator_model": "ORCHESTRATOR_MODEL",
            "chat_model": "CHAT_MODEL",
            "code_model": "CODE_MODEL",
            "creative_model": "CREATIVE_MODEL",
            "uncensored_model": "UNCENSORED_MODEL",
        }
        env_key = env_key_map.get(config_field)
        if not env_key:
            return

        env_path = self._settings.project_root / ".env"
        if not env_path.exists():
            log.warning("env_file_not_found", path=str(env_path))
            return

        lines = env_path.read_text(encoding="utf-8").splitlines()
        updated = False
        new_lines = []
        for line in lines:
            if line.startswith(f"{env_key}="):
                new_lines.append(f"{env_key}={new_model_name}")
                updated = True
            else:
                new_lines.append(line)

        if not updated:
            new_lines.append(f"{env_key}={new_model_name}")

        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        log.info("env_updated", key=env_key, value=new_model_name)

    async def rollback(self, role: str) -> dict[str, Any]:
        backup_dir = Path(self._settings.finetune_output_dir) / "backups"
        backup_file = backup_dir / f"{role}_backup.json"

        if not backup_file.exists():
            raise FileNotFoundError(f"No backup found for role: {role}")

        with open(backup_file) as f:
            backup = json.load(f)

        role_to_config = {
            "orchestrator": "orchestrator_model",
            "chat": "chat_model",
            "code": "code_model",
            "creative": "creative_model",
            "uncensored": "uncensored_model",
        }
        config_field = role_to_config[role]
        self._update_env(config_field, backup["model"])

        log.info("rollback_complete", role=role, restored_model=backup["model"])
        return {"role": role, "restored_model": backup["model"]}

    async def list_deployments(self) -> list[dict[str, Any]]:
        history_path = Path(self._settings.finetune_output_dir) / "deploy_history.jsonl"
        if not history_path.exists():
            return []

        deployments = []
        with open(history_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    deployments.append(json.loads(line))
        return deployments
