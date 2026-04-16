#!/usr/bin/env python3
"""MIYA AI Watchdog — Autonomous error detection, diagnosis, and repair.

Uses Anthropic Claude (primary) or OpenAI GPT-4o (fallback) to analyze
errors in the MIYA project and generate fixes automatically.

Usage (standalone):
    python3 watchdog_ai.py --check          # Run one health check cycle
    python3 watchdog_ai.py --monitor        # Continuous monitoring loop
    python3 watchdog_ai.py --diagnose FILE  # Diagnose a specific log file

Called by watchdog.sh after server startup for continuous monitoring.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_DIR / "backend"
FRONTEND_DIR = PROJECT_DIR / "frontend"
LOGS_DIR = PROJECT_DIR / "logs"
MODELS_DIR = BACKEND_DIR / "models"
VENV_PYTHON = str(PROJECT_DIR / ".venv" / "bin" / "python3")

LOG_FILE = LOGS_DIR / "watchdog_ai.log"
BACKEND_LOG = LOGS_DIR / "backend.log"
FRONTEND_LOG = LOGS_DIR / "frontend.log"

BLOCKED_COMMANDS = [
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=", ":(){:|:&};:",
    "chmod -R 777 /", "shutdown", "reboot", "init 0", "init 6",
    "mv /* ", "curl | bash", "wget | bash", "pip install --upgrade pip",
    "> /dev/sda", "mv /usr", "rm -rf /home", "rm -rf /etc",
]

ALLOWED_PATH_PREFIX = str(PROJECT_DIR)

MAX_RETRIES_PER_ERROR = 3
MAX_LOG_CONTEXT = 80
MAX_FILE_READ = 200

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str, level: str = "INFO") -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Diagnosis:
    error_type: str = ""
    description: str = ""
    root_cause: str = ""
    fix_actions: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0

@dataclass
class FixResult:
    success: bool = False
    action: str = ""
    output: str = ""
    error: str = ""

# ---------------------------------------------------------------------------
# AI Client — Anthropic primary, OpenAI fallback
# ---------------------------------------------------------------------------

class AIClient:
    """Thin wrapper over Anthropic / OpenAI HTTP APIs using only stdlib."""

    def __init__(self) -> None:
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        if not self.anthropic_key and not self.openai_key:
            raise RuntimeError(
                "No AI API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env"
            )

    def ask(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Send a prompt and return the text response."""
        if self.anthropic_key:
            try:
                return self._call_anthropic(system, user, max_tokens)
            except Exception as exc:
                log(f"Anthropic failed: {exc}, trying OpenAI fallback", "WARN")

        if self.openai_key:
            return self._call_openai(system, user, max_tokens)

        raise RuntimeError("All AI providers failed")

    def _call_anthropic(self, system: str, user: str, max_tokens: int) -> str:
        import urllib.request
        import urllib.error

        body = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.anthropic_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        return data["content"][0]["text"]

    def _call_openai(self, system: str, user: str, max_tokens: int) -> str:
        import urllib.request

        body = json.dumps({
            "model": "gpt-4o",
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        return data["choices"][0]["message"]["content"]

# ---------------------------------------------------------------------------
# Safety guard
# ---------------------------------------------------------------------------

AUDIT_LOG = LOGS_DIR / "watchdog_audit.log"

def audit(action: str, detail: str, result: str) -> None:
    """Immutable audit trail — every AI action is recorded."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = json.dumps({
        "timestamp": ts, "action": action,
        "detail": detail[:500], "result": result,
    }, ensure_ascii=False)
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


class SafetyGuard:
    """Validates every action before execution."""

    EDITABLE_SUFFIXES = {
        ".py", ".sh", ".txt", ".md", ".json", ".yaml", ".yml",
        ".cfg", ".ini", ".toml", ".html", ".css", ".js",
    }

    PROTECTED_FILES = {
        ".env", "credentials.json", "secrets.yaml", "id_rsa",
        "id_ed25519", ".gitignore",
    }

    @staticmethod
    def is_safe_command(cmd: str) -> tuple[bool, str]:
        for blocked in BLOCKED_COMMANDS:
            if blocked in cmd:
                return False, f"Blocked pattern: {blocked}"

        if re.search(r"curl\s.*\|\s*(bash|sh)", cmd):
            return False, "Blocked: piping download to shell"
        if re.search(r"wget\s.*\|\s*(bash|sh)", cmd):
            return False, "Blocked: piping download to shell"

        return True, ""

    @staticmethod
    def is_safe_path(path: str) -> tuple[bool, str]:
        p = Path(path)
        try:
            resolved = str(p.resolve(strict=False))
        except (OSError, ValueError):
            return False, f"Cannot resolve path: {path}"

        if not resolved.startswith(ALLOWED_PATH_PREFIX):
            return False, f"Path outside project: {resolved}"

        if p.is_symlink():
            target = str(p.resolve())
            if not target.startswith(ALLOWED_PATH_PREFIX):
                return False, f"Symlink escapes project dir: {target}"

        return True, ""

    @staticmethod
    def is_safe_file_edit(path: str, content: str) -> tuple[bool, str]:
        safe, reason = SafetyGuard.is_safe_path(path)
        if not safe:
            return False, reason

        fname = Path(path).name
        if fname in SafetyGuard.PROTECTED_FILES:
            return False, f"Protected file cannot be edited: {fname}"

        if Path(path).suffix not in SafetyGuard.EDITABLE_SUFFIXES:
            return False, f"Cannot edit file type: {Path(path).suffix}"

        return True, ""

    @staticmethod
    def is_safe_pip_install(package: str) -> tuple[bool, str]:
        if any(c in package for c in [";", "&", "|", "`", "$", "(", ")"]):
            return False, f"Suspicious characters in package name: {package}"
        if "/" in package and "://" not in package:
            return False, f"Local path installs not allowed: {package}"
        return True, ""

# ---------------------------------------------------------------------------
# Health Checker
# ---------------------------------------------------------------------------

class HealthChecker:
    """Runs diagnostics and collects error info."""

    def __init__(self) -> None:
        self.guard = SafetyGuard()

    def check_all(self) -> list[dict[str, Any]]:
        errors: list[dict[str, Any]] = []

        checks = [
            self._check_backend_health,
            self._check_frontend_health,
            self._check_import_health,
            self._check_backend_log_errors,
            self._check_frontend_log_errors,
            self._check_model_files,
        ]

        for check_fn in checks:
            try:
                result = check_fn()
                if result:
                    errors.append(result)
            except Exception as exc:
                log(f"Check {check_fn.__name__} crashed: {exc}", "WARN")

        return errors

    def _check_backend_health(self) -> dict | None:
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:8000/health", timeout=10) as r:
                data = json.loads(r.read())
                if data.get("status") != "ok":
                    return {"type": "backend_unhealthy", "details": json.dumps(data)}
        except Exception as exc:
            return {"type": "backend_down", "details": str(exc)}
        return None

    def _check_frontend_health(self) -> dict | None:
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:7860/", timeout=10):
                pass
        except Exception as exc:
            return {"type": "frontend_down", "details": str(exc)}
        return None

    def _check_import_health(self) -> dict | None:
        test_script = (
            "from app.core.llm_engine import LLMEngine; "
            "from app.core.orchestrator import MiyaOrchestrator; "
            "from app.tools.registry import ToolRegistry; "
            "from app.agents.chat_agent import ChatAgent; "
            "print('OK')"
        )
        try:
            result = subprocess.run(
                [VENV_PYTHON, "-c", test_script],
                capture_output=True, text=True, timeout=30,
                cwd=str(BACKEND_DIR),
            )
            if result.returncode != 0:
                return {
                    "type": "import_error",
                    "details": result.stderr[-2000:] if result.stderr else result.stdout[-2000:],
                }
        except subprocess.TimeoutExpired:
            return {"type": "import_timeout", "details": "Import test timed out after 30s"}
        except Exception as exc:
            return {"type": "import_check_failed", "details": str(exc)}
        return None

    def _check_backend_log_errors(self) -> dict | None:
        return self._scan_log(BACKEND_LOG, "backend_log_error")

    def _check_frontend_log_errors(self) -> dict | None:
        return self._scan_log(FRONTEND_LOG, "frontend_log_error")

    def _scan_log(self, log_path: Path, error_type: str) -> dict | None:
        if not log_path.exists():
            return None

        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return None

        recent = lines[-MAX_LOG_CONTEXT:]
        error_lines = [
            l for l in recent
            if any(kw in l.lower() for kw in ["error", "traceback", "exception", "failed", "critical"])
        ]

        if len(error_lines) >= 3:
            return {
                "type": error_type,
                "details": "\n".join(recent[-50:]),
            }
        return None

    def _check_model_files(self) -> dict | None:
        required = ["mistral-7b-instruct-v0.3-q4_k_m.gguf", "qwen3.5-27b-opus-q4_k_m.gguf"]
        for model in required:
            path = MODELS_DIR / model
            if path.exists():
                size = path.stat().st_size
                if size < 50_000_000:
                    return {"type": "model_corrupted", "details": f"{model} is only {size} bytes"}
            # Don't report missing — download may still be running
        return None

# ---------------------------------------------------------------------------
# AI Diagnoser
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are MIYA's autonomous watchdog AI. Your job is to diagnose and fix errors \
in the MIYA AI assistant project.

Project location: {project_dir}
Backend: {project_dir}/backend/app/
Frontend desktop: {project_dir}/frontend/desktop/
Frontend voice: {project_dir}/frontend/voice/
Models: {project_dir}/backend/models/
Venv python: {venv_python}

You MUST respond with ONLY a JSON object (no markdown, no explanation):
{{
  "error_type": "short_name",
  "description": "what went wrong",
  "root_cause": "why it happened",
  "confidence": 0.0-1.0,
  "fix_actions": [
    {{
      "type": "shell",
      "command": "the command to run",
      "cwd": "/working/directory"
    }},
    {{
      "type": "pip_install",
      "package": "package_name"
    }},
    {{
      "type": "file_edit",
      "path": "/absolute/path/to/file.py",
      "find": "exact text to find",
      "replace": "replacement text"
    }},
    {{
      "type": "restart_backend"
    }},
    {{
      "type": "restart_frontend"
    }}
  ]
}}

Rules:
- ONLY edit files inside {project_dir}/
- NEVER use rm -rf, dd, mkfs, shutdown, reboot
- NEVER modify .env files (API keys, secrets)
- NEVER delete model files (they take hours to download)
- For pip install, use the project venv: {venv_python} -m pip install ...
- Keep fixes minimal and targeted
- If unsure, set confidence < 0.3 and use fewer actions
""".format(project_dir=PROJECT_DIR, venv_python=VENV_PYTHON)


class AIDiagnoser:
    """Uses AI to diagnose errors and propose fixes."""

    def __init__(self, ai: AIClient) -> None:
        self._ai = ai
        self._history: list[str] = []

    def diagnose(self, error: dict[str, Any]) -> Diagnosis:
        context_parts = [
            f"Error type: {error['type']}",
            f"Details:\n{error['details'][:3000]}",
        ]

        if self._history:
            context_parts.append(
                f"Previous fix attempts (failed):\n" +
                "\n".join(f"  - {h}" for h in self._history[-5:])
            )

        relevant_files = self._gather_context(error)
        if relevant_files:
            context_parts.append("Relevant file contents:")
            for path, content in relevant_files.items():
                context_parts.append(f"\n--- {path} ---\n{content}")

        user_prompt = "\n\n".join(context_parts)

        try:
            raw = self._ai.ask(SYSTEM_PROMPT, user_prompt, max_tokens=2048)
        except Exception as exc:
            log(f"AI call failed: {exc}", "ERROR")
            return Diagnosis(error_type="ai_failed", description=str(exc))

        return self._parse_response(raw)

    def record_attempt(self, description: str) -> None:
        self._history.append(description)

    def _gather_context(self, error: dict) -> dict[str, str]:
        files: dict[str, str] = {}
        etype = error["type"]
        details = error.get("details", "")

        if etype == "import_error":
            file_matches = re.findall(r'File "([^"]+)"', details)
            for fpath in file_matches[:3]:
                if fpath.startswith(str(PROJECT_DIR)):
                    content = self._read_file(fpath)
                    if content:
                        files[fpath] = content

        if etype in ("backend_down", "backend_log_error"):
            for f in ["app/main.py", "app/config.py"]:
                full = BACKEND_DIR / f
                content = self._read_file(str(full))
                if content:
                    files[str(full)] = content

        if etype in ("frontend_down", "frontend_log_error"):
            for f in ["main.py", "chat_interface.py"]:
                full = FRONTEND_DIR / "desktop" / f
                content = self._read_file(str(full))
                if content:
                    files[str(full)] = content

        return files

    def _read_file(self, path: str) -> str | None:
        try:
            p = Path(path)
            if not p.exists() or not str(p.resolve()).startswith(str(PROJECT_DIR)):
                return None
            text = p.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            if len(lines) > MAX_FILE_READ:
                return "\n".join(lines[:MAX_FILE_READ]) + f"\n... ({len(lines) - MAX_FILE_READ} more lines)"
            return text
        except Exception:
            return None

    def _parse_response(self, raw: str) -> Diagnosis:
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            log(f"AI returned non-JSON: {text[:200]}", "WARN")
            return Diagnosis(error_type="parse_failed", description=text[:500])

        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError as exc:
            log(f"JSON parse failed: {exc}", "WARN")
            return Diagnosis(error_type="parse_failed", description=text[:500])

        return Diagnosis(
            error_type=data.get("error_type", "unknown"),
            description=data.get("description", ""),
            root_cause=data.get("root_cause", ""),
            fix_actions=data.get("fix_actions", []),
            confidence=float(data.get("confidence", 0.0)),
        )

# ---------------------------------------------------------------------------
# Fix Executor
# ---------------------------------------------------------------------------

class FixExecutor:
    """Safely executes AI-proposed fix actions."""

    def __init__(self) -> None:
        self.guard = SafetyGuard()

    def execute_actions(self, actions: list[dict[str, Any]]) -> list[FixResult]:
        results: list[FixResult] = []
        for action in actions:
            atype = action.get("type", "")
            log(f"Executing action: {atype} — {json.dumps(action, default=str)[:200]}")

            try:
                if atype == "shell":
                    r = self._run_shell(action)
                elif atype == "pip_install":
                    r = self._run_pip(action)
                elif atype == "file_edit":
                    r = self._run_file_edit(action)
                elif atype == "restart_backend":
                    r = self._restart_backend()
                elif atype == "restart_frontend":
                    r = self._restart_frontend()
                else:
                    r = FixResult(success=False, action=atype, error=f"Unknown action type: {atype}")
            except Exception as exc:
                r = FixResult(success=False, action=atype, error=str(exc))

            results.append(r)
            audit(atype, json.dumps(action, default=str)[:500],
                  "OK" if r.success else f"FAIL: {r.error[:200]}")
            log(f"  Result: {'OK' if r.success else 'FAIL'} — {r.output[:200] if r.output else r.error[:200]}")

            if not r.success:
                log(f"  Stopping action chain due to failure", "WARN")
                break

        return results

    def _run_shell(self, action: dict) -> FixResult:
        cmd = action.get("command", "")
        cwd = action.get("cwd", str(PROJECT_DIR))

        safe, reason = self.guard.is_safe_command(cmd)
        if not safe:
            return FixResult(success=False, action="shell", error=f"BLOCKED: {reason}")

        safe, reason = self.guard.is_safe_path(cwd)
        if not safe:
            return FixResult(success=False, action="shell", error=f"BLOCKED: {reason}")

        try:
            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True, text=True, timeout=60,
                cwd=cwd,
                env={**os.environ, "PATH": os.environ.get("PATH", "")},
            )
            output = result.stdout[-2000:] + result.stderr[-2000:]
            return FixResult(
                success=result.returncode == 0,
                action="shell",
                output=output.strip(),
                error="" if result.returncode == 0 else f"exit code {result.returncode}",
            )
        except subprocess.TimeoutExpired:
            return FixResult(success=False, action="shell", error="Command timed out (60s)")
        except Exception as exc:
            return FixResult(success=False, action="shell", error=str(exc))

    def _run_pip(self, action: dict) -> FixResult:
        package = action.get("package", "")
        if not package:
            return FixResult(success=False, action="pip_install", error="No package specified")

        safe, reason = self.guard.is_safe_pip_install(package)
        if not safe:
            return FixResult(success=False, action="pip_install", error=f"BLOCKED: {reason}")

        cmd = f"{VENV_PYTHON} -m pip install {package}"
        return self._run_shell({"command": cmd, "cwd": str(PROJECT_DIR)})

    def _run_file_edit(self, action: dict) -> FixResult:
        path = action.get("path", "")
        find_text = action.get("find", "")
        replace_text = action.get("replace", "")

        if not path or not find_text:
            return FixResult(success=False, action="file_edit", error="Missing path or find text")

        safe, reason = self.guard.is_safe_file_edit(path, replace_text)
        if not safe:
            return FixResult(success=False, action="file_edit", error=f"BLOCKED: {reason}")

        p = Path(path)
        if not p.exists():
            return FixResult(success=False, action="file_edit", error=f"File not found: {path}")

        try:
            content = p.read_text(encoding="utf-8")
            if find_text not in content:
                return FixResult(success=False, action="file_edit", error=f"Text not found in {path}")

            count = content.count(find_text)
            if count > 1:
                content = content.replace(find_text, replace_text, 1)
            else:
                content = content.replace(find_text, replace_text)

            backup = p.with_suffix(p.suffix + ".bak")
            if not backup.exists():
                p.rename(backup)
                backup_path = backup
            else:
                backup_path = None

            p.write_text(content, encoding="utf-8")
            return FixResult(
                success=True, action="file_edit",
                output=f"Edited {path} (replaced {count} occurrence(s))",
            )
        except Exception as exc:
            return FixResult(success=False, action="file_edit", error=str(exc))

    def _restart_backend(self) -> FixResult:
        try:
            subprocess.run(
                ["bash", "-c", "lsof -ti:8000 | xargs kill 2>/dev/null || true"],
                capture_output=True, timeout=10,
            )
            time.sleep(2)

            subprocess.Popen(
                [VENV_PYTHON, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
                cwd=str(BACKEND_DIR),
                stdout=open(str(BACKEND_LOG), "a"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            for _ in range(30):
                time.sleep(2)
                try:
                    import urllib.request
                    with urllib.request.urlopen("http://localhost:8000/health", timeout=5):
                        return FixResult(success=True, action="restart_backend", output="Backend restarted and healthy")
                except Exception:
                    continue

            return FixResult(success=False, action="restart_backend", error="Backend started but health check failed after 60s")
        except Exception as exc:
            return FixResult(success=False, action="restart_backend", error=str(exc))

    def _restart_frontend(self) -> FixResult:
        try:
            subprocess.run(
                ["bash", "-c", "lsof -ti:7860 | xargs kill 2>/dev/null || true"],
                capture_output=True, timeout=10,
            )
            time.sleep(2)

            subprocess.Popen(
                [VENV_PYTHON, "main.py", "--port", "7860", "--api-url", "http://localhost:8000"],
                cwd=str(FRONTEND_DIR / "desktop"),
                stdout=open(str(FRONTEND_LOG), "a"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            time.sleep(5)
            return FixResult(success=True, action="restart_frontend", output="Frontend restarted")
        except Exception as exc:
            return FixResult(success=False, action="restart_frontend", error=str(exc))

# ---------------------------------------------------------------------------
# Watchdog AI Orchestrator
# ---------------------------------------------------------------------------

class WatchdogAI:
    """Main orchestrator: check -> diagnose -> fix -> verify loop."""

    def __init__(self) -> None:
        self._load_env()
        self.ai = AIClient()
        self.checker = HealthChecker()
        self.diagnoser = AIDiagnoser(self.ai)
        self.executor = FixExecutor()
        self.check_interval = int(os.getenv("WATCHDOG_CHECK_INTERVAL", "120"))
        self.max_retries = int(os.getenv("WATCHDOG_MAX_RETRIES", "3"))
        self._error_counts: dict[str, int] = {}
        self._total_fixes = 0
        self._total_api_calls = 0

    def _load_env(self) -> None:
        env_path = PROJECT_DIR / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if key and value and key not in os.environ:
                        os.environ[key] = value

    def run_once(self) -> dict[str, Any]:
        """Single check-diagnose-fix cycle. Returns summary."""
        log("=== Health check started ===")
        errors = self.checker.check_all()

        if not errors:
            log("All checks passed — system healthy")
            return {"status": "healthy", "errors": 0, "fixed": 0}

        log(f"Found {len(errors)} error(s)")
        fixed = 0

        for error in errors:
            etype = error["type"]
            count = self._error_counts.get(etype, 0)

            if count >= self.max_retries:
                log(f"Skipping {etype} — max retries ({self.max_retries}) reached", "WARN")
                continue

            log(f"Diagnosing: {etype}")
            self._total_api_calls += 1
            diagnosis = self.diagnoser.diagnose(error)

            if not diagnosis.fix_actions:
                log(f"AI returned no fix actions for {etype}", "WARN")
                self._error_counts[etype] = count + 1
                continue

            if diagnosis.confidence < 0.2:
                log(f"AI confidence too low ({diagnosis.confidence:.1%}) — skipping", "WARN")
                self._error_counts[etype] = count + 1
                continue

            log(f"AI diagnosis: {diagnosis.description} (confidence: {diagnosis.confidence:.0%})")
            log(f"Root cause: {diagnosis.root_cause}")
            log(f"Fix actions: {len(diagnosis.fix_actions)}")

            results = self.executor.execute_actions(diagnosis.fix_actions)
            all_ok = all(r.success for r in results)

            if all_ok:
                log(f"FIXED: {etype}")
                fixed += 1
                self._total_fixes += 1
                self._error_counts.pop(etype, None)
            else:
                failed = [r for r in results if not r.success]
                desc = f"{etype}: {', '.join(r.error[:80] for r in failed)}"
                self.diagnoser.record_attempt(desc)
                self._error_counts[etype] = count + 1
                log(f"Fix attempt {count + 1}/{self.max_retries} failed for {etype}", "WARN")

        # Verify after all fixes
        if fixed > 0:
            log("Verifying fixes...")
            time.sleep(5)
            remaining = self.checker.check_all()
            still_broken = len(remaining)
            log(f"Post-fix status: {still_broken} error(s) remaining")
        else:
            still_broken = len(errors)

        summary = {
            "status": "healthy" if still_broken == 0 else "degraded",
            "errors_found": len(errors),
            "fixed": fixed,
            "remaining": still_broken,
            "total_fixes": self._total_fixes,
            "total_api_calls": self._total_api_calls,
        }
        log(f"=== Check complete: {json.dumps(summary)} ===")
        return summary

    def monitor(self) -> None:
        """Continuous monitoring loop."""
        log("========================================")
        log("  MIYA AI WATCHDOG — MONITOR MODE")
        log(f"  Check interval: {self.check_interval}s")
        log(f"  Max retries per error: {self.max_retries}")
        log(f"  Anthropic key: {'SET' if self.ai.anthropic_key else 'NOT SET'}")
        log(f"  OpenAI key: {'SET' if self.ai.openai_key else 'NOT SET'}")
        log("========================================")

        while True:
            try:
                self.run_once()
            except Exception as exc:
                log(f"Monitor loop error: {exc}", "ERROR")
                log(traceback.format_exc(), "ERROR")

            log(f"Next check in {self.check_interval}s...")
            time.sleep(self.check_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="MIYA AI Watchdog")
    parser.add_argument("--check", action="store_true", help="Run one health check cycle")
    parser.add_argument("--monitor", action="store_true", help="Continuous monitoring loop")
    parser.add_argument("--diagnose", metavar="FILE", help="Diagnose a specific log file")
    args = parser.parse_args()

    if args.check:
        try:
            wd = WatchdogAI()
            result = wd.run_once()
        except RuntimeError:
            # No API keys — just run health checks without AI
            checker = HealthChecker()
            errors = checker.check_all()
            result = {
                "status": "healthy" if not errors else "errors_found",
                "errors": [{"type": e["type"], "details": e["details"][:200]} for e in errors],
                "note": "AI diagnosis unavailable — no API keys set",
            }
        print(json.dumps(result, indent=2))
    elif args.monitor:
        wd = WatchdogAI()
        wd.monitor()
    elif args.diagnose:
        wd = WatchdogAI()
        path = Path(args.diagnose)
        if not path.exists():
            print(f"File not found: {args.diagnose}")
            sys.exit(1)
        content = path.read_text(errors="replace")[-5000:]
        error = {"type": "manual_diagnosis", "details": content}
        diagnosis = wd.diagnoser.diagnose(error)
        print(json.dumps({
            "error_type": diagnosis.error_type,
            "description": diagnosis.description,
            "root_cause": diagnosis.root_cause,
            "confidence": diagnosis.confidence,
            "fix_actions": diagnosis.fix_actions,
        }, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
