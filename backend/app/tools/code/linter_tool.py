from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LinterTool(BaseTool):
    name = "linter"
    description = "Lint and format-check code (Python via ruff, with extensibility for other languages)"
    category = "code"
    parameters = {
        "code": {
            "type": "string",
            "description": "Source code to lint",
        },
        "language": {
            "type": "string",
            "description": "Programming language",
            "default": "python",
        },
        "fix": {
            "type": "boolean",
            "description": "Apply auto-fixes and return corrected code",
            "default": False,
        },
    }

    async def execute(self, input_data: dict[str, Any]) -> Any:
        code = input_data.get("code", "")
        if not code:
            return {"error": "'code' is required"}

        language = input_data.get("language", "python").lower()
        fix = input_data.get("fix", False)

        if language == "python":
            return await self._lint_python(code, fix)

        return {"error": f"Linting not yet supported for '{language}'"}

    async def _lint_python(self, code: str, fix: bool) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            cmd = ["ruff", "check", "--output-format=json"]
            if fix:
                cmd.append("--fix")
            cmd.append(tmp_path)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            import json
            diagnostics = []
            raw = stdout.decode(errors="replace").strip()
            if raw:
                try:
                    diagnostics = json.loads(raw)
                except json.JSONDecodeError:
                    diagnostics = [{"message": raw}]

            result: dict[str, Any] = {
                "language": "python",
                "issues": [
                    {
                        "code": d.get("code", ""),
                        "message": d.get("message", ""),
                        "line": d.get("location", {}).get("row"),
                        "col": d.get("location", {}).get("column"),
                    }
                    for d in diagnostics
                    if isinstance(d, dict)
                ],
                "issue_count": len(diagnostics),
            }

            if fix:
                result["fixed_code"] = Path(tmp_path).read_text()

            return result

        except FileNotFoundError:
            return {"error": "'ruff' is not installed. Install with: pip install ruff"}
        except Exception as exc:
            logger.error("linter_error", error=str(exc))
            return {"error": str(exc)}
        finally:
            Path(tmp_path).unlink(missing_ok=True)
