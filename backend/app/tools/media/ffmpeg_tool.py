from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class FFmpegTool(BaseTool):
    name = "ffmpeg"
    description = "Media conversion: transcode, extract audio, and resize video/audio via ffmpeg"
    category = "media"
    parameters = {
        "action": {
            "type": "string",
            "enum": ["convert", "extract_audio", "resize", "info"],
            "description": "Operation to perform",
        },
        "input_path": {
            "type": "string",
            "description": "Input file path",
        },
        "output_path": {
            "type": "string",
            "description": "Output file path",
        },
        "format": {
            "type": "string",
            "description": "Target format (mp4, mp3, wav, webm, etc.)",
        },
        "width": {
            "type": "integer",
            "description": "Target width for resize",
        },
        "height": {
            "type": "integer",
            "description": "Target height for resize (-1 to keep aspect)",
            "default": -1,
        },
        "extra_args": {
            "type": "array",
            "description": "Additional ffmpeg arguments",
        },
    }

    def __init__(self) -> None:
        self._ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        self._ffprobe = shutil.which("ffprobe") or "ffprobe"

    async def _run(self, cmd: list[str], timeout: int = 120) -> dict[str, Any]:
        logger.debug("ffmpeg_cmd", cmd=" ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"error": f"Command timed out after {timeout}s"}

        if proc.returncode != 0:
            return {"error": stderr.decode(errors="replace").strip()}
        return {"stdout": stdout.decode(errors="replace"), "stderr": stderr.decode(errors="replace")}

    async def execute(self, input_data: dict[str, Any]) -> Any:
        action = input_data.get("action", "")
        input_path = input_data.get("input_path", "")

        if not input_path:
            return {"error": "'input_path' is required"}
        if not Path(input_path).exists():
            return {"error": f"File not found: {input_path}"}

        try:
            if action == "info":
                cmd = [
                    self._ffprobe, "-v", "quiet",
                    "-print_format", "json",
                    "-show_format", "-show_streams",
                    input_path,
                ]
                result = await self._run(cmd)
                if "error" in result:
                    return result
                import json
                return {"info": json.loads(result["stdout"])}

            output_path = input_data.get("output_path", "")
            if not output_path:
                return {"error": "'output_path' is required for this action"}

            if action == "convert":
                fmt = input_data.get("format", "")
                extra = input_data.get("extra_args", [])
                cmd = [self._ffmpeg, "-y", "-i", input_path]
                if fmt:
                    cmd.extend(["-f", fmt])
                cmd.extend(extra)
                cmd.append(output_path)
                result = await self._run(cmd)
                if "error" in result:
                    return result
                return {"status": "ok", "output_path": output_path}

            if action == "extract_audio":
                cmd = [
                    self._ffmpeg, "-y", "-i", input_path,
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    output_path,
                ]
                result = await self._run(cmd)
                if "error" in result:
                    return result
                return {"status": "ok", "output_path": output_path}

            if action == "resize":
                width = input_data.get("width", 1280)
                height = input_data.get("height", -1)
                scale = f"scale={width}:{height}"
                cmd = [
                    self._ffmpeg, "-y", "-i", input_path,
                    "-vf", scale, output_path,
                ]
                result = await self._run(cmd)
                if "error" in result:
                    return result
                return {"status": "ok", "output_path": output_path}

            return {"error": f"Unknown action '{action}'"}

        except Exception as exc:
            logger.error("ffmpeg_error", action=action, error=str(exc))
            return {"error": str(exc)}
