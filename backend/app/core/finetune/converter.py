from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)

QUANT_TYPES = ("q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16")


class GGUFConverter:
    """Converts a fine-tuned HF model (merged 16-bit) to GGUF format.

    Strategies:
    1. Unsloth built-in: model.save_pretrained_gguf (fastest)
    2. llama.cpp convert_hf_to_gguf.py + quantize binary (fallback)
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    async def convert(
        self,
        merged_model_dir: str,
        output_name: str = "miya-finetuned",
        quant_type: str = "q4_k_m",
    ) -> dict[str, Any]:
        if quant_type not in QUANT_TYPES:
            raise ValueError(f"Unsupported quant type: {quant_type}. Use: {QUANT_TYPES}")

        merged_path = Path(merged_model_dir)
        if not merged_path.exists():
            raise FileNotFoundError(f"Merged model not found: {merged_model_dir}")

        output_dir = Path(self._settings.finetune_output_dir) / "gguf"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{output_name}-{quant_type}.gguf"

        log.info(
            "conversion_started",
            input=merged_model_dir,
            output=str(output_file),
            quant=quant_type,
        )

        try:
            result = await self._convert_with_unsloth(
                merged_path, output_file, quant_type
            )
        except Exception as exc:
            log.warning("unsloth_convert_failed", error=str(exc))
            result = await self._convert_with_llama_cpp(
                merged_path, output_file, quant_type
            )

        return result

    async def _convert_with_unsloth(
        self,
        merged_path: Path,
        output_file: Path,
        quant_type: str,
    ) -> dict[str, Any]:
        def _do_convert() -> dict[str, Any]:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(merged_path),
                load_in_4bit=False,
            )

            model.save_pretrained_gguf(
                str(output_file.parent),
                tokenizer,
                quantization_method=quant_type,
            )

            final = output_file.parent / f"unsloth.{quant_type.upper()}.gguf"
            if final.exists() and final != output_file:
                shutil.move(str(final), str(output_file))

            return {
                "method": "unsloth",
                "output_path": str(output_file),
                "size_mb": output_file.stat().st_size / (1024 * 1024)
                if output_file.exists()
                else 0,
            }

        return await asyncio.to_thread(_do_convert)

    async def _convert_with_llama_cpp(
        self,
        merged_path: Path,
        output_file: Path,
        quant_type: str,
    ) -> dict[str, Any]:
        f16_gguf = output_file.parent / f"{output_file.stem}-f16.gguf"

        convert_script = self._find_convert_script()
        if not convert_script:
            raise RuntimeError(
                "llama.cpp convert_hf_to_gguf.py not found. "
                "Install: pip install llama-cpp-python"
            )

        proc = await asyncio.create_subprocess_exec(
            "python3",
            str(convert_script),
            str(merged_path),
            "--outtype",
            "f16",
            "--outfile",
            str(f16_gguf),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Convert failed: {stderr.decode()}")

        if quant_type != "f16":
            quantize_bin = shutil.which("llama-quantize") or shutil.which("quantize")
            if quantize_bin:
                proc2 = await asyncio.create_subprocess_exec(
                    quantize_bin,
                    str(f16_gguf),
                    str(output_file),
                    quant_type.upper(),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc2.communicate()
                f16_gguf.unlink(missing_ok=True)
            else:
                shutil.move(str(f16_gguf), str(output_file))
                log.warning("quantize_binary_not_found_using_f16")
        else:
            shutil.move(str(f16_gguf), str(output_file))

        return {
            "method": "llama_cpp",
            "output_path": str(output_file),
            "size_mb": output_file.stat().st_size / (1024 * 1024)
            if output_file.exists()
            else 0,
        }

    def _find_convert_script(self) -> Path | None:
        candidates = [
            Path("/usr/local/lib/python3.11/site-packages/llama_cpp/convert_hf_to_gguf.py"),
            Path.home() / ".local" / "lib" / "python3.11" / "site-packages" / "llama_cpp" / "convert_hf_to_gguf.py",
        ]

        try:
            import llama_cpp
            pkg_dir = Path(llama_cpp.__file__).parent
            convert = pkg_dir / "convert_hf_to_gguf.py"
            if convert.exists():
                return convert
        except ImportError:
            pass

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    async def list_converted(self) -> list[dict[str, Any]]:
        gguf_dir = Path(self._settings.finetune_output_dir) / "gguf"
        if not gguf_dir.exists():
            return []

        models = []
        for gguf in gguf_dir.glob("*.gguf"):
            models.append({
                "name": gguf.name,
                "path": str(gguf),
                "size_mb": gguf.stat().st_size / (1024 * 1024),
            })
        return models
