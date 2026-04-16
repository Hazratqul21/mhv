from __future__ import annotations

import argparse
import asyncio
import uuid
from typing import Any

import gradio as gr
import httpx

from chat_interface import build_chat_column, format_response
from settings_panel import AppSettings, build_settings_panel

_settings = AppSettings()
_session_id: str = f"sess_{uuid.uuid4().hex[:12]}"
_client = httpx.AsyncClient(timeout=120.0)


async def _api(path: str, **kwargs) -> dict[str, Any]:
    url = f"{_settings.api_url}/api/v1{path}"
    resp = await _client.post(url, json=kwargs)
    resp.raise_for_status()
    return resp.json()


async def send_message(
    user_msg: str,
    history: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    if not user_msg.strip():
        return "", history

    history = history + [{"role": "user", "content": user_msg}]

    try:
        payload: dict[str, Any] = {
            "message": user_msg,
            "session_id": _session_id,
            "stream": _settings.stream,
        }
        if _settings.temperature is not None:
            payload["temperature"] = _settings.temperature
        if _settings.max_tokens:
            payload["max_tokens"] = _settings.max_tokens
        result = await _api("/chat", **payload)
        reply = format_response(result)
    except httpx.HTTPStatusError as exc:
        reply = f"**Error {exc.response.status_code}**: {exc.response.text}"
    except httpx.ConnectError:
        reply = "Cannot reach the Miya API. Is the backend running?"
    except Exception as exc:
        reply = f"**Error**: {exc}"

    history = history + [{"role": "assistant", "content": reply}]
    return "", history


async def upload_file(file_bytes: bytes | None) -> str:
    if file_bytes is None:
        return "No file selected."
    try:
        url = f"{_settings.api_url}/api/v1/upload"
        resp = await _client.post(url, files={"file": ("upload", file_bytes)})
        resp.raise_for_status()
        data = resp.json()
        return f"Uploaded **{data['filename']}** ({data['size_bytes']} bytes)"
    except Exception as exc:
        return f"Upload failed: {exc}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Miya AI") as demo:
        gr.Markdown("# Miya AI Assistant")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot, msg_box, send_btn, file_upload = build_chat_column(
                    send_fn=send_message,
                    upload_fn=upload_file,
                )

            settings_col, settings_refs = build_settings_panel(_settings)

        # Wire settings changes back to the runtime config object
        def _update_api_url(val: str):
            _settings.api_url = val.rstrip("/")

        def _update_temp(val: float):
            _settings.temperature = val

        def _update_tokens(val: int):
            _settings.max_tokens = val

        def _update_stream(val: bool):
            _settings.stream = val

        settings_refs["api_url"].change(_update_api_url, inputs=[settings_refs["api_url"]])
        settings_refs["temperature"].change(_update_temp, inputs=[settings_refs["temperature"]])
        settings_refs["max_tokens"].change(_update_tokens, inputs=[settings_refs["max_tokens"]])
        settings_refs["stream"].change(_update_stream, inputs=[settings_refs["stream"]])

        # New session button
        def new_session():
            global _session_id
            _session_id = f"sess_{uuid.uuid4().hex[:12]}"
            return []

        gr.Button("New Session").click(fn=new_session, outputs=[chatbot])

    return demo


def main():
    parser = argparse.ArgumentParser(description="Miya desktop UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    _settings.api_url = args.api_url
    demo = build_ui()
    demo.launch(
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(primary_hue="indigo"),
        css="#miya-chatbot { font-size: 15px; }",
    )


if __name__ == "__main__":
    main()
