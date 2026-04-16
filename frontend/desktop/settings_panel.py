from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gradio as gr


@dataclass
class AppSettings:
    """Runtime settings managed by the settings panel."""

    api_url: str = "http://localhost:8000"
    model: str = "auto"
    temperature: float = 0.7
    max_tokens: int = 1024
    dark_theme: bool = True
    stream: bool = True


def build_settings_panel(
    defaults: AppSettings | None = None,
) -> tuple[gr.Column, dict[str, Any]]:
    """Create the settings sidebar and return a dict of component references.

    Usage::

        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column(scale=3):
                    ...  # chat
                col, refs = build_settings_panel()
    """
    defaults = defaults or AppSettings()
    refs: dict[str, Any] = {}

    with gr.Column(scale=1, min_width=260) as col:
        gr.Markdown("### Settings")

        refs["api_url"] = gr.Textbox(
            label="API URL",
            value=defaults.api_url,
            interactive=True,
        )
        refs["model"] = gr.Dropdown(
            label="Model",
            choices=["auto", "chat", "code", "vision"],
            value=defaults.model,
            interactive=True,
        )
        refs["temperature"] = gr.Slider(
            label="Temperature",
            minimum=0.0,
            maximum=2.0,
            step=0.05,
            value=defaults.temperature,
        )
        refs["max_tokens"] = gr.Slider(
            label="Max Tokens",
            minimum=64,
            maximum=8192,
            step=64,
            value=defaults.max_tokens,
        )
        refs["stream"] = gr.Checkbox(
            label="Stream responses",
            value=defaults.stream,
        )
        refs["theme_toggle"] = gr.Button("Toggle light / dark")

    return col, refs
