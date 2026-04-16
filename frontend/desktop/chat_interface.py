from __future__ import annotations

from typing import Any, Callable, Coroutine

import gradio as gr


def create_chatbot() -> gr.Chatbot:
    """Pre-configured Chatbot component with markdown + code-block rendering."""
    return gr.Chatbot(
        label="Miya",
        elem_id="miya-chatbot",
        height=560,
    )


def format_response(result: dict[str, Any]) -> str:
    """Build a rich display string from an API response."""
    text = result.get("response", "")
    parts: list[str] = [text]

    agent = result.get("agent_used")
    tools = result.get("tools_used", [])
    exec_time = result.get("execution_time_ms", 0)
    confidence = result.get("confidence", 0.0)

    footer_bits: list[str] = []
    if agent:
        footer_bits.append(f"Agent: **{agent}**")
    if tools:
        footer_bits.append(f"Tools: {', '.join(tools)}")
    if exec_time:
        footer_bits.append(f"{exec_time} ms")
    if confidence is not None:
        footer_bits.append(f"Confidence: {confidence:.0%}")

    if footer_bits:
        parts.append("\n\n---\n" + " · ".join(footer_bits))

    return "".join(parts)


def build_chat_column(
    send_fn: Callable[..., Coroutine],
    upload_fn: Callable[..., Coroutine] | None = None,
) -> tuple[gr.Chatbot, gr.Textbox, gr.Button, gr.File | None]:
    """Compose the main chat column and wire up events.

    Returns the key components so the caller can attach additional logic.
    """
    chatbot = create_chatbot()

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Ask Miya anything …",
            show_label=False,
            lines=1,
            scale=8,
            elem_id="miya-input",
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    file_upload: gr.File | None = None
    if upload_fn is not None:
        file_upload = gr.File(
            label="Attach file (max 10 MB)",
            file_count="single",
            type="binary",
            visible=True,
        )
        upload_status = gr.Textbox(label="Upload status", interactive=False, visible=True)
        file_upload.upload(
            fn=upload_fn,
            inputs=[file_upload],
            outputs=[upload_status],
        )

    send_btn.click(
        fn=send_fn,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot],
    )
    msg_box.submit(
        fn=send_fn,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot],
    )

    return chatbot, msg_box, send_btn, file_upload
