from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OpenCVTool(BaseTool):
    name = "opencv"
    description = "Computer vision utilities: face detection, image resizing, and basic transformations"
    category = "media"
    parameters = {
        "action": {
            "type": "string",
            "enum": ["detect_faces", "resize_image", "to_grayscale", "info"],
            "description": "Operation to perform",
        },
        "image_path": {
            "type": "string",
            "description": "Path to the input image",
        },
        "output_path": {
            "type": "string",
            "description": "Path for the output image",
        },
        "width": {
            "type": "integer",
            "description": "Target width for resize",
        },
        "height": {
            "type": "integer",
            "description": "Target height for resize",
        },
    }

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

    async def execute(self, input_data: dict[str, Any]) -> Any:
        action = input_data.get("action", "")
        image_path = input_data.get("image_path", "")

        if not image_path:
            return {"error": "'image_path' is required"}
        if not Path(image_path).exists():
            return {"error": f"File not found: {image_path}"}

        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"error": f"Could not decode image: {image_path}"}

            if action == "info":
                h, w = img.shape[:2]
                channels = img.shape[2] if len(img.shape) == 3 else 1
                return {
                    "width": w,
                    "height": h,
                    "channels": channels,
                    "dtype": str(img.dtype),
                }

            if action == "detect_faces":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self._face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                )
                face_list = [
                    {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                    for (x, y, w, h) in faces
                ]
                output_path = input_data.get("output_path")
                if output_path and len(faces) > 0:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imwrite(output_path, img)

                return {"faces": face_list, "count": len(face_list)}

            if action == "resize_image":
                width = input_data.get("width")
                height = input_data.get("height")
                if not width and not height:
                    return {"error": "'width' or 'height' is required for resize"}
                h, w = img.shape[:2]
                if width and not height:
                    ratio = width / w
                    height = int(h * ratio)
                elif height and not width:
                    ratio = height / h
                    width = int(w * ratio)
                resized = cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_AREA)
                output_path = input_data.get("output_path", "")
                if output_path:
                    cv2.imwrite(output_path, resized)
                    return {"status": "ok", "output_path": output_path, "width": int(width), "height": int(height)}
                _, buf = cv2.imencode(".png", resized)
                return {
                    "width": int(width),
                    "height": int(height),
                    "image_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                }

            if action == "to_grayscale":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                output_path = input_data.get("output_path", "")
                if output_path:
                    cv2.imwrite(output_path, gray)
                    return {"status": "ok", "output_path": output_path}
                _, buf = cv2.imencode(".png", gray)
                return {"image_b64": base64.b64encode(buf.tobytes()).decode("ascii")}

            return {"error": f"Unknown action '{action}'"}

        except Exception as exc:
            logger.error("opencv_error", action=action, error=str(exc))
            return {"error": str(exc)}
