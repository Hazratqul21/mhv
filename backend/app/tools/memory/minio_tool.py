from __future__ import annotations

import io
from typing import Any

from minio import Minio
from minio.error import S3Error

from app.config import get_settings
from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MinioTool(BaseTool):
    name = "minio"
    description = "Object storage operations: upload, download, and list objects in MinIO"
    category = "memory"
    parameters = {
        "action": {
            "type": "string",
            "enum": ["upload", "download", "list_objects"],
            "description": "Operation to perform",
        },
        "bucket": {
            "type": "string",
            "description": "Bucket name (defaults to configured bucket)",
        },
        "object_name": {
            "type": "string",
            "description": "Object key / path inside the bucket",
        },
        "data": {
            "type": "string",
            "description": "Base64-encoded data or UTF-8 text to upload",
        },
        "content_type": {
            "type": "string",
            "description": "MIME type of the uploaded object",
            "default": "application/octet-stream",
        },
        "prefix": {
            "type": "string",
            "description": "Prefix filter for list_objects",
        },
    }

    def __init__(self) -> None:
        settings = get_settings()
        self._client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_root_user,
            secret_key=settings.minio_root_password,
            secure=False,
        )
        self._default_bucket = settings.minio_bucket

    def _ensure_bucket(self, bucket: str) -> None:
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)

    async def execute(self, input_data: dict[str, Any]) -> Any:
        action = input_data.get("action", "")
        bucket = input_data.get("bucket", self._default_bucket)

        try:
            self._ensure_bucket(bucket)

            if action == "upload":
                object_name = input_data.get("object_name", "")
                data_str = input_data.get("data", "")
                content_type = input_data.get("content_type", "application/octet-stream")
                if not object_name or not data_str:
                    return {"error": "'object_name' and 'data' are required"}

                raw = data_str.encode("utf-8")
                stream = io.BytesIO(raw)
                self._client.put_object(
                    bucket,
                    object_name,
                    stream,
                    length=len(raw),
                    content_type=content_type,
                )
                return {
                    "status": "ok",
                    "bucket": bucket,
                    "object_name": object_name,
                    "size": len(raw),
                }

            if action == "download":
                object_name = input_data.get("object_name", "")
                if not object_name:
                    return {"error": "'object_name' is required"}
                response = self._client.get_object(bucket, object_name)
                data = response.read()
                response.close()
                response.release_conn()
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    import base64
                    text = base64.b64encode(data).decode("ascii")
                return {
                    "bucket": bucket,
                    "object_name": object_name,
                    "size": len(data),
                    "content": text,
                }

            if action == "list_objects":
                prefix = input_data.get("prefix", "")
                objects = self._client.list_objects(bucket, prefix=prefix, recursive=True)
                items = []
                for obj in objects:
                    items.append({
                        "name": obj.object_name,
                        "size": obj.size,
                        "last_modified": str(obj.last_modified) if obj.last_modified else None,
                    })
                return {"bucket": bucket, "prefix": prefix, "objects": items}

            return {"error": f"Unknown action '{action}'"}

        except S3Error as exc:
            logger.error("minio_s3_error", action=action, error=str(exc))
            return {"error": str(exc)}
        except Exception as exc:
            logger.error("minio_error", action=action, error=str(exc))
            return {"error": str(exc)}
