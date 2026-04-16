from __future__ import annotations

from typing import Any

import numpy as np

from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class NumpyTool(BaseTool):
    name = "numpy"
    description = "Numerical operations: statistics, matrix math, and array manipulation"
    category = "data"
    parameters = {
        "action": {
            "type": "string",
            "enum": ["stats", "matrix_multiply", "dot", "eigenvalues", "solve", "normalize"],
            "description": "Operation to perform",
        },
        "data": {
            "type": "array",
            "description": "Input array or matrix (nested list)",
        },
        "data_b": {
            "type": "array",
            "description": "Second matrix (for multiply, dot, solve)",
        },
    }

    async def execute(self, input_data: dict[str, Any]) -> Any:
        action = input_data.get("action", "")
        raw = input_data.get("data")

        if raw is None:
            return {"error": "'data' is required"}

        try:
            arr = np.array(raw, dtype=float)

            if action == "stats":
                result = {
                    "shape": list(arr.shape),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "median": float(np.median(arr)),
                    "sum": float(np.sum(arr)),
                }
                if arr.ndim >= 2:
                    result["column_means"] = np.mean(arr, axis=0).tolist()
                return result

            if action == "matrix_multiply":
                raw_b = input_data.get("data_b")
                if raw_b is None:
                    return {"error": "'data_b' required for matrix_multiply"}
                b = np.array(raw_b, dtype=float)
                product = (arr @ b).tolist()
                return {"result": product}

            if action == "dot":
                raw_b = input_data.get("data_b")
                if raw_b is None:
                    return {"error": "'data_b' required for dot"}
                b = np.array(raw_b, dtype=float)
                return {"result": float(np.dot(arr.flatten(), b.flatten()))}

            if action == "eigenvalues":
                if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                    return {"error": "Eigenvalue decomposition requires a square matrix"}
                vals, vecs = np.linalg.eig(arr)
                return {
                    "eigenvalues": vals.real.tolist(),
                    "eigenvectors": vecs.real.tolist(),
                }

            if action == "solve":
                raw_b = input_data.get("data_b")
                if raw_b is None:
                    return {"error": "'data_b' required for solve (Ax = b)"}
                b = np.array(raw_b, dtype=float)
                x = np.linalg.solve(arr, b)
                return {"solution": x.tolist()}

            if action == "normalize":
                norm = np.linalg.norm(arr)
                if norm == 0:
                    return {"error": "Cannot normalize a zero vector/matrix"}
                return {"result": (arr / norm).tolist(), "norm": float(norm)}

            return {"error": f"Unknown action '{action}'"}

        except np.linalg.LinAlgError as exc:
            return {"error": f"Linear algebra error: {exc}"}
        except Exception as exc:
            logger.error("numpy_error", action=action, error=str(exc))
            return {"error": str(exc)}
