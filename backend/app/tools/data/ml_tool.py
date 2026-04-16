from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np

from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MLTool(BaseTool):
    name = "ml"
    description = "Machine learning utilities: train simple models, predict, and classify using scikit-learn"
    category = "data"
    parameters = {
        "action": {
            "type": "string",
            "enum": ["train", "predict", "classify", "cluster", "evaluate"],
            "description": "ML operation",
        },
        "model_type": {
            "type": "string",
            "enum": [
                "linear_regression",
                "logistic_regression",
                "random_forest_classifier",
                "random_forest_regressor",
                "kmeans",
            ],
            "description": "Model type to use",
        },
        "X": {
            "type": "array",
            "description": "Feature matrix (2-D list)",
        },
        "y": {
            "type": "array",
            "description": "Target labels/values",
        },
        "n_clusters": {
            "type": "integer",
            "description": "Number of clusters for KMeans",
            "default": 3,
        },
        "model_b64": {
            "type": "string",
            "description": "Base64-encoded pickled model (for predict)",
        },
    }

    _MODEL_MAP: dict[str, type] = {}

    @staticmethod
    def _get_model(model_type: str, **kwargs: Any):
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.cluster import KMeans

        models = {
            "linear_regression": LinearRegression,
            "logistic_regression": LogisticRegression,
            "random_forest_classifier": RandomForestClassifier,
            "random_forest_regressor": RandomForestRegressor,
            "kmeans": KMeans,
        }
        cls = models.get(model_type)
        if cls is None:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls(**kwargs)

    async def execute(self, input_data: dict[str, Any]) -> Any:
        action = input_data.get("action", "")

        try:
            if action == "train":
                return self._train(input_data)
            if action == "predict":
                return self._predict(input_data)
            if action == "classify":
                return self._classify(input_data)
            if action == "cluster":
                return self._cluster(input_data)
            if action == "evaluate":
                return self._evaluate(input_data)
            return {"error": f"Unknown action '{action}'"}

        except Exception as exc:
            logger.error("ml_error", action=action, error=str(exc))
            return {"error": str(exc)}

    def _train(self, d: dict[str, Any]) -> dict[str, Any]:
        import base64
        from sklearn.model_selection import train_test_split

        X = np.array(d.get("X", []))
        y = np.array(d.get("y", []))
        model_type = d.get("model_type", "linear_regression")

        if X.size == 0 or y.size == 0:
            return {"error": "'X' and 'y' are required"}

        kwargs: dict[str, Any] = {}
        if model_type == "kmeans":
            kwargs["n_clusters"] = int(d.get("n_clusters", 3))

        model = self._get_model(model_type, **kwargs)

        if model_type == "kmeans":
            model.fit(X)
        else:
            model.fit(X, y)

        buf = io.BytesIO()
        pickle.dump(model, buf)
        model_bytes = buf.getvalue()

        return {
            "status": "ok",
            "model_type": model_type,
            "model_b64": base64.b64encode(model_bytes).decode("ascii"),
            "model_size_bytes": len(model_bytes),
        }

    def _predict(self, d: dict[str, Any]) -> dict[str, Any]:
        import base64

        model_b64 = d.get("model_b64", "")
        X = np.array(d.get("X", []))
        if not model_b64 or X.size == 0:
            return {"error": "'model_b64' and 'X' are required"}

        model = pickle.loads(base64.b64decode(model_b64))
        predictions = model.predict(X)
        return {"predictions": predictions.tolist()}

    def _classify(self, d: dict[str, Any]) -> dict[str, Any]:
        import base64

        model_b64 = d.get("model_b64", "")
        X = np.array(d.get("X", []))
        if not model_b64 or X.size == 0:
            return {"error": "'model_b64' and 'X' are required"}

        model = pickle.loads(base64.b64decode(model_b64))
        predictions = model.predict(X)

        result: dict[str, Any] = {"predictions": predictions.tolist()}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            result["probabilities"] = proba.tolist()
        return result

    def _cluster(self, d: dict[str, Any]) -> dict[str, Any]:
        X = np.array(d.get("X", []))
        n_clusters = int(d.get("n_clusters", 3))
        if X.size == 0:
            return {"error": "'X' is required"}

        model = self._get_model("kmeans", n_clusters=n_clusters)
        labels = model.fit_predict(X)
        return {
            "labels": labels.tolist(),
            "centers": model.cluster_centers_.tolist(),
            "inertia": float(model.inertia_),
        }

    def _evaluate(self, d: dict[str, Any]) -> dict[str, Any]:
        import base64
        from sklearn.metrics import (
            accuracy_score,
            mean_squared_error,
            r2_score,
            classification_report,
        )

        model_b64 = d.get("model_b64", "")
        X = np.array(d.get("X", []))
        y = np.array(d.get("y", []))
        if not model_b64 or X.size == 0 or y.size == 0:
            return {"error": "'model_b64', 'X', and 'y' are required"}

        model = pickle.loads(base64.b64decode(model_b64))
        preds = model.predict(X)

        is_classifier = hasattr(model, "predict_proba") or hasattr(model, "classes_")
        if is_classifier:
            return {
                "accuracy": float(accuracy_score(y, preds)),
                "report": classification_report(y, preds, output_dict=True),
            }
        return {
            "mse": float(mean_squared_error(y, preds)),
            "r2": float(r2_score(y, preds)),
        }
