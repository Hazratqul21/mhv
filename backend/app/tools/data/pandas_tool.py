from __future__ import annotations

import io
from typing import Any

import pandas as pd

from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PandasTool(BaseTool):
    name = "pandas"
    description = "Tabular data analysis: read CSV/JSON, describe statistics, and query dataframes"
    category = "data"
    parameters = {
        "action": {
            "type": "string",
            "enum": ["read_csv", "read_json", "describe", "query_df", "head", "value_counts"],
            "description": "Operation to perform",
        },
        "file_path": {
            "type": "string",
            "description": "Path to the data file",
        },
        "csv_data": {
            "type": "string",
            "description": "Inline CSV data (alternative to file_path)",
        },
        "query": {
            "type": "string",
            "description": "Pandas query expression (for query_df)",
        },
        "column": {
            "type": "string",
            "description": "Column name (for value_counts)",
        },
        "n_rows": {
            "type": "integer",
            "description": "Number of rows for head",
            "default": 10,
        },
    }

    def _load_df(self, input_data: dict[str, Any]) -> pd.DataFrame:
        file_path = input_data.get("file_path", "")
        csv_data = input_data.get("csv_data", "")

        if csv_data:
            return pd.read_csv(io.StringIO(csv_data))
        if file_path:
            if file_path.endswith(".json"):
                return pd.read_json(file_path)
            return pd.read_csv(file_path)
        raise ValueError("Provide 'file_path' or 'csv_data'")

    async def execute(self, input_data: dict[str, Any]) -> Any:
        action = input_data.get("action", "")

        try:
            df = self._load_df(input_data)

            if action in ("read_csv", "read_json"):
                n = int(input_data.get("n_rows", 10))
                return {
                    "columns": list(df.columns),
                    "shape": list(df.shape),
                    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                    "head": df.head(n).to_dict(orient="records"),
                }

            if action == "describe":
                desc = df.describe(include="all").fillna("").to_dict()
                return {
                    "columns": list(df.columns),
                    "shape": list(df.shape),
                    "statistics": desc,
                }

            if action == "query_df":
                query = input_data.get("query", "")
                if not query:
                    return {"error": "'query' is required"}
                result = df.query(query)
                return {
                    "shape": list(result.shape),
                    "data": result.head(100).to_dict(orient="records"),
                    "total_matches": len(result),
                }

            if action == "head":
                n = int(input_data.get("n_rows", 10))
                return {"data": df.head(n).to_dict(orient="records"), "shape": list(df.shape)}

            if action == "value_counts":
                column = input_data.get("column", "")
                if not column or column not in df.columns:
                    return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}
                vc = df[column].value_counts().head(20)
                return {"column": column, "counts": vc.to_dict()}

            return {"error": f"Unknown action '{action}'"}

        except Exception as exc:
            logger.error("pandas_error", action=action, error=str(exc))
            return {"error": str(exc)}
