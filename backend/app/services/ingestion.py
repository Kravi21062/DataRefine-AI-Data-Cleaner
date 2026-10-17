from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def load_uploaded_file(file_obj) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if file_obj is None:
        raise ValueError("No file provided")

    filename = getattr(file_obj, "filename", "upload") or "upload"
    name_lower = filename.lower()
    if not (name_lower.endswith(".csv") or name_lower.endswith(".xlsx") or name_lower.endswith(".xls")):
        raise ValueError("Unsupported file type. Upload a CSV or Excel file.")

    data = file_obj.file.read() if hasattr(file_obj, "file") else file_obj.read()
    if not data:
        raise ValueError("Uploaded file is empty.")

    file_like = BytesIO(data)
    if name_lower.endswith(".csv"):
        df = pd.read_csv(file_like)
        file_type = "csv"
    else:
        df = pd.read_excel(file_like)
        file_type = "excel"

    if df.empty:
        raise ValueError("Dataset is empty after parsing.")

    df = df.replace({"\ufeff": ""})
    file_info = {
        "name": filename,
        "type": file_type,
        "size_bytes": len(data),
    }
    return df, file_info


def safe_preview(df: pd.DataFrame, limit: int = 10) -> List[Dict[str, Any]]:
    preview = df.head(limit).copy()
    return preview.to_dict(orient="records")
