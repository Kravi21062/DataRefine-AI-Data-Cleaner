from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from app.core.session_store import get_active_df, get_history, get_session, set_clean_df
from app.services.cleaning import apply_cleaning_operation
from app.services.ai_service import generate_recommendations
from app.services.export_service import generate_pdf_report
from app.services.profiling import compute_quality_score, profile_dataset, summarize_differences


def dataset_summary(session_id: str) -> Dict[str, Any]:
    session = get_session(session_id)
    if session is None:
        raise KeyError("session not found")
    df = get_active_df(session_id)
    if df is None:
        return {"rows": 0, "cols": 0, "missing": 0, "duplicates": 0, "quality_score": 0.0, "columns": []}
    return profile_dataset(df)


def dataset_preview(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    session = get_session(session_id)
    if session is None:
        raise KeyError("session not found")
    df = get_active_df(session_id)
    if df is None:
        return []
    return df.head(limit).to_dict(orient="records")


def apply_operation(session_id: str, operation: str, **kwargs) -> Dict[str, Any]:
    session = get_session(session_id)
    if session is None:
        raise KeyError("session not found")
    before_df = get_active_df(session_id).copy()
    after_df = apply_cleaning_operation(before_df, operation, **kwargs)
    set_clean_df(session_id, after_df)
    diff = summarize_differences(before_df, after_df)
    session["history"].append({
        "label": operation,
        "meta": kwargs,
        "rows_before": diff["rows_before"],
        "rows_after": diff["rows_after"],
    })
    return diff


def generate_recommendation_payload(session_id: str) -> Dict[str, Any]:
    session = get_session(session_id)
    if session is None:
        raise KeyError("session not found")
    df = get_active_df(session_id)
    if df is None:
        return {"source": "rule_based", "items": []}
    return generate_recommendations(df)


def export_csv_bytes(session_id: str) -> bytes:
    df = get_active_df(session_id)
    if df is None:
        raise ValueError("No dataset loaded")
    return df.to_csv(index=False).encode("utf-8")


def export_xlsx_bytes(session_id: str) -> bytes:
    df = get_active_df(session_id)
    if df is None:
        raise ValueError("No dataset loaded")
    from io import BytesIO
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buffer.getvalue()


def export_pdf_bytes(session_id: str) -> bytes:
    session = get_session(session_id)
    if session is None:
        raise KeyError("session not found")
    raw_df = session["raw_df"]
    clean_df = session["clean_df"]
    return generate_pdf_report(raw_df, clean_df, session["history"], session.get("dataset_name", "dataset.csv"))
