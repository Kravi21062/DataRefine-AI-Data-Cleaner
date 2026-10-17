import json
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from app.core.config import settings


def _rule_based_recommendations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    rows = len(df)
    if df.duplicated().sum() > 0:
        recs.append({
            "title": "Remove duplicate rows",
            "column": "-",
            "issue": "Duplicate records",
            "recommendation": "drop_duplicates",
            "severity": "high",
            "confidence": 0.95,
            "evidence": f"{int(df.duplicated().sum())} duplicate rows detected",
        })

    for col in df.columns:
        s = df[col]
        missing_pct = float(s.isna().mean() * 100)
        if missing_pct > 20:
            recs.append({
                "title": "Missing values",
                "column": col,
                "issue": f"{missing_pct:.1f}% missing",
                "recommendation": "fill_missing",
                "severity": "high" if missing_pct > 50 else "medium",
                "confidence": 0.88,
                "evidence": f"{int(s.isna().sum())} missing values in column {col}",
            })
        if s.nunique(dropna=True) == 1:
            recs.append({
                "title": "Constant column",
                "column": col,
                "issue": "Low information value",
                "recommendation": "drop_column",
                "severity": "low",
                "confidence": 0.72,
                "evidence": "Column contains a single unique value",
            })
        if pd.api.types.is_numeric_dtype(s) and not s.dropna().empty:
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            if iqr and len(s.dropna()) > 10:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_pct = float(((s < lower) | (s > upper)).mean() * 100)
                if outlier_pct > 5:
                    recs.append({
                        "title": "Potential outliers",
                        "column": col,
                        "issue": f"{outlier_pct:.1f}% outliers",
                        "recommendation": "remove_outliers",
                        "severity": "medium",
                        "confidence": 0.8,
                        "evidence": f"Outlier threshold based on IQR for {col}",
                    })

    if not recs:
        recs.append({
            "title": "Dataset looks healthy",
            "column": "-",
            "issue": "No major quality issues found",
            "recommendation": "no_action",
            "severity": "low",
            "confidence": 0.55,
            "evidence": "Profiling did not identify severe issues",
        })
    return recs


def _call_ollama(profile_text: str) -> Optional[List[Dict[str, Any]]]:
    try:
        payload = {
            "model": settings.ollama_model,
            "prompt": f"You are a data quality assistant. Return JSON with keys title, column, issue, recommendation, severity, confidence, evidence. Data profile:\n{profile_text}",
            "stream": False,
        }
        response = requests.post(settings.ollama_url, json=payload, timeout=30)
        if response.status_code != 200:
            return None
        data = response.json()
        text = (data.get("response") or "").strip()
        if not text:
            return None
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        return None
    except Exception:
        return None


def generate_recommendations(df: pd.DataFrame) -> Dict[str, Any]:
    profile = "\n".join(
        [
            f"{col}: dtype={str(df[col].dtype)}, missing={int(df[col].isna().sum())}, unique={int(df[col].nunique(dropna=True))}"
            for col in df.columns
        ]
    )
    ai = _call_ollama(profile)
    if ai:
        return {"source": "ai", "items": ai}
    return {"source": "rule_based", "items": _rule_based_recommendations(df)}
