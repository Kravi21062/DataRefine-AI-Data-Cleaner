from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"rows": 0, "cols": 0, "missing": 0, "duplicates": 0, "quality_score": 0.0, "columns": []}

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    datetime_cols = [
        col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or str(df[col].dtype).lower().startswith("datetime")
    ]

    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "quality_score": round(compute_quality_score(df), 2),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "columns": [
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "missing_pct": round(float(df[col].isna().mean() * 100), 2),
                "unique": int(df[col].nunique(dropna=True)),
            }
            for col in df.columns
        ],
    }


def compute_quality_score(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0

    rows = max(len(df), 1)
    cols = max(df.shape[1], 1)
    missing_ratio = float(df.isna().sum().sum() / (rows * cols))
    duplicate_ratio = float(df.duplicated().sum() / rows)

    numeric_df = df.select_dtypes(include=[np.number])
    outlier_ratio = 0.0
    if not numeric_df.empty:
        ratios = []
        for col in numeric_df.columns:
            s = numeric_df[col].dropna()
            if s.empty or s.std(ddof=0) == 0:
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            ratios.append(float(((s < lower) | (s > upper)).mean()))
        if ratios:
            outlier_ratio = float(np.mean(ratios))

    score = 100.0
    score -= min(missing_ratio, 1.0) * 45.0
    score -= min(duplicate_ratio, 1.0) * 25.0
    score -= min(outlier_ratio, 1.0) * 30.0
    return max(0.0, min(100.0, score))
