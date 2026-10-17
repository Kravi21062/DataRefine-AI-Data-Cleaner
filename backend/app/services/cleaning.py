from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .profiling import compute_quality_score


def fill_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    out = df.copy()
    if strategy == "drop":
        return out.dropna()

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in out.columns if col not in numeric_cols]

    if strategy == "mean":
        for col in numeric_cols:
            if out[col].isna().any():
                out[col] = out[col].fillna(out[col].mean())
        for col in non_numeric_cols:
            if out[col].isna().any():
                mode = out[col].mode(dropna=True)
                out[col] = out[col].fillna(mode.iloc[0] if not mode.empty else "unknown")
    elif strategy == "median":
        for col in numeric_cols:
            if out[col].isna().any():
                out[col] = out[col].fillna(out[col].median())
        for col in non_numeric_cols:
            if out[col].isna().any():
                mode = out[col].mode(dropna=True)
                out[col] = out[col].fillna(mode.iloc[0] if not mode.empty else "unknown")
    elif strategy == "mode":
        for col in out.columns:
            if out[col].isna().any():
                mode = out[col].mode(dropna=True)
                if not mode.empty:
                    out[col] = out[col].fillna(mode.iloc[0])
                else:
                    out[col] = out[col].fillna("unknown")
    elif strategy == "ffill":
        out = out.ffill().bfill()
    elif strategy == "bfill":
        out = out.bfill().ffill()
    else:
        raise ValueError(f"Unsupported missing-value strategy: {strategy}")
    return out


def drop_duplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    if subset:
        return df.drop_duplicates(subset=subset)
    return df.drop_duplicates()


def detect_outliers(df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> pd.Series:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.Series(False, index=df.index)

    flags = []
    for col in numeric_df.columns:
        s = numeric_df[col].dropna()
        if s.empty or s.std(ddof=0) == 0:
            continue
        if method.lower() == "zscore":
            z = (s - s.mean()) / s.std(ddof=0)
            flags.append((np.abs(z) > threshold).reindex(s.index, fill_value=False))
        else:
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            flags.append(((s < lower) | (s > upper)).reindex(s.index, fill_value=False))

    if not flags:
        return pd.Series(False, index=df.index)
    combined = pd.concat(flags, axis=1).fillna(False)
    return combined.any(axis=1).reindex(df.index, fill_value=False)


def convert_column_type(df: pd.DataFrame, column: str, target_type: str) -> pd.DataFrame:
    out = df.copy()
    if column not in out.columns:
        raise ValueError(f"Column '{column}' not found")

    if target_type == "numeric":
        out[column] = pd.to_numeric(out[column], errors="coerce")
    elif target_type == "string":
        out[column] = out[column].astype(str)
    elif target_type == "datetime":
        out[column] = pd.to_datetime(out[column], errors="coerce")
    elif target_type == "category":
        out[column] = out[column].astype("category")
    else:
        raise ValueError(f"Unsupported target type: {target_type}")
    return out


def standardize_text_columns(df: pd.DataFrame, columns: Optional[List[str]] = None, lowercase: bool = True, strip: bool = True) -> pd.DataFrame:
    out = df.copy()
    columns = columns or [col for col in out.columns if out[col].dtype == object]
    for col in columns:
        if col not in out.columns:
            continue
        series = out[col].astype(str)
        if strip:
            series = series.str.strip()
        if lowercase:
            series = series.str.lower()
        out[col] = series
    return out


def replace_invalid_values(df: pd.DataFrame, invalid_tokens: Optional[List[str]] = None, treat_negative_as_invalid: bool = False) -> pd.DataFrame:
    out = df.copy()
    invalid_literals = invalid_tokens or ["na", "n/a", "none", "null", "nan", "-"]
    lower_tokens = {str(i).lower() for i in invalid_literals}

    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].apply(lambda val: np.nan if pd.notna(val) and str(val).strip().lower() in lower_tokens else val)

    if treat_negative_as_invalid:
        for col in out.select_dtypes(include=[np.number]).columns:
            out.loc[out[col] < 0, col] = np.nan
    return out


def apply_cleaning_operation(df: pd.DataFrame, operation: str, **kwargs) -> pd.DataFrame:
    if operation == "fill_missing":
        return fill_missing_values(df, strategy=kwargs.get("strategy", "mean"))
    if operation == "drop_duplicates":
        return drop_duplicate_rows(df, subset=kwargs.get("subset"))
    if operation == "remove_outliers":
        flags = detect_outliers(df, method=kwargs.get("method", "iqr"), threshold=float(kwargs.get("threshold", 1.5)))
        return df[~flags].copy()
    if operation == "convert_type":
        return convert_column_type(df, kwargs["column"], kwargs.get("target_type", "numeric"))
    if operation == "standardize_text":
        return standardize_text_columns(
            df,
            columns=kwargs.get("columns"),
            lowercase=bool(kwargs.get("lowercase", True)),
            strip=bool(kwargs.get("strip", True)),
        )
    if operation == "replace_invalid":
        return replace_invalid_values(df, invalid_tokens=kwargs.get("invalid_tokens"), treat_negative_as_invalid=bool(kwargs.get("treat_negative_as_invalid", False)))
    raise ValueError(f"Unsupported operation: {operation}")


def summarize_differences(before_df: pd.DataFrame, after_df: pd.DataFrame) -> Dict[str, Any]:
    before = profile_dataset(before_df)
    after = profile_dataset(after_df)
    return {
        "rows_before": before["rows"],
        "rows_after": after["rows"],
        "cols_before": before["cols"],
        "cols_after": after["cols"],
        "missing_before": before["missing"],
        "missing_after": after["missing"],
        "duplicates_before": before["duplicates"],
        "duplicates_after": after["duplicates"],
        "quality_before": before["quality_score"],
        "quality_after": after["quality_score"],
        "quality_delta": round(after["quality_score"] - before["quality_score"], 2),
    }
