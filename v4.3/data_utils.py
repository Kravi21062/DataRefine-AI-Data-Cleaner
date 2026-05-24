import os
from io import BytesIO
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder

import json
import urllib.request
import urllib.error
from urllib.parse import urlparse
import socket as _socket

# ollama settings
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3")

# file size
def get_file_size_mb(file) -> float:
    return round(file.size / (1024 * 1024), 2)

def load_dataset(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    if uploaded_file is None:
        return None, None

    file_size_mb = get_file_size_mb(uploaded_file)
    if file_size_mb > 200:
        raise ValueError("File size exceeds 200 MB limit. Please upload a smaller file.")

    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        file_type = "CSV"
    elif filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
        file_type = "Excel"
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

    file_info = {
        "name": uploaded_file.name,
        "size_mb": file_size_mb,
        "type": file_type,
    }
    return df, file_info

def compute_basic_metrics(df: pd.DataFrame) -> Dict[str, int]:
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }

def build_dtype_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.notnull().sum(),
            "Missing Count": df.isnull().sum(),
            "Missing %": (df.isnull().mean() * 100).round(2),
            "Unique Values": df.nunique(dropna=True),
        }
    )
    return summary.reset_index().rename(columns={"index": "Column"})

# data quality score
def compute_quality_score(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0

    n_rows = len(df)
    missing_ratio = df.isnull().sum().sum() / (n_rows * max(df.shape[1], 1))
    duplicate_ratio = df.duplicated().sum() / max(n_rows, 1)

    numeric_df = df.select_dtypes(include=[np.number])
    outlier_ratio = 0.0
    if not numeric_df.empty:
        outlier_flags = []
        for col in numeric_df.columns:
            q1 = numeric_df[col].quantile(0.25)
            q3 = numeric_df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0: continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            flags = ~numeric_df[col].between(lower, upper)
            outlier_flags.append(flags)
        if outlier_flags:
            combined = pd.concat(outlier_flags, axis=1).any(axis=1)
            outlier_ratio = combined.mean()

    score = 100.0
    score -= min(missing_ratio, 1.0) * 40.0
    score -= min(duplicate_ratio, 1.0) * 30.0
    score -= min(outlier_ratio, 1.0) * 30.0

    return float(max(0.0, min(100.0, score)))

# handle missing values using different methods
def handle_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    df = df.copy()
    if strategy == "Drop":
        return df.dropna()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if strategy == "Mean":
        if numeric_cols: df[numeric_cols] = SimpleImputer(strategy="mean").fit_transform(df[numeric_cols])
        if cat_cols: df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    elif strategy == "Median":
        if numeric_cols: df[numeric_cols] = SimpleImputer(strategy="median").fit_transform(df[numeric_cols])
        if cat_cols: df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    elif strategy == "Mode":
        if numeric_cols: df[numeric_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[numeric_cols])
        if cat_cols: df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    elif strategy == "KNN Imputation":
        if numeric_cols:
            valid_numeric = [c for c in numeric_cols if not df[c].isna().all()]
            if valid_numeric:
                try:
                    imputed = KNNImputer(n_neighbors=5).fit_transform(df[valid_numeric])
                    df[valid_numeric] = imputed
                except Exception as e:
                    st.error(f"KNN Imputation Failed: Ensure numeric columns don't contain text data. Details: {e}")
        if cat_cols:
            df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    return df

# Outliers IQR Method-1
def detect_outliers_iqr(df: pd.DataFrame) -> pd.Series:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.Series(False, index=df.index)

    outlier_flags = []
    for col in numeric_df.columns:
        q1, q3 = numeric_df[col].quantile(0.25), numeric_df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0: continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flags = ~numeric_df[col].between(lower, upper)
        outlier_flags.append(flags)

    if not outlier_flags:
        return pd.Series(False, index=df.index)
    return pd.concat(outlier_flags, axis=1).any(axis=1)

# Outliers Z-SCORE Method-2
def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty: return pd.Series(False, index=df.index)

    non_constant = [c for c in numeric_df.columns if numeric_df[c].std(ddof=0) > 0]
    if not non_constant: return pd.Series(False, index=df.index)

    numeric_df = numeric_df[non_constant]
    zscores = (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0)
    flags = (zscores.abs() > threshold).any(axis=1)
    return flags.reindex(df.index, fill_value=False)

# Columns Operations
def apply_column_operation(df: pd.DataFrame, operation: str, column: str, **kwargs) -> pd.DataFrame:
    df = df.copy()
    if column not in df.columns: return df

    if operation == "Rename column":
        new_name = kwargs.get("new_name")
        if new_name and new_name not in df.columns:
            df = df.rename(columns={column: new_name})
    elif operation == "Change datatype":
        target_type = kwargs.get("target_type")
        if target_type == "int": df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
        elif target_type == "float": df[column] = pd.to_numeric(df[column], errors="coerce")
        elif target_type == "string": df[column] = df[column].astype("string")
        elif target_type == "datetime": df[column] = pd.to_datetime(df[column], errors="coerce")
    elif operation == "Drop column":
        df = df.drop(columns=[column])
    elif operation == "Encode categorical":
        if df[column].dtype == "O" or df[column].dtype.name == "category":
            df = encode_categorical(df, column, method="Label")
        else:
            df[column + "_encoded"] = pd.factorize(df[column])[0]
    return df

# Clean Column Names replace spaces with underscores
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {col: str(col).strip().replace(" ", "_").replace("-", "_").replace("/", "_").lower() for col in df.columns}
    return df.copy().rename(columns=new_cols)

def standardize_text_columns(df: pd.DataFrame, columns: Optional[List[str]] = None, lowercase: bool = True, strip: bool = True) -> pd.DataFrame:
    df = df.copy()
    columns = columns or df.select_dtypes(include=["object"]).columns.tolist()
    for col in columns:
        if col not in df.columns: continue
        mask = df[col].notna()
        series = df[col].copy()
        if strip: series[mask] = series[mask].astype(str).str.strip()
        if lowercase: series[mask] = series[mask].astype(str).str.lower()
        df[col] = series
    return df

# replace_invalid_values
def replace_invalid_values(df: pd.DataFrame, columns: Optional[List[str]] = None, invalid_tokens: Optional[List[str]] = None, replace_with=np.nan, treat_negative_as_invalid: bool = False) -> pd.DataFrame:
    df = df.copy()
    columns = columns or df.columns.tolist()
    invalid_tokens = invalid_tokens or ["na", "n/a", "none", "-", "--", "null", "nan"]
    lower_tokens = {str(t).lower() for t in invalid_tokens}

    for col in columns:
        if col not in df.columns: continue
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: replace_with if pd.notna(x) and str(x).lower() in lower_tokens else x)

    if treat_negative_as_invalid:
        for col in df.select_dtypes(include=[np.number]).columns:
            df.loc[df[col] < 0, col] = replace_with
    return df


# Encode Categorical Using methods Label and One-Hot
def encode_categorical(df: pd.DataFrame, column: str, method: str = "Label") -> pd.DataFrame:
    df = df.copy()
    if column not in df.columns: return df

    if method == "One-Hot":
        dummies = pd.get_dummies(df[column], prefix=column, dummy_na=False)
        df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
    else:
        encoded_series = df[column].astype("string").fillna("NA").astype(str)
        df[column + "_encoded"] = LabelEncoder().fit_transform(encoded_series)
    return df

# Scale Numeric using Min-Max method
def scale_numeric(df: pd.DataFrame, method: str = "Min-Max") -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        if col.endswith("_scaled") or col.endswith("_std"): continue
        series = df[col].astype(float)
        if method == "Standard":
            std = series.std(ddof=0)
            if std != 0: df[f"{col}_std"] = (series - series.mean()) / std
        else:
            min_v, max_v = series.min(), series.max()
            if max_v != min_v: df[f"{col}_scaled"] = (series - min_v) / (max_v - min_v)
    return df

# Build dataset Profile
def _build_dataset_profile(df: pd.DataFrame) -> str:
    lines = [f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns\n\nColumn details:"]
    for col in df.columns:
        missing_pct, unique_count, dtype = df[col].isnull().mean() * 100, df[col].nunique(dropna=True), str(df[col].dtype)
        extra = ""
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            series = df[col].dropna()
            extra = f", skewness={float(series.skew()):.2f}, has_negatives={bool((series < 0).any())}" if len(series) > 2 else ""
        lines.append(f"  - '{col}': dtype={dtype}, missing={missing_pct:.1f}%, unique_values={unique_count}{extra}")
    lines.append(f"\nDuplicate rows: {int(df.duplicated().sum())}")
    return "\n".join(lines)


# call ollama phi3 model
def _call_ollama_phi3(profile: str) -> Optional[List[Dict[str, str]]]:
    prompt = f"""You are a data quality expert. Analyze the dataset profile below.
{profile}
Rules:
- Return ONLY a valid JSON array (no markdown).
- Keep each field ULTRA SHORT (max 3-5 words).
- Use action keywords ONLY (e.g., "Impute Median", "Drop Column", "Group Categories"). Do not write full sentences.
Each element must have exactly these keys: "Column", "Issue Detected", "Suggestion", "Severity" ("High", "Medium", "Low", "Info")
JSON array:"""

    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "num_predict": 1024}}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    try:
        parsed = urlparse(OLLAMA_URL)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        test = _socket.create_connection((host, port), timeout=3)
        test.close()
    except Exception:
        return None  

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw_text = json.loads(resp.read().decode("utf-8")).get("response", "").strip()
        start, end = raw_text.find("["), raw_text.rfind("]") + 1
        if start == -1 or end == 0: return None

        valid = []
        for item in json.loads(raw_text[start:end]):
            if isinstance(item, dict) and all(k in item for k in ("Column", "Issue Detected", "Suggestion", "Severity")):
                item["Severity"] = item["Severity"].strip().capitalize()
                if item["Severity"] not in ("High", "Medium", "Low", "Info"): item["Severity"] = "Medium"
                valid.append(item)
        return valid if valid else None
    except Exception:
        return None

# Rule  based suggestions
def _rule_based_suggestions(df: pd.DataFrame) -> List[Dict[str, str]]:
    suggestions = []
    total_rows = len(df)
    duplicate_count = df.duplicated().sum()
    
    if duplicate_count > 0:
        suggestions.append({"Column": "-", "Issue Detected": f"Duplicates ({duplicate_count})", "Suggestion": "Remove duplicate rows", "Severity": "High" if duplicate_count / total_rows > 0.1 else "Medium"})

    for col in df.columns:
        series, dtype = df[col], df[col].dtype
        missing_pct, unique_count = series.isnull().mean() * 100, series.nunique(dropna=True)

        if missing_pct > 50: suggestions.append({"Column": col, "Issue Detected": f"High missing ({missing_pct:.1f}%)", "Suggestion": "Drop column or use KNN imputation", "Severity": "High"})
        elif 0 < missing_pct <= 50: suggestions.append({"Column": col, "Issue Detected": f"Missing ({missing_pct:.1f}%)", "Suggestion": "Impute mean/median/mode", "Severity": "Medium"})

        if unique_count == 1: suggestions.append({"Column": col, "Issue Detected": "Constant values", "Suggestion": "Drop column", "Severity": "Low"})

        if pd.api.types.is_object_dtype(dtype) and unique_count > 100: suggestions.append({"Column": col, "Issue Detected": f"High cardinality ({unique_count})", "Suggestion": "Use encoding or grouping", "Severity": "Medium"})

        if pd.api.types.is_numeric_dtype(dtype):
            outlier_pct = (detect_outliers_iqr(df[[col]]).sum() / total_rows) * 100
            if outlier_pct > 0: suggestions.append({"Column": col, "Issue Detected": f"Outliers ({outlier_pct:.1f}%)", "Suggestion": "Cap or remove using IQR", "Severity": "High" if outlier_pct > 10 else "Medium"})
            
            skew = series.dropna().skew()
            if abs(skew) > 1: suggestions.append({"Column": col, "Issue Detected": f"Skewed ({skew:.2f})", "Suggestion": "Apply log transform", "Severity": "Medium"})

    if not suggestions: suggestions.append({"Column": "-", "Issue Detected": "No major issues", "Suggestion": "Dataset looks clean", "Severity": "Info"})
    return suggestions

# Generate cleaning suggestions
def generate_cleaning_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame(columns=["Column", "Issue Detected", "Suggestion", "Severity"])
    
    suggestions = _call_ollama_phi3(_build_dataset_profile(df))
    source = "AI (phi3)" if suggestions else "Rule-based (Ollama unavailable)"
    
    result_df = pd.DataFrame(suggestions or _rule_based_suggestions(df))
    result_df["Source"] = source
    return result_df

# Build before vs after comparison
def build_before_after_comparison(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    mb, ma = compute_basic_metrics(raw_df), compute_basic_metrics(clean_df)
    _pct = lambda n, d: round(n / max(d, 1) * 100, 2)

    comparison = pd.DataFrame({
        "Metric": ["Rows", "Columns", "Missing Values", "Duplicate Rows", "Missing % of Cells", "Duplicate % of Rows"],
        "Before": [mb["rows"], mb["cols"], mb["missing"], mb["duplicates"], _pct(mb["missing"], mb["rows"] * mb["cols"]), _pct(mb["duplicates"], mb["rows"])],
        "After": [ma["rows"], ma["cols"], ma["missing"], ma["duplicates"], _pct(ma["missing"], ma["rows"] * ma["cols"]), _pct(ma["duplicates"], ma["rows"])],
    })
    return comparison, mb, ma

_ID_KEYWORDS = {"id", "key", "code", "index", "tract", "zip", "postal", "phone", "fax", "ssn", "uuid", "guid"}
def _is_id_col(name: str) -> bool: return any(kw in name.lower() for kw in _ID_KEYWORDS)

# top 4 insights find
# distribution
def _analyze_distribution(df: pd.DataFrame) -> List[Dict[str, Any]]:
    insights = []
    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        if _is_id_col(col): continue
        series = numeric_df[col].dropna()
        if series.empty or series.var() == 0: continue
        skew, var = float(series.skew()), float(series.var())
        insights.append({
            "type": "distribution", "title": f"Distribution of {col} ({'right' if skew > 0 else 'left'}-skewed)",
            "description": f"Column `{col}` shows a {'right' if skew > 0 else 'left'}-skewed distribution indicating potential outliers.",
            "score": abs(skew) * 0.5 + np.log1p(var), "columns": [col]
        })
    return insights

# correlation
def _analyze_correlation(df: pd.DataFrame) -> List[Dict[str, Any]]:
    insights = []
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2: return insights

    corr = numeric_df.corr().replace([np.inf, -np.inf], np.nan).dropna(how="all").dropna(axis=1, how="all")
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = float(corr.iloc[i, j])
            if abs(c) >= 0.4:
                insights.append({
                    "type": "correlation", "title": f"{cols[i]} and {cols[j]} are {'positively' if c > 0 else 'negatively'} correlated",
                    "description": f"Features `{cols[i]}` and `{cols[j]}` have a correlation of {c:.2f}.",
                    "score": abs(c) * 2.0, "columns": [cols[i], cols[j]]
                })
    return insights

# categories
def _analyze_categories(df: pd.DataFrame) -> List[Dict[str, Any]]:
    insights = []
    for col in df.select_dtypes(exclude=[np.number]).columns:
        counts = df[col].astype(str).value_counts(dropna=True)
        if counts.empty: continue
        top_share, contrib = float(counts.iloc[0]) / float(counts.sum()), float(counts.head(5).sum() / counts.sum())
        if 0.1 <= top_share <= 0.95:
            insights.append({
                "type": "category", "title": f"Top categories in `{col}`",
                "description": f"Column `{col}` is dominated by `{counts.index[0]}` ({top_share*100:.1f}%).",
                "score": contrib * 0.8, "columns": [col]
            })
    return insights

# trends
def _analyze_trends(df: pd.DataFrame) -> List[Dict[str, Any]]:
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or any(k in str(c).lower() for k in ["date", "time"])]
    if not date_cols: return []

    date_col = date_cols[0]
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col]).sort_values(date_col)

    numeric_df = tmp.select_dtypes(include=[np.number])
    if numeric_df.empty: return []
    
    target_col = numeric_df.var().sort_values(ascending=False).index[0]
    
    try: daily = tmp.set_index(date_col)[target_col].resample("D").mean().dropna()
    except Exception: return []

    if len(daily) < 3 or daily.values.std() == 0: return []

    corr = np.corrcoef(np.arange(len(daily)), daily.values)[0, 1]
    if abs(float(corr)) >= 0.3:
        return [{
            "type": "trend", "title": f"Trend of `{target_col}` over `{date_col}`",
            "description": f"Average value of `{target_col}` appears to be {'increasing' if corr > 0 else 'decreasing'}.",
            "score": abs(float(corr)) * 2.0, "columns": [date_col, target_col]
        }]
    return []

# Ai insight analsis
def ai_insight_analysis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty: return []
    all_insights = _analyze_distribution(df) + _analyze_correlation(df) + _analyze_categories(df) + _analyze_trends(df)

    seen_types, diverse = {}, []
    for insight in sorted(all_insights, key=lambda x: x.get("score", 0.0), reverse=True):
        t = insight["type"]
        if seen_types.get(t, 0) < 2:
            diverse.append(insight)
            seen_types[t] = seen_types.get(t, 0) + 1
        if len(diverse) == 4: break
    return diverse