from io import BytesIO
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder

import json
import urllib.request
import urllib.error
import socket as _socket



# File helpers
# ---------------------------------------------------------------------------
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

# Basic metrics
# ---------------------------------------------------------------------------
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

#  data quality score based on penalty for missing values(40), duplicates(30), outliers(30) 
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
            if iqr == 0:
                continue
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



# Cleaning operations
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    df = df.copy()

    if strategy == "Drop":
        return df.dropna()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if strategy == "Mean":
        if numeric_cols:
            df[numeric_cols] = SimpleImputer(strategy="mean").fit_transform(df[numeric_cols])
        if cat_cols:
            df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    elif strategy == "Median":
        if numeric_cols:
            df[numeric_cols] = SimpleImputer(strategy="median").fit_transform(df[numeric_cols])
        if cat_cols:
            df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    elif strategy == "Mode":
        if numeric_cols:
            df[numeric_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[numeric_cols])
        if cat_cols:
            df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    # ML-based imputation using k-nearest neighbours
    elif strategy == "KNN Imputation":
        if numeric_cols:
            imputed = KNNImputer(n_neighbors=5).fit_transform(df[numeric_cols])
            df[numeric_cols] = imputed
        if cat_cols:
            df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    return df


def detect_outliers_iqr(df: pd.DataFrame) -> pd.Series:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.Series(False, index=df.index)

    outlier_flags = []
    for col in numeric_df.columns:
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flags = ~numeric_df[col].between(lower, upper)
        outlier_flags.append(flags)

    if not outlier_flags:
        return pd.Series(False, index=df.index)
    return pd.concat(outlier_flags, axis=1).any(axis=1)


def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.Series(False, index=df.index)

    zscores = (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0)
    flags = (zscores.abs() > threshold).any(axis=1)
    return flags.reindex(df.index, fill_value=False)


def detect_outliers_isolation_forest(
    df: pd.DataFrame, contamination: float = 0.05
) -> Tuple[pd.Series, Dict[str, float]]:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.Series(False, index=df.index), {"n_anomalies": 0, "anomaly_ratio": 0.0}

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    preds = model.fit_predict(numeric_df)
    flags = pd.Series(preds == -1, index=df.index)

    n_anomalies = int(flags.sum())
    return flags, {"n_anomalies": n_anomalies, "anomaly_ratio": float(flags.mean())}

# columns operations: rename, change dtype, drop, encode categorical
def apply_column_operation(
    df: pd.DataFrame, operation: str, column: str, **kwargs
) -> pd.DataFrame:
    df = df.copy()
    if column not in df.columns:
        return df

    if operation == "Rename column":
        new_name = kwargs.get("new_name")
        if new_name and new_name not in df.columns:
            df = df.rename(columns={column: new_name})

    elif operation == "Change datatype":
        target_type = kwargs.get("target_type")
        if target_type == "int":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
        elif target_type == "float":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif target_type == "string":
            df[column] = df[column].astype(str)
        elif target_type == "datetime":
            df[column] = pd.to_datetime(df[column], errors="coerce")

    elif operation == "Drop column":
        df = df.drop(columns=[column])

    elif operation == "Encode categorical":
        if df[column].dtype == "O" or df[column].dtype.name == "category":
            encoder = LabelEncoder()
            series = df[column].astype(str).fillna("NA")
            df[column + "_encoded"] = encoder.fit_transform(series)
        else:
            df[column + "_encoded"] = pd.factorize(df[column])[0]

    return df

# Standardize column names
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = {}
    for col in df.columns:
        new_name = (
            str(col).strip().replace(" ", "_").replace("-", "_").replace("/", "_").lower()
        )
        new_cols[col] = new_name
    return df.rename(columns=new_cols)

# Text cleaning
def standardize_text_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lowercase: bool = True,
    strip: bool = True,
) -> pd.DataFrame:

    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue
        
        mask = df[col].notna()
        series = df[col].copy()
        if strip:
            series[mask] = series[mask].astype(str).str.strip()
        if lowercase:
            series[mask] = series[mask].astype(str).str.lower()
        df[col] = series
    return df

# Replace common invalid values(na, n/a, none, -, --, null) with NaN and also remove negative values
def replace_invalid_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    invalid_tokens: Optional[List[str]] = None,
    replace_with=np.nan,
    treat_negative_as_invalid: bool = False,
) -> pd.DataFrame:
    
    df = df.copy()

    if columns is None:
        columns = df.columns.tolist()

    if invalid_tokens is None:
        invalid_tokens = ["na", "n/a", "none", "-", "--", "null", "nan"]

    lower_tokens = {str(t).lower() for t in invalid_tokens}

    for col in columns:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: replace_with if pd.notna(x) and str(x).lower() in lower_tokens else x
            )

    if treat_negative_as_invalid:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df.loc[df[col] < 0, col] = replace_with

    return df

# Category fix like M - male, m - male, F - Female, f - female
def standardize_categories(
    df: pd.DataFrame, column: str, mapping: Dict[str, str]
) -> pd.DataFrame:
    df = df.copy()
    if column not in df.columns:
        return df

    def _map_value(x):
        if pd.isna(x):
            return x
        key = str(x).strip().lower()
        for src, dst in mapping.items():
            if key == str(src).strip().lower():
                return dst
        return x

    df[column] = df[column].apply(_map_value)
    return df

# Encode a single categorical column using label or one-hot encoding
def encode_categorical(
    df: pd.DataFrame, column: str, method: str = "Label"
) -> pd.DataFrame:
    df = df.copy()
    if column not in df.columns:
        return df

    if method == "One-Hot":
        dummies = pd.get_dummies(df[column], prefix=column, dummy_na=False)
        df = df.drop(columns=[column])
        df = pd.concat([df, dummies], axis=1)
    else:
        encoder = LabelEncoder()
        series = df[column].astype(str).fillna("NA")
        df[column + "_encoded"] = encoder.fit_transform(series)
    return df

# Validate and optionally fix numeric ranges. mode: 'cap' – clip values, 'nan' – set out-of-range to NaN
def validate_range(
    df: pd.DataFrame,
    column: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    mode: str = "cap",
) -> pd.DataFrame:

    df = df.copy()
    if column not in df.columns:
        return df

    if not pd.api.types.is_numeric_dtype(df[column].dtype):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    series = df[column]

    if min_value is not None:
        series = series.mask(series < min_value, np.nan) if mode == "nan" else series.clip(lower=min_value)

    if max_value is not None:
        series = series.mask(series > max_value, np.nan) if mode == "nan" else series.clip(upper=max_value)

    df[column] = series
    return df

# Scale numeric columns using Min-Max or Standard (z-score) scaling
def scale_numeric(df: pd.DataFrame, method: str = "Min-Max") -> pd.DataFrame:

    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return df

    for col in numeric_cols:
        if col.endswith("_scaled") or col.endswith("_std"):
            continue

        series = df[col].astype(float)

        if method == "Standard":
            mean = series.mean()
            std = series.std(ddof=0)
            if std == 0:
                continue
            scaled = (series - mean) / std
            new_name = f"{col}_std"
        else:
            min_v = series.min()
            max_v = series.max()
            if max_v == min_v:
                continue
            scaled = (series - min_v) / (max_v - min_v)
            new_name = f"{col}_scaled"

        df[new_name] = scaled

    return df

# column type detection heuristic: numeric, categorical, date, text
def detect_feature_types(df: pd.DataFrame) -> pd.DataFrame:
   
    rows = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            inferred = "Numeric"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            inferred = "Date"
        else:
            unique = df[col].nunique(dropna=True)
            n_rows = max(len(df), 1)
            
            threshold = max(10, min(100, int(n_rows * 0.05)))
            inferred = "Categorical" if unique <= threshold else "Text"

        rows.append(
            {
                "Column": col,
                "Pandas Dtype": str(dtype),
                "Inferred Type": inferred,
                "Unique Values": df[col].nunique(dropna=True),
            }
        )
    return pd.DataFrame(rows)


# Cleaning suggestions  — AI-powered Ollama phi3
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"

# Build a dataset summary text profile for AI analysis, including shape, column types, missingness, uniqueness, and basic numeric stats.
def _build_dataset_profile(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append("")
    lines.append("Column details:")

    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        unique_count = df[col].nunique(dropna=True)
        dtype = str(df[col].dtype)

        if pd.api.types.is_numeric_dtype(df[col].dtype):
            series = df[col].dropna()
            skew = float(series.skew()) if len(series) > 2 else 0.0
            has_negatives = bool((series < 0).any())
            extra = f", skewness={skew:.2f}, has_negatives={has_negatives}"
        else:
            extra = ""

        lines.append(
            f"  - '{col}': dtype={dtype}, missing={missing_pct:.1f}%, "
            f"unique_values={unique_count}{extra}"
        )

    dup_count = int(df.duplicated().sum())
    lines.append(f"\nDuplicate rows: {dup_count}")

    return "\n".join(lines)

# call ollama phi3 model and return JSON suggestions list, model off None
def _call_ollama_phi3(profile: str) -> Optional[List[Dict[str, str]]]:
    prompt = f"""You are a data quality expert. Analyze the following dataset profile and generate actionable data cleaning suggestions.

{profile}

Return ONLY a valid JSON array (no markdown, no explanation). Each element must have exactly these keys:
- "Column": column name (use "-" for dataset-level issues)
- "Issue Detected": brief description of the issue
- "Suggestion": specific, actionable cleaning recommendation
- "Severity": one of "High", "Medium", "Low", or "Info"

Focus on: missing values, outliers, high cardinality, low cardinality numeric columns, skewness, duplicate rows, and data type mismatches.
Return at most 10 suggestions. If the dataset looks clean, return a single Info-level entry.

JSON array:"""

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1024,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Quick pre-check: is Ollama port open (max 3 sec wait, no hanging)
    try:
        test = _socket.create_connection(("127.0.0.1", 11434), timeout=3)
        test.close()
    except Exception:
        return None  # Ollama not running — skip straight to fallback

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            raw_text = body.get("response", "").strip()

        # Extract JSON array from response (phi3 may wrap with extra text)
        start = raw_text.find("[")
        end = raw_text.rfind("]") + 1
        if start == -1 or end == 0:
            return None

        suggestions = json.loads(raw_text[start:end])

        # Validate structure
        valid = []
        for item in suggestions:
            if isinstance(item, dict) and all(
                k in item for k in ("Column", "Issue Detected", "Suggestion", "Severity")
            ):
                # Normalize Severity
                item["Severity"] = item["Severity"].strip().capitalize()
                if item["Severity"] not in ("High", "Medium", "Low", "Info"):
                    item["Severity"] = "Medium"
                valid.append(item)

        return valid if valid else None

    except Exception:
        return None

# fallback rule-based suggestions uesd when ollama is unavailable
def _rule_based_suggestions(df: pd.DataFrame) -> List[Dict[str, str]]:
    suggestions = []

    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        unique_count = df[col].nunique(dropna=True)
        dtype = df[col].dtype

        if missing_pct > 50:
            suggestions.append({
                "Column": col,
                "Issue Detected": f"High missingness ({missing_pct:.1f}%)",
                "Suggestion": "Consider dropping column or applying KNN imputation.",
                "Severity": "High",
            })
        elif 0 < missing_pct <= 50:
            suggestions.append({
                "Column": col,
                "Issue Detected": f"Moderate missingness ({missing_pct:.1f}%)",
                "Suggestion": "Impute with mean/median (numeric) or mode (categorical).",
                "Severity": "Medium",
            })

        if dtype == "O" and unique_count > 100:
            suggestions.append({
                "Column": col,
                "Issue Detected": f"High cardinality categorical ({unique_count} unique)",
                "Suggestion": "Consider hashing, target encoding, or dimensionality reduction.",
                "Severity": "Medium",
            })

        if pd.api.types.is_numeric_dtype(dtype) and unique_count < 10:
            suggestions.append({
                "Column": col,
                "Issue Detected": "Numeric with low cardinality",
                "Suggestion": "Review if it should be treated as categorical.",
                "Severity": "Low",
            })

    if not suggestions:
        suggestions.append({
            "Column": "-",
            "Issue Detected": "No critical issues detected.",
            "Suggestion": "Dataset appears clean. Validate domain-specific rules.",
            "Severity": "Info",
        })

    return suggestions

# Generate AI-powered data cleaning suggestions using Ollama phi3. Falls back to rule-based suggestions if Ollama is unavailable.
def generate_cleaning_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    
    if df is None or df.empty:
        return pd.DataFrame(columns=["Column", "Issue Detected", "Suggestion", "Severity"])

    profile = _build_dataset_profile(df)
    suggestions = _call_ollama_phi3(profile)

    source = "AI (phi3)"
    if suggestions is None:

        suggestions = _rule_based_suggestions(df)
        source = "Rule-based (Ollama unavailable)"

    result_df = pd.DataFrame(suggestions)
    result_df["Source"] = source
    return result_df


# Before/After comparison
# ---------------------------------------------------------------------------

def build_before_after_comparison(
    raw_df: pd.DataFrame, clean_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:

    metrics_before = compute_basic_metrics(raw_df)
    metrics_after = compute_basic_metrics(clean_df)

    def _pct(numerator, denom):
        return round(numerator / max(denom, 1) * 100, 2)

    comparison = pd.DataFrame(
        {
            "Metric": [
                "Rows",
                "Columns",
                "Missing Values",
                "Duplicate Rows",
                "Missing % of Cells",
                "Duplicate % of Rows",
            ],
            "Before": [
                metrics_before["rows"],
                metrics_before["cols"],
                metrics_before["missing"],
                metrics_before["duplicates"],
                _pct(metrics_before["missing"], metrics_before["rows"] * metrics_before["cols"]),
                _pct(metrics_before["duplicates"], metrics_before["rows"]),
            ],
            "After": [
                metrics_after["rows"],
                metrics_after["cols"],
                metrics_after["missing"],
                metrics_after["duplicates"],
                _pct(metrics_after["missing"], metrics_after["rows"] * metrics_after["cols"]),
                _pct(metrics_after["duplicates"], metrics_after["rows"]),
            ],
        }
    )
    return comparison, metrics_before, metrics_after


# AI Insight Analysis
# ---------------------------------------------------------------------------

_ID_KEYWORDS = {"id", "key", "code", "index", "tract", "zip", "postal",
                "phone", "fax", "ssn", "uuid", "guid"}

def _is_id_col(name: str) -> bool:
    low = name.lower()
    return any(kw in low for kw in _ID_KEYWORDS)

# ckeck skewness and variance
def _analyze_distribution(df: pd.DataFrame) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return insights

    for col in numeric_df.columns:
        if _is_id_col(col):
            continue
        series = numeric_df[col].dropna()
        if series.empty or series.var() == 0:
            continue
        skew = float(series.skew())
        var = float(series.var())
        score = abs(skew) * 0.5 + np.log1p(var)
        direction = "right-skewed" if skew > 0 else "left-skewed"
        insights.append(
            {
                "type": "distribution",
                "title": f"Distribution of {col} ({direction})",
                "description": (
                    f"Column `{col}` shows a {direction} distribution with noticeable variance, "
                    "indicating potential outliers or long tails."
                ),
                "score": score,
                "columns": [col],
            }
        )
    return insights

# correlation detect
def _analyze_correlation(df: pd.DataFrame) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return insights

    corr = (
        numeric_df.corr()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="all")
        .dropna(axis=1, how="all")
    )
    if corr.empty:
        return insights

    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = float(corr.iloc[i, j])
            strength = abs(c)
            if strength < 0.4:
                continue
            col_x, col_y = cols[i], cols[j]
            relation = "positively" if c > 0 else "negatively"
            insights.append(
                {
                    "type": "correlation",
                    "title": f"{col_x} and {col_y} are {relation} correlated",
                    "description": (
                        f"Features `{col_x}` and `{col_y}` have a correlation of {c:.2f}, "
                        "suggesting a meaningful linear relationship."
                    ),
                    "score": strength * 2.0,
                    "columns": [col_x, col_y],
                }
            )
    return insights

# categoey analysis: dominant category, top categories, category imbalance
def _analyze_categories(df: pd.DataFrame) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []
    cat_df = df.select_dtypes(exclude=[np.number])
    if cat_df.empty:
        return insights

    for col in cat_df.columns:
        series = cat_df[col].astype(str)
        counts = series.value_counts(dropna=True)
        if counts.empty:
            continue
        total = float(counts.sum())
        top_value = counts.index[0]
        top_share = float(counts.iloc[0]) / total

        if top_share < 0.1 or top_share > 0.95:
            continue

        top_n = counts.head(5)
        contrib = float(top_n.sum() / total)
        title = f"Category split in `{col}`" if len(counts) <= 2 else f"Top categories in `{col}`"
        insights.append(
            {
                "type": "category",
                "title": title,
                "description": (
                    f"Column `{col}` is dominated by value `{top_value}` "
                    f"({top_share*100:.1f}% of records). "
                    f"Top 5 categories together cover about {contrib*100:.1f}% of the dataset."
                ),
                "score": contrib * 0.8,
                "columns": [col],
            }
        )
    return insights

# teand analysis to detect increasing/decreasing trend over time-based, date, correlation over time
def _analyze_trends(df: pd.DataFrame) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []

    date_cols = [
        col
        for col in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[col])
        or "date" in str(col).lower()
        or "time" in str(col).lower()
    ]
    if not date_cols:
        return insights

    date_col = date_cols[0]
    date_series = pd.to_datetime(df[date_col], errors="coerce")
    if date_series.dropna().empty:
        return insights

    tmp = df.copy()
    tmp[date_col] = date_series
    tmp = tmp.dropna(subset=[date_col])

    numeric_df = tmp.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return insights

    target_col = numeric_df.var().sort_values(ascending=False).index[0]

    daily = tmp.set_index(date_col)[target_col].resample("D").mean().dropna()
    if len(daily) < 3:
        return insights

    x = np.arange(len(daily))
    y = daily.values
    if y.std() == 0:
        return insights

    corr = np.corrcoef(x, y)[0, 1]
    strength = abs(float(corr))
    if strength < 0.3:
        return insights

    direction = "increasing" if corr > 0 else "decreasing"
    insights.append(
        {
            "type": "trend",
            "title": f"Trend of `{target_col}` over `{date_col}`",
            "description": (
                f"Over time, the average value of `{target_col}` appears to be {direction}, "
                f"with a trend strength of about {corr:.2f}."
            ),
            "score": strength * 2.0,
            "columns": [date_col, target_col],
        }
    )
    return insights

# analyze  a cleaned dataset and return the top 4 most informative insights by blending distribution, correlation
def ai_insight_analysis(df: pd.DataFrame) -> List[Dict[str, Any]]:

    if df is None or df.empty:
        return []

    all_insights: List[Dict[str, Any]] = []
    all_insights.extend(_analyze_distribution(df))
    all_insights.extend(_analyze_correlation(df))
    all_insights.extend(_analyze_categories(df))
    all_insights.extend(_analyze_trends(df))

    if not all_insights:
        return []

    ranked = sorted(all_insights, key=lambda x: x.get("score", 0.0), reverse=True)


    seen_types: Dict[str, int] = {}
    diverse: List[Dict[str, Any]] = []
    for insight in ranked:
        t = insight["type"]
        if seen_types.get(t, 0) < 2:
            diverse.append(insight)
            seen_types[t] = seen_types.get(t, 0) + 1
        if len(diverse) == 4:
            break

    return diverse