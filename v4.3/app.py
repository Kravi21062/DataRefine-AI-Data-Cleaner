import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from theme import PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, BACKGROUND_COLOR
from data_utils import (
    load_dataset,
    compute_basic_metrics,
    build_dtype_summary,
    compute_quality_score,
    handle_missing_values,
    detect_outliers_iqr,
    detect_outliers_zscore,
    apply_column_operation,
    generate_cleaning_suggestions,
    build_before_after_comparison,
    clean_column_names,
    standardize_text_columns,
    replace_invalid_values,
    encode_categorical,
    scale_numeric,
    ai_insight_analysis,
)
from chart_utils import (
    plot_before_after_boxplot,
    plot_before_after_bar,
    render_quality_gauge,
    plot_insight_chart,
)
from pdf_utils import build_pdf_report

_ICON_PATH = os.path.join(os.path.dirname(__file__), "icons", "logo.png")
_PAGE_ICON = _ICON_PATH if os.path.exists(_ICON_PATH) else ""

st.set_page_config(
    page_title="DataRefine - AI Data Cleaner",
    page_icon=_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# custom css
def inject_custom_css():
    st.markdown("""
    <style>
    /* App background */
    .stApp {
        background-color: #0F172A !important;
        background-image: radial-gradient(circle at top right, #1E293B, #0F172A);
        color: #F8FAFC;
    }
    
    /* Animation */
    .stApp {
        background: linear-gradient(-45deg, #020617, #0f172a, #1e293b, #020617);
        background-size: 400% 400%;
        animation: gradientMove 10s ease infinite;
    }

    @keyframes gradientMove {
        0% {background-position: 0%}
        50% {background-position: 100%}
        100% {background-position: 0%}
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0B1120 !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Titles & Text */
    .title-text { font-size: 34px; 
    font-weight: 700; 
    color: #F8FAFC; 
    margin-bottom: 10px; 
    }
    
    .subtitle-text { font-size: 16px; 
    color: #94A3B8; 
    margin-bottom: 20px; 
    }
    
    .feature { margin: 8px 0; 
    font-size: 15px; 
    color: #CBD5E1; 
    }

    /* Primary Button (Gradient) */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #22c55e);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(34,197,94,0.3);
    }

    /* Secondary / Normal Buttons (Cleaning Page etc.) */
    .stButton>button:not([kind="primary"]) {
        background-color: #1E293B;
        color: #F8FAFC;
        border: 1px solid #334155;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stButton>button:not([kind="primary"]):hover {
        border-color: #6366f1;
        color: #6366f1;
        background-color: #0F172A;
    }

    /* Clean Metric Cards */
    div[data-testid="metric-container"] {
        background-color: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
        transition: transform 0.2s, border-color 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.5);
    }

    /* Modern Tabs (Cleaning Page) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        border: 1px solid #334155;
        border-bottom: none;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(99,102,241,0.1) 0%, rgba(15,23,42,0) 100%);
        color: #F8FAFC !important;
        border-bottom: 2px solid #6366f1;
    }

    /* Radio Buttons (Sidebar Navigation) */
    div[role="radiogroup"] > label {
        padding: 6px 10px;
        border-radius: 6px;
        transition: 0.2s;
    }
    div[role="radiogroup"] > label:hover {
        background: rgba(99,102,241,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# page header
def page_header(title: str, description: str = ""):
    st.markdown(f"<h3 style='margin-top: -8px; margin-bottom: 0;'>{title}</h3>", unsafe_allow_html=True)
    if description:
        st.markdown(f"<p style='color: #94A3B8; font-size: 15px; margin-top: 5px;'>{description}</p>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 25px; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

# initialize session state variables
def init_session_state() -> None:
    defaults = {
        "raw_df": None,
        "clean_df": None,
        "file_info": None,
        "current_page": "Home",
        "quality_scores": {}, 
        "last_outlier_summary": None, 
        "cleaning_history": [], 
        "pdf_bytes": None,
        "pdf_ready": False,
        "ai_suggestions_df": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state() 

# Get the currently active DataFrame (clean or raw)
def get_active_df() -> Optional[pd.DataFrame]:
    if st.session_state.get("clean_df") is not None:
        return st.session_state.clean_df
    return st.session_state.get("raw_df")

# undo snapshot management
def _save_undo_snapshot(label: str, df: pd.DataFrame, steps: List[str] = None, source: str = "Manual") -> None:
    history: List[Tuple] = st.session_state.cleaning_history
    history.append((label, df.copy(), steps or [], source))
    if len(history) > 3:
        history.pop(0)
    st.session_state.cleaning_history = history

# 
def _apply_and_snapshot(label: str, new_df: pd.DataFrame, steps: List[str] = None, source: str = "Manual") -> None:
    if st.session_state.clean_df is not None:
        _save_undo_snapshot(label, st.session_state.clean_df, steps=steps, source=source)
    st.session_state.clean_df = new_df

# delta metrics display
def _delta_metrics(before_df: pd.DataFrame, after_df: pd.DataFrame) -> None:
    b = compute_basic_metrics(before_df)
    a = compute_basic_metrics(after_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", a["rows"], delta=a["rows"] - b["rows"], delta_color="inverse")
    c2.metric("Missing Values", a["missing"], delta=a["missing"] - b["missing"], delta_color="inverse")
    c3.metric("Duplicate Rows", a["duplicates"], delta=a["duplicates"] - b["duplicates"], delta_color="inverse")
    q_b = compute_quality_score(before_df)
    q_a = compute_quality_score(after_df)
    c4.metric("Quality Score", f"{q_a:.1f}", delta=f"{q_a - q_b:+.1f}")

# 
def _auto_clean_from_suggestions(df: pd.DataFrame, suggestions_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, List[str], str]:
    working_df = df.copy()
    steps_used: List[str] = []

    
    def _apply_rule_based_baseline(current_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        baseline_df = current_df.copy()
        baseline_steps: List[str] = []
        if int(baseline_df.isnull().sum().sum()) > 0:
            baseline_df = handle_missing_values(baseline_df, "Mean")
            baseline_steps.append("Missing values imputed (Mean for numeric / most_frequent for categorical)")

        dup_count_local = int(baseline_df.duplicated().sum())
        if dup_count_local > 0:
            baseline_df = baseline_df.drop_duplicates()
            baseline_steps.append(f"Dropped {dup_count_local} duplicate rows")

        outlier_flags_local = detect_outliers_iqr(baseline_df)
        n_outliers_local = int(outlier_flags_local.sum())
        if n_outliers_local > 0:
            baseline_df = baseline_df[~outlier_flags_local]
            baseline_steps.append(f"Removed {n_outliers_local} outlier rows (IQR method)")

        if not baseline_steps:
            baseline_steps.append("No issues found — dataset was already clean")
        return baseline_df, baseline_steps

    if suggestions_df is None or suggestions_df.empty:
        auto_clean_source = "Rule-based suggestions"
        working_df, steps_used = _apply_rule_based_baseline(working_df)
        return working_df, steps_used, auto_clean_source

    source_val = suggestions_df["Source"].iloc[0] if "Source" in suggestions_df.columns and not suggestions_df.empty else ""
    auto_clean_source = "AI (Ollama phi3)" if "phi3" in str(source_val) else "Rule-based suggestions"

    # Rule-based source: missing + duplicates + outliers only.
    if auto_clean_source == "Rule-based suggestions":
        working_df, steps_used = _apply_rule_based_baseline(working_df)
        return working_df, steps_used, auto_clean_source

    inferred_texts: List[str] = []
    drop_columns: List[str] = []
    encode_columns: List[str] = []
    text_clean_columns: List[str] = []
    wants_duplicates = False
    wants_outliers = False
    wants_negative_fix = False
    wants_scaling = False
    missing_strategy: Optional[str] = None

    for _, row in suggestions_df.iterrows():
        col_name = str(row.get("Column", "")).strip()
        issue_text = str(row.get("Issue Detected", ""))
        suggestion_text = str(row.get("Suggestion", ""))
        combined = f"{issue_text} {suggestion_text}".lower()
        inferred_texts.append(combined)

        if "drop column" in combined and col_name in working_df.columns:
            drop_columns.append(col_name)
        if ("encode" in combined or "label" in combined or "one-hot" in combined) and col_name in working_df.columns:
            encode_columns.append(col_name)
        if any(tok in combined for tok in ["text", "lower", "strip", "trim", "standardize"]) and col_name in working_df.columns:
            text_clean_columns.append(col_name)
        if "duplicate" in combined:
            wants_duplicates = True
        if any(tok in combined for tok in ["outlier", "anomal"]):
            wants_outliers = True
        if any(tok in combined for tok in ["negative", "remove negatives"]):
            wants_negative_fix = True
        if any(tok in combined for tok in ["scale", "normalize", "standardize numeric"]):
            wants_scaling = True

        if "impute median" in combined or "median" in combined:
            missing_strategy = "Median"
        elif missing_strategy is None and ("impute mode" in combined or "most_frequent" in combined or "mode" in combined):
            missing_strategy = "Mode"
        elif missing_strategy is None and ("impute mean" in combined or "mean" in combined):
            missing_strategy = "Mean"
        elif missing_strategy is None and ("missing" in combined or "impute" in combined):
            missing_strategy = "Mean"

    if wants_negative_fix:
        before_missing = int(working_df.isnull().sum().sum())
        working_df = replace_invalid_values(working_df, treat_negative_as_invalid=True)
        after_missing = int(working_df.isnull().sum().sum())
        if after_missing > before_missing:
            steps_used.append("Replaced invalid/negative numeric values with missing")

    unique_drop_cols = [c for c in dict.fromkeys(drop_columns) if c in working_df.columns]
    if unique_drop_cols:
        working_df = working_df.drop(columns=unique_drop_cols)
        steps_used.append(f"Dropped suggested column(s): {', '.join(unique_drop_cols)}")

    if missing_strategy and int(working_df.isnull().sum().sum()) > 0:
        working_df = handle_missing_values(working_df, missing_strategy)
        steps_used.append(f"Imputed missing values using {missing_strategy}")

    if wants_duplicates:
        dup_count = int(working_df.duplicated().sum())
        if dup_count > 0:
            working_df = working_df.drop_duplicates()
            steps_used.append(f"Dropped {dup_count} duplicate rows")

    if wants_outliers:
        outlier_flags = detect_outliers_iqr(working_df)
        n_outliers = int(outlier_flags.sum())
        if n_outliers > 0:
            working_df = working_df[~outlier_flags]
            steps_used.append(f"Removed {n_outliers} outlier rows (IQR method)")

    unique_text_cols = [c for c in dict.fromkeys(text_clean_columns) if c in working_df.columns]
    if unique_text_cols:
        working_df = standardize_text_columns(working_df, columns=unique_text_cols, lowercase=True, strip=True)
        steps_used.append(f"Standardized text in: {', '.join(unique_text_cols)}")

    unique_encode_cols = [c for c in dict.fromkeys(encode_columns) if c in working_df.columns]
    encoded_any = []
    for col in unique_encode_cols:
        if col in working_df.columns and str(working_df[col].dtype) in ("object", "category", "string"):
            working_df = encode_categorical(working_df, col, method="Label")
            encoded_any.append(col)
    if encoded_any:
        steps_used.append(f"Applied label encoding on: {', '.join(encoded_any)}")

    if wants_scaling:
        before_cols = set(working_df.columns)
        working_df = scale_numeric(working_df, method="Standard")
        added = [c for c in working_df.columns if c not in before_cols and c.endswith("_std")]
        if added:
            steps_used.append(f"Scaled numeric columns (Standard): {len(added)} generated")

    if not steps_used:
        # Fallback to baseline cleaning when AI suggestions are too generic.
        working_df, steps_used = _apply_rule_based_baseline(working_df)
        steps_used = [f"Fallback: {s}" for s in steps_used]
    return working_df, steps_used, auto_clean_source

# home page
def render_home_page():
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown('<div class="title-text">DataRefine - AI Data Cleaner</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle-text">A modern AI-powered data cleaning & profiling tool for fast, reliable preprocessing.</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature">Upload CSV / Excel datasets</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature">Profile schema & missing values</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature">Smart cleaning & outlier detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature">Compare before vs after</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature">Export professional reports</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Start Cleaning"):
            st.session_state.current_page = "Upload"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True) 
        if os.path.exists(_ICON_PATH):
            st.image(_ICON_PATH, width=280)
        else:
            st.markdown(
                "<div style='font-size: 84px; text-align: center; margin-top: 24px;'>🧹</div>",
                unsafe_allow_html=True,
            )

# upload page
def render_upload_page() -> None:
    page_header("Upload Dataset", "Upload a CSV or Excel file (up to 200 MB) to begin profiling and cleaning.")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xls", "xlsx"],
        help="Supported formats: .csv, .xls, .xlsx (max 200 MB)",
    )
    if uploaded_file is not None:
        try:
            df, file_info = load_dataset(uploaded_file)
        except ValueError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
        st.session_state.file_info = file_info
        st.session_state.raw_df = df.copy()
        st.session_state.clean_df = df.copy()
        st.session_state.cleaning_history = []
        st.session_state.quality_scores = {} 
        st.session_state.pdf_bytes = None
        st.session_state.pdf_ready = False
        st.session_state.ai_suggestions_df = None

        fi = file_info or {}
        st.success("File uploaded successfully.")
        c1, c2, c3 = st.columns(3)
        c1.metric("File Name", fi.get("name", "-"))
        c2.metric("File Size (MB)", fi.get("size_mb", 0))
        c3.metric("File Type", fi.get("type", "-"))

        st.write("Preview (first 5 rows):")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Proceed to Dashboard", type="primary"):
            st.session_state.current_page = "Dashboard"
            st.rerun()
    else:
        st.info("No file uploaded yet.")

# dashboard page
def render_dashboard_page() -> None:
    page_header("Data Overview Dashboard", "Quick summary and profiling of your dataset schema and statistics.")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset first from the **Upload** page")
        return

    df = get_active_df()
    metrics = compute_basic_metrics(df)
    quality = compute_quality_score(df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Rows", metrics["rows"])
    c2.metric("Total Columns", metrics["cols"])
    c3.metric("Missing Values", metrics["missing"])
    c4.metric("Duplicate Rows", metrics["duplicates"])
    c5.metric("Quality Score (/ 100)", f"{quality:.1f}")

    st.markdown("---")
    st.markdown("#### Schema Summary")
    st.dataframe(build_dtype_summary(df), use_container_width=True, height=320)

    st.markdown("#### Data Preview (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True, height=320)

    st.markdown("---")
    st.markdown("#### Statistical Summary (Numerical Columns)")
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        st.info("No numerical columns found in the dataset")
    else:
        stats = num_df.describe().T.reset_index().rename(columns={"index": "Column"})
        for col in stats.columns:
            if col != "Column":
                stats[col] = pd.to_numeric(stats[col], errors="coerce").round(4)
        stats["skewness"] = stats["Column"].map(lambda c: round(float(num_df[c].skew()), 4))
        stats["kurtosis"] = stats["Column"].map(lambda c: round(float(num_df[c].kurt()), 4))
        st.dataframe(stats, use_container_width=True, height=min(420, 45 + len(stats) * 38))

# cleaning page
def render_cleaning_page() -> None:
    page_header("Data Cleaning", "Apply AI-powered automated cleaning or manually execute data transformations.")

    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset first from the **Upload** page.")
        return

    df = get_active_df()
    tabs = st.tabs(["Smart Cleaning", "Manual Cleaning"])

    with tabs[0]:
        st.markdown("#### AI Cleaning Analysis")
        metrics = compute_basic_metrics(df)
        quality_now = compute_quality_score(df)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rows", metrics["rows"])
        m2.metric("Columns", metrics["cols"])
        m3.metric("Missing Values", metrics["missing"])
        m4.metric("Duplicate Rows", metrics["duplicates"])
        m5.metric("Data Quality Score", f"{quality_now:.2f}")

        st.markdown("**Recommended Cleaning Actions**")
            
        if st.button("Get AI Suggestions", key="get_ai_suggestions", type="primary"):
            with st.spinner("Analyzing dataset profile with AI..."):
                st.session_state["ai_suggestions_df"] = generate_cleaning_suggestions(df)

        suggestions_df = st.session_state.get("ai_suggestions_df", None)
        if suggestions_df is not None:
            if "Source" in suggestions_df.columns:
                source_val = suggestions_df["Source"].iloc[0] if not suggestions_df.empty else ""
                if "phi3" in source_val:
                    st.success("Suggestions generated by **Ollama phi3** AI model.")
                else:
                    st.warning("Ollama phi3 unavailable — showing **rule-based** suggestions.")
                display_df = suggestions_df.drop(columns=["Source"])
            else:
                display_df = suggestions_df
            st.dataframe(display_df, use_container_width=True, height=260)
        else:
            st.caption("Click the button above to analyze the dataset and get cleaning suggestions.")

        st.markdown("##### AI Auto Clean")
        st.caption("Automatically handles missing values, duplicates, and outliers.")

        if st.button("Auto Clean using AI Suggestions", key="auto_clean_ai"):
            with st.spinner("Applying cleaning operations..."):
                suggestions_df = st.session_state.get("ai_suggestions_df", None)
                working_df, steps_used, auto_clean_source = _auto_clean_from_suggestions(df, suggestions_df)

                auto_label = f"Auto Clean ({auto_clean_source})"
                _apply_and_snapshot(auto_label, working_df, steps=steps_used, source=auto_clean_source)

                quality_after = compute_quality_score(working_df)

            st.success(f"Auto cleaning complete! · Source: **{auto_clean_source}**")
            for step in steps_used:
                st.write(f" {step}")

            st.markdown("#### Impact Summary")
            _delta_metrics(df, working_df)

            st.markdown("#### Quality Score")
            render_quality_gauge(quality_now, quality_after)

        st.markdown("---")
        _render_undo_controls(suffix="smart")

    with tabs[1]:
        st.markdown("#### Manual Cleaning Operations")
        sub_tabs = st.tabs(["Missing Values", "Duplicates", "Outliers", "Column Operations", "Text & Categories", "Scaling"])

        with sub_tabs[0]:
            st.markdown("##### Handle Missing Values")
            if int(df.isnull().sum().sum()) == 0:
                st.success("No missing values in current dataset.")
            else:
                st.caption("Tip: Run **Replace Invalid Values** (Text tab) before imputing.")
                strategy = st.selectbox("Strategy", ["Drop", "Mean", "Median", "Mode", "KNN Imputation"])
                if st.button("Apply Missing Value Strategy", key="apply_missing"):
                    with st.spinner("Imputing missing values..."):
                        new_df = handle_missing_values(df, strategy)
                    _apply_and_snapshot(f"Missing: {strategy}", new_df)
                    st.success(f"Applied '{strategy}'.")
                    _delta_metrics(df, new_df)

        with sub_tabs[1]:
            st.markdown("##### Remove Duplicate Rows")
            dup_count = int(df.duplicated().sum())
            st.info(f"Current duplicate rows: **{dup_count}**")
            if dup_count > 0 and st.button("Drop Duplicates", key="drop_dupes"):
                new_df = df.drop_duplicates()
                _apply_and_snapshot("Drop Duplicates", new_df)
                st.success(f"Removed {dup_count} duplicate rows.")
                _delta_metrics(df, new_df)

        with sub_tabs[2]:
            st.markdown("##### Outlier Detection & Removal")
            outlier_method = st.selectbox("Detection Method", ["IQR", "Z-Score"])
            numeric_cols_list = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols_list:
                st.info("No numeric columns available for outlier detection.")
            else:
                z_thresh = 3.0
                if outlier_method == "Z-Score":
                    z_thresh = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)

                if st.button("Detect & Remove Outliers", key="remove_outliers"):
                    with st.spinner("Detecting outliers..."):
                        if outlier_method == "IQR":
                            flags = detect_outliers_iqr(df)
                        else:
                            flags = detect_outliers_zscore(df, threshold=z_thresh)

                    n_flagged = int(flags.sum())
                    if n_flagged == 0:
                        st.success("No outliers detected with current settings.")
                    else:
                        new_df = df[~flags]
                        _apply_and_snapshot(f"Outliers ({outlier_method})", new_df)
                        st.success(f"Removed {n_flagged} outlier rows.")
                        _delta_metrics(df, new_df)
                        col_to_plot = numeric_cols_list[0]
                        st.markdown(f"##### Distribution: `{col_to_plot}`")
                        plot_before_after_boxplot(df[col_to_plot], new_df[col_to_plot], col_to_plot)

        with sub_tabs[3]:
            st.markdown("##### Column Operations")
            col_selected = st.selectbox("Select Column", df.columns.tolist(), key="col_op_col")
            operation = st.selectbox("Operation", ["Rename column", "Change datatype", "Drop column", "Encode categorical"], key="col_op_op")
            kwargs: Dict = {}
            if operation == "Rename column":
                kwargs["new_name"] = st.text_input("New column name", key="col_rename")
            elif operation == "Change datatype":
                kwargs["target_type"] = st.selectbox("Target type", ["int", "float", "string", "datetime"], key="col_dtype")

            if st.button("Apply Column Operation", key="apply_col_op"):
                if operation == "Rename column" and not kwargs.get("new_name"):
                    st.warning("Please enter a new column name.")
                else:
                    new_df = apply_column_operation(df, operation, col_selected, **kwargs)
                    _apply_and_snapshot(f"{operation}: {col_selected}", new_df)
                    st.success(f"Applied '{operation}' on `{col_selected}`.")

            st.markdown("---")
            if st.button("Standardize Column Names", key="clean_col_names"):
                new_df = clean_column_names(df)
                _apply_and_snapshot("Clean Column Names", new_df)
                st.success("Column names standardized to snake_case.")

        with sub_tabs[4]:
            st.markdown("##### Replace Invalid Values")
            treat_neg = st.checkbox("Treat negative numeric values as invalid")
            if st.button("Replace Invalid Values", key="replace_invalid"):
                before_missing = int(df.isnull().sum().sum())
                new_df = replace_invalid_values(df, treat_negative_as_invalid=treat_neg)
                new_missing = int(new_df.isnull().sum().sum())
                _apply_and_snapshot("Replace Invalid Values", new_df)
                st.success(f"Done. Missing values: {before_missing} → {new_missing}")

            st.markdown("---")
            st.markdown("##### Text Standardization")
            text_cols = df.select_dtypes(include=["object"]).columns.tolist()
            if text_cols:
                selected_text_cols = st.multiselect("Columns to clean", text_cols, default=text_cols)
                col_lower, col_strip = st.checkbox("Lowercase", value=True), st.checkbox("Strip whitespace", value=True)
                if st.button("Apply Text Cleaning", key="apply_text"):
                    new_df = standardize_text_columns(df, columns=selected_text_cols, lowercase=col_lower, strip=col_strip)
                    _apply_and_snapshot("Text Standardization", new_df)
                    st.success("Text columns cleaned.")

            st.markdown("---")
            st.markdown("##### Encode Categorical Column")
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if cat_cols:
                encode_col = st.selectbox("Column to encode", cat_cols, key="encode_col")
                encode_method = st.selectbox("Encoding method", ["Label", "One-Hot"], key="encode_method")
                if st.button("Apply Encoding", key="apply_encode"):
                    new_df = encode_categorical(df, encode_col, method=encode_method)
                    _apply_and_snapshot(f"Encode {encode_col} ({encode_method})", new_df)
                    st.success(f"Encoded `{encode_col}` using {encode_method} encoding.")

        with sub_tabs[5]:
            st.markdown("##### Numeric Scaling")
            scale_method = st.selectbox("Scaling Method", ["Min-Max", "Standard"], key="scale_method")
            numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
            to_scale = [c for c in numeric_cols_all if not c.endswith("_scaled") and not c.endswith("_std")]
            if not to_scale:
                st.info("All numeric columns are already scaled.")
            else:
                if st.button("Apply Scaling", key="apply_scale"):
                    new_df = scale_numeric(df, method=scale_method)
                    _apply_and_snapshot(f"Scale Numeric ({scale_method})", new_df)
                    st.success(f"Scaled {len(to_scale)} column(s) using {scale_method}.")

        st.markdown("---")
        _render_undo_controls(suffix="manual")

# undo controls (shared between smart and manual cleaning tabs)
def _render_undo_controls(suffix: str = "") -> None:
    history: List[Tuple] = st.session_state.cleaning_history
    if not history:
        st.caption("No cleaning steps to undo")
        return

    step_labels = [f"Step {i+1}: {item[0]}" for i, item in enumerate(history)]
    selected = st.selectbox("Restore to step", options=["— current —"] + step_labels[::-1], key=f"undo_select_{suffix}")

    if selected != "— current —" and st.button("Restore Selected Step", key=f"do_undo_{suffix}"):
        idx = len(history) - 1 - step_labels[::-1].index(selected)
        restored_df = history[idx][1]
        st.session_state.clean_df = restored_df.copy()
        st.session_state.cleaning_history = history[:idx]
        st.success(f"Restored to: {selected}")
        st.rerun()

# before vs after page
def render_before_after_page() -> None:
    page_header("Before vs After", "Compare raw and cleaned datasets to understand the impact of your operations.")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset first from the **Upload** page.")
        return
    if st.session_state.clean_df is None:
        st.info("No cleaning steps applied yet.")
        return

    raw_df, clean_df = st.session_state.raw_df, st.session_state.clean_df
    comparison, _, _ = build_before_after_comparison(raw_df, clean_df)
    figs = plot_before_after_bar(comparison)

    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        st.subheader("Comparison Table")
        st.dataframe(comparison, use_container_width=True, height=340)
    with col2:
        st.subheader("Row & Column Counts")
        st.plotly_chart(figs[0], use_container_width=True)

    st.markdown("---")
    st.subheader("Data Quality Score")
    render_quality_gauge(compute_quality_score(raw_df), compute_quality_score(clean_df))

# data visualization page
def render_visualization_page() -> None:
    page_header("Data Visualization (AI Insights)", "AI analyzes the dataset and selects the 4 most informative insights across distributions, correlations, categories, and trends.")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset first from the **Upload** page.")
        return
    df = get_active_df()
    top_insights = ai_insight_analysis(df)
    if not top_insights:
        st.info("No meaningful insights found. Try cleaning/transforming the dataset first.")
        return
    # Each insight is a dict with 'title', 'description', and info needed to plot the chart.
    def _render_single(insight: Dict) -> None:
        st.markdown(f"**{insight.get('title', 'Insight')}**")
        if desc := insight.get("description"): st.caption(desc)
        plot_insight_chart(df, insight)

    col1, col2 = st.columns(2)
    if len(top_insights) >= 1:
        with col1: _render_single(top_insights[0])
    if len(top_insights) >= 2:
        with col2: _render_single(top_insights[1])

    col3, col4 = st.columns(2)
    if len(top_insights) >= 3:
        with col3: _render_single(top_insights[2])
    if len(top_insights) >= 4:
        with col4: _render_single(top_insights[3])

# export page: PDF report + cleaned dataset download
def render_export_page() -> None:
    page_header("Export Report & Dataset", "Download your fully cleaned dataset or generate a comprehensive PDF summary.")
    if st.session_state.raw_df is None:
        st.warning("Please upload and process a dataset before exporting.")
        return

    raw_df, clean_df = st.session_state.raw_df, get_active_df()
    file_info = st.session_state.file_info or {}
    file_name = file_info.get("name", "-")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Export Profiling Report (PDF)")
        st.markdown("Generate a **PDF summary report** with metrics, quality scores, and cleaning outcomes.")
        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Generating PDF Report... This may take a few seconds."):
                try:
                    pdf_bytes = build_pdf_report(
                            raw_df, clean_df, file_name=file_name,
                            cleaning_history=st.session_state.get("cleaning_history", []),
                        )
                    st.session_state.pdf_bytes = pdf_bytes
                    st.session_state.pdf_ready = True
                except Exception as e:
                    st.error(f"Failed to generate PDF report: {e}")
                    st.session_state.pdf_ready = False

        if st.session_state.get("pdf_ready") and st.session_state.get("pdf_bytes"):
            st.success("PDF ready to download.")
            st.download_button(
                label="Download Report", data=st.session_state.pdf_bytes,
                file_name="DataRefine_AI_report.pdf", mime="application/pdf", key="pdf_download_btn",
            )

    with col_right:
        st.markdown("#### Export Cleaned Dataset\nDownload the **cleaned dataset** after all preprocessing steps.")
        if clean_df is not None and not clean_df.empty:
            base_name = os.path.splitext(file_name)[0] + "_cleaned" if file_name != "-" else "cleaned_dataset"
            fmt = st.radio("Select format", options=["CSV", "Excel (.xlsx)"], horizontal=True, key="export_fmt_radio")

            if fmt == "CSV":
                st.download_button(
                    label="Download Cleaned CSV", data=clean_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{base_name}.csv", mime="text/csv", key="csv_download_btn",
                )
            else:
                from io import BytesIO
                buf = BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    clean_df.to_excel(writer, index=False, sheet_name="Cleaned Data")
                buf.seek(0)
                st.download_button(
                    label="Download Cleaned Excel", data=buf.getvalue(),
                    file_name=f"{base_name}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="xlsx_download_btn",
                )
        else:
            st.info("No cleaned dataset yet. Apply cleaning steps first.")

# main app with sidebar navigation
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### DataRefine - AI Data Cleaner")
        st.caption("AI-powered data cleaning & profiling.")

        if st.session_state.raw_df is not None:
            df = get_active_df()
            metrics = compute_basic_metrics(df)
            st.success("Dataset Loaded")
            st.metric("Rows", metrics["rows"])
            st.metric("Columns", metrics["cols"])
            st.metric("Quality Score (/ 100)", f"{compute_quality_score(df):.1f}")
        else:
            st.info("No dataset loaded yet.")

        st.markdown("---")
        pages = ["Home", "Upload", "Dashboard", "Cleaning", "Before vs After", "Data Visualization", "Export Report"]
        current = st.session_state.current_page
        default_index = pages.index(current) if current in pages else 0
        return st.radio("Navigation", options=pages, index=default_index)

# main function to run the app
def main() -> None:
    inject_custom_css()
    selected_page = render_sidebar()
    st.session_state.current_page = selected_page

    pages = {
        "Home": render_home_page,
        "Upload": render_upload_page,
        "Dashboard": render_dashboard_page,
        "Cleaning": render_cleaning_page,
        "Before vs After": render_before_after_page,
        "Data Visualization": render_visualization_page,
        "Export Report": render_export_page
    }
    pages.get(selected_page, render_home_page)()

if __name__ == "__main__":
    main()