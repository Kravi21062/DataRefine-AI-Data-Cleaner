from typing import Dict, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from theme import PRIMARY_COLOR, SECONDARY_COLOR

_FONT_FAMILY = "Inter, Segoe UI, Arial, sans-serif"
_TITLE_SIZE = 15
_AXIS_SIZE = 12
_TICK_SIZE = 11


# Shared layout builder
# ---------------------------------------------------------------------------
def _layout(extra: dict = None) -> dict:
    base = dict(
        font=dict(family=_FONT_FAMILY, size=_AXIS_SIZE),
        title_font=dict(family=_FONT_FAMILY, size=_TITLE_SIZE),
        title_x=0.03,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=70, r=60, t=70, b=80),   # r=60 so text labels are never clipped
        height=420,
        xaxis=dict(
            title_font=dict(size=_AXIS_SIZE), tickfont=dict(size=_TICK_SIZE),
            # showgrid=True, gridcolor="#e3e8f0", # #e3e8f0
            linecolor="#94a3b8", zeroline=False, # #94a3b8
        ),
        yaxis=dict(
            title_font=dict(size=_AXIS_SIZE), tickfont=dict(size=_TICK_SIZE),
            # showgrid=True, gridcolor="#e3e8f0", # #e3e8f0
            linecolor="#94a3b8", zeroline=False, # #94a3b8
        ),
        legend=dict(
            font=dict(size=_TICK_SIZE), bgcolor="rgba(0,0,0,0)", 
        ),
    )
    if extra:
        # deep-merge xaxis / yaxis overrides
        for k, v in extra.items():
            if k in ("xaxis", "yaxis") and k in base and isinstance(v, dict):
                base[k] = {**base[k], **v}
            else:
                base[k] = v
    return base


def _apply(fig: go.Figure, extra: dict = None) -> go.Figure:
    fig.update_layout(**_layout(extra))
    return fig


# Dashboard charts
# ---------------------------------------------------------------------------
def plot_missing_values_bar(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.info("No missing values detected.")
        return

    data = missing.reset_index()
    data.columns = ["Column", "Missing"]

    fig = px.bar(data, x="Column", y="Missing",
                 title="Missing Values per Column",
                 color_discrete_sequence=[PRIMARY_COLOR], text="Missing")
    fig.update_traces(textposition="outside", textfont_size=10, marker_line_width=0,
                      cliponaxis=False)
    _apply(fig, {"xaxis": {"title": "Column", "tickangle": -38, "showgrid": False},
                 "yaxis": {"title": "Missing Count"},
                 "margin": dict(l=70, r=60, t=70, b=110)})
    st.plotly_chart(fig, use_container_width=True)
        

def plot_missing_values_heatmap(df: pd.DataFrame) -> None:
    if df.empty:
        return
    null_m = df.isnull().astype(int)
    if null_m.sum().sum() == 0:
        st.info("No missing values to display in heatmap.")
        return
    sample = null_m.sample(n=min(200, len(null_m)), random_state=42)
    fig = px.imshow(sample.T, color_continuous_scale=["#1e293b", "#ef4444"],
                    aspect="auto", title="Missing Values Heatmap (Sampled Rows)")
    _apply(fig, {"xaxis": {"title": "Row Index (sampled)", "showgrid": False},
                 "yaxis": {"title": "Column", "showgrid": False},
                 "coloraxis_showscale": False})
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        st.info("Not enough numeric columns for correlation heatmap.")
        return
    corr = num.corr().round(2)
    n = len(corr.columns)
    h = min(650, max(400, n * 52))
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title="Correlation Heatmap (Numeric Features)", aspect="auto")
    fig.update_traces(textfont_size=10)
    _apply(fig, {"height": h,
                 "xaxis": {"tickangle": -38, "showgrid": False, "title": ""},
                 "yaxis": {"showgrid": False, "title": ""},
                 "margin": dict(l=90, r=60, t=70, b=80),
                 "coloraxis_colorbar": dict(title="r", tickfont=dict(size=10), len=0.8)})
    st.plotly_chart(fig, use_container_width=True)


def plot_distributions(df: pd.DataFrame) -> None:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        st.info("No numeric columns for distribution plots.")
        return
    top_cols = num.var().sort_values(ascending=False).head(3).index.tolist()
    colors = [PRIMARY_COLOR, SECONDARY_COLOR, "#F97316"]
    fig = go.Figure()
    for i, col in enumerate(top_cols):
        fig.add_trace(go.Histogram(x=num[col], name=col, opacity=0.75, nbinsx=30,
                                   marker_color=colors[i % 3],
                                   marker_line=dict(width=0.4, color="rgba(0,0,0,0.3)")))
    _apply(fig, {"barmode": "overlay",
                 "title": "Distribution of Top Numeric Columns (by Variance)",
                 "xaxis": {"title": "Value"},
                 "yaxis": {"title": "Count"},
                 "legend": dict(orientation="h", yanchor="bottom", y=1.04,
                                xanchor="left", x=0, font=dict(size=_TICK_SIZE),
                                bgcolor="rgba(0,0,0,0)")})
    st.plotly_chart(fig, use_container_width=True)


# Cleaning-page charts
# ---------------------------------------------------------------------------
def plot_before_after_boxplot(before: pd.Series, after: pd.Series, col_name: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Box(y=before.dropna(), name="Before Cleaning",
                         marker_color=PRIMARY_COLOR, boxmean="sd", line_width=1.5))
    fig.add_trace(go.Box(y=after.dropna(), name="After Cleaning",
                         marker_color=SECONDARY_COLOR, boxmean="sd", line_width=1.5))
    _apply(fig, {"title": f"Outlier Impact — {col_name}",
                 "xaxis": {"title": "Dataset Version", "showgrid": False},
                 "yaxis": {"title": col_name},
                 "height": 400})
    st.plotly_chart(fig, use_container_width=True)

# quality score gauge chart show
def render_quality_gauge(before: float, after: float) -> None:
    steps = [{"range": [0, 40], "color": "rgba(239,68,68,0.15)"},
             {"range": [40, 70], "color": "rgba(234,179,8,0.15)"},
             {"range": [70, 100], "color": "rgba(34,197,94,0.15)"}]
    fig = go.Figure()
    for i, (val, ref, color, label, domain) in enumerate([
        (before, None, PRIMARY_COLOR, "Before Cleaning", [0, 0.46]),
        (after,  before, SECONDARY_COLOR, "After Cleaning", [0.54, 1]),
    ]):
        kw = dict(mode="gauge+number" + ("+delta" if ref is not None else ""),
                  value=val,
                  title={"text": f"Quality Score<br>{label}",
                         "font": {"size": 13, "family": _FONT_FAMILY}},
                  number={"font": {"size": 34}, "suffix": "/100"},
                  gauge={"axis": {"range": [0, 100], "tickfont": {"size": 10}},
                         "bar": {"color": color, "thickness": 0.25}, "steps": steps},
                  domain={"x": domain, "y": [0.1, 0.9]})
        if ref is not None:
            kw["delta"] = {"reference": ref, "valueformat": ".1f",
                           "increasing": {"color": "#22C55E"},
                           "decreasing": {"color": "#EF4444"}}
        fig.add_trace(go.Indicator(**kw))
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=20, b=20),
                      paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=_FONT_FAMILY))
    st.plotly_chart(fig, use_container_width=True)


# Before/After comparison chart
# ---------------------------------------------------------------------------
def plot_before_after_bar(comparison: pd.DataFrame) -> go.Figure:
    counts_rows = comparison[comparison["Metric"].isin(
        ["Rows", "Columns", "Missing Values", "Duplicate Rows"])]
    def _bar_fig(sub: pd.DataFrame, title: str, y_label: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sub["Metric"], y=sub["Before"], name="Before Cleaning",
                             marker_color=PRIMARY_COLOR, marker_line_width=0,
                             text=sub["Before"], textposition="outside",
                             textfont_size=11, cliponaxis=False))
        fig.add_trace(go.Bar(x=sub["Metric"], y=sub["After"], name="After Cleaning",
                             marker_color=SECONDARY_COLOR, marker_line_width=0,
                             text=sub["After"], textposition="outside",
                             textfont_size=11, cliponaxis=False))
        _apply(fig, {"barmode": "group", "title": title,
                     "xaxis": {"title": "Metric", "showgrid": False,
                               "tickangle": -20},
                     "yaxis": {"title": y_label},
                     "legend": dict(orientation="h", yanchor="bottom", y=1.04,
                                    xanchor="left", x=0,
                                    font=dict(size=_TICK_SIZE),
                                    bgcolor="rgba(0,0,0,0)"),
                     "margin": dict(l=70, r=60, t=80, b=80)})
        return fig

    fig1 = _bar_fig(counts_rows, "Before vs After — Row & Column Counts", "Count")
    return [fig1]


# AI Insight chart dispatcher
# ---------------------------------------------------------------------------

# Columns that are IDs / meaningless for distribution insight
_ID_KEYWORDS = {"id", "key", "code", "index", "tract", "zip", "postal",
                "phone", "fax", "ssn", "uuid", "guid"}


def _is_id_column(col_name: str) -> bool:
    low = col_name.lower()
    return any(kw in low for kw in _ID_KEYWORDS)

# Return non-overlapping annotation positions for mean & median lines.
def _annotation_positions(mean_v: float, median_v: float):

    if abs(mean_v - median_v) < abs(mean_v) * 0.01:
        return "top right", "bottom right"
    if mean_v >= median_v:
        return "top right", "top left"
    return "top left", "top right"


def plot_insight_chart(df: pd.DataFrame, insight: Dict[str, Any]) -> None:
    if df is None or df.empty:
        st.info("No data available for visualization.")
        return

    insight_type = insight.get("type")
    cols = insight.get("columns", []) or []
    title = insight.get("title", "Insight")

    # Distribution 
    if insight_type == "distribution" and cols:
        col = cols[0]
        if col not in df.columns:
            st.info(f"Column `{col}` not found.")
            return
        if _is_id_column(col):
            st.info(f"`{col}` looks like an ID column — skipping distribution chart.")
            return

        series = df[col].dropna()
        mean_v, median_v = series.mean(), series.median()
        pos_mean, pos_median = _annotation_positions(mean_v, median_v)

        fig = px.histogram(df, x=col, nbins=30, title=title,
                           color_discrete_sequence=[PRIMARY_COLOR], opacity=0.85)
        fig.update_traces(marker_line_width=0.4,
                          marker_line_color="rgba(0,0,0,0.3)")

        fmt = ".2f" if abs(mean_v) < 1e6 else ".3g"
        fig.add_vline(x=mean_v,   line_dash="dash", line_color="#F97316", line_width=1.8,
                      annotation_text=f"Mean: {mean_v:{fmt}}",
                      annotation_position=pos_mean,
                      annotation_font=dict(size=10, color="#F97316"),
                      annotation_bgcolor="rgba(0,0,0,0.5)",
                      annotation_borderpad=3)
        fig.add_vline(x=median_v, line_dash="dot",  line_color="#22C55E", line_width=1.8,
                      annotation_text=f"Median: {median_v:{fmt}}",
                      annotation_position=pos_median,
                      annotation_font=dict(size=10, color="#22C55E"),
                      annotation_bgcolor="rgba(0,0,0,0.5)",
                      annotation_borderpad=3)

        _apply(fig, {"xaxis": {"title": col},
                     "yaxis": {"title": "Count"},
                     "margin": dict(l=70, r=60, t=90, b=80)})
        st.plotly_chart(fig, use_container_width=True)
        return

    # Correlation 
    if insight_type == "correlation" and len(cols) >= 2:
        x_col, y_col = cols[0], cols[1]
        if x_col not in df.columns or y_col not in df.columns:
            st.info("Columns not found for correlation insight.")
            return
        # sample large datasets so browser does not freeze
        _MAX_SCATTER = 3000
        plot_df = df[[x_col, y_col]].dropna()
        if len(plot_df) > _MAX_SCATTER:
            plot_df = plot_df.sample(n=_MAX_SCATTER, random_state=42)
            st.caption(f"Showing a random sample of {_MAX_SCATTER:,} rows for performance.")
        fig = px.scatter(plot_df, x=x_col, y=y_col, opacity=0.60, title=title,
                         color_discrete_sequence=[PRIMARY_COLOR])
        fig.update_traces(marker=dict(size=4,
                                      line=dict(width=0.3, color="rgba(0,0,0,0.4)")))
        _apply(fig, {"xaxis": {"title": x_col},
                     "yaxis": {"title": y_col}})
        st.plotly_chart(fig, use_container_width=True)
        return

    # Category
    if insight_type == "category" and cols:
        col = cols[0]
        if col not in df.columns:
            st.info(f"Column `{col}` not found.")
            return
        counts = (df[col].astype(str).value_counts().head(10)
                  .sort_values(ascending=True).reset_index())
        counts.columns = [col, "Count"]

        # Estimate max label length to set left margin
        max_label_len = counts[col].str.len().max()
        left_margin = max(80, min(260, max_label_len * 7))

        fig = px.bar(counts, x="Count", y=col, orientation="h", title=title,
                     color_discrete_sequence=[PRIMARY_COLOR], text="Count")
        fig.update_traces(textposition="outside", textfont_size=10,
                          marker_line_width=0, cliponaxis=False)
        _apply(fig, {"xaxis": {"title": "Count"},
                     "yaxis": {"title": col, "automargin": True,
                               "showgrid": False},
                     "margin": dict(l=left_margin, r=80, t=70, b=60)})
        st.plotly_chart(fig, use_container_width=True)
        return

    # Proportion
    if insight_type == "proportion" and cols:
        col = cols[0]
        if col not in df.columns:
            st.info(f"Column `{col}` not found.")
            return
        counts = df[col].astype(str).value_counts().head(8).reset_index()
        counts.columns = [col, "Count"]
        fig = px.pie(counts, names=col, values="Count", title=title, hole=0.38)
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          textfont_size=11,
                          marker=dict(line=dict(color="rgba(0,0,0,0.2)", width=1)))
        fig.update_layout(height=420, margin=dict(l=30, r=30, t=70, b=60),
                          paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family=_FONT_FAMILY, size=_AXIS_SIZE),
                          title_font=dict(family=_FONT_FAMILY, size=_TITLE_SIZE),
                          title_x=0.03, showlegend=True,
                          legend=dict(font=dict(size=_TICK_SIZE),
                                      bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)
        return

    # Trend
    if insight_type == "trend" and len(cols) >= 2:
        date_col, value_col = cols[0], cols[1]
        if date_col not in df.columns or value_col not in df.columns:
            st.info("Columns not found for trend insight.")
            return
        tmp = df[[date_col, value_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col])
        if tmp.empty:
            st.info("No valid dates for trend insight.")
            return
        daily = (tmp.set_index(date_col)[value_col]
                 .resample("D").mean().dropna().reset_index())
        if daily.empty:
            st.info("Not enough data to plot trend.")
            return
        fig = px.line(daily, x=date_col, y=value_col, title=title, markers=True,
                      color_discrete_sequence=[PRIMARY_COLOR])
        fig.update_traces(line=dict(width=2), marker=dict(size=5))
        _apply(fig, {"xaxis": {"title": date_col},
                     "yaxis": {"title": value_col}})
        st.plotly_chart(fig, use_container_width=True)
        return

    st.info("Insight type not supported for visualization.")