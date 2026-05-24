from __future__ import annotations
from io import BytesIO
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether, PageBreak,
)
from reportlab.platypus.flowables import Flowable
from data_utils import compute_basic_metrics, compute_quality_score

C_PRIMARY = colors.HexColor("#4F46E5")
C_SUCCESS = colors.HexColor("#22C55E")
C_WARNING = colors.HexColor("#F97316")
C_DANGER = colors.HexColor("#EF4444")
C_BG_LIGHT  = colors.HexColor("#F8F9FF")
C_BG_HEADER = colors.HexColor("#EEF2FF")
C_RULE = colors.HexColor("#C7D2FE")
C_TEXT = colors.HexColor("#1E293B")
C_MUTED = colors.HexColor("#64748B")
C_WHITE = colors.white

PAGE_W, PAGE_H = A4
MARGIN   = 2.0 * cm
CONTENT_W = PAGE_W - 2 * MARGIN
_PALETTE = ["#4F46E5", "#22C55E", "#F97316", "#EF4444", "#8B5CF6", "#06B6D4"]

# generate pdf report styles
def _build_styles():
    base = getSampleStyleSheet()
    def S(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=base[parent], **kw)
    return {
        "title": S("DR_Title", fontSize=22, leading=28, textColor=C_PRIMARY, fontName="Helvetica-Bold", spaceAfter=4),
        "subtitle": S("DR_Sub", fontSize=10, leading=14, textColor=C_MUTED, fontName="Helvetica", spaceAfter=2),
        "meta": S("DR_Meta", fontSize=8, leading=11, textColor=C_MUTED, fontName="Helvetica-Oblique"),
        "section": S("DR_Sec", fontSize=13, leading=18, textColor=C_PRIMARY, fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6),
        "body": S("DR_Body", fontSize=9, leading=13, textColor=C_TEXT, fontName="Helvetica", spaceAfter=4),
        "body_bold": S("DR_BodyB", fontSize=9, leading=13, textColor=C_TEXT, fontName="Helvetica-Bold"),
        "caption": S("DR_Cap", fontSize=8, leading=11, textColor=C_MUTED, fontName="Helvetica-Oblique", spaceAfter=4),
        "table_header":S("DR_TH", fontSize=8, leading=10, textColor=C_WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER),
        "table_cell": S("DR_TC", fontSize=8, leading=11, textColor=C_TEXT, fontName="Helvetica"),
        "table_cell_r":S("DR_TCR", fontSize=8, leading=11, textColor=C_TEXT, fontName="Helvetica", alignment=TA_RIGHT),
        "kpi_value": S("DR_KPIVal", fontSize=20, leading=24, textColor=C_PRIMARY, fontName="Helvetica-Bold", alignment=TA_CENTER),
        "kpi_label": S("DR_KPILbl", fontSize=7, leading=9, textColor=C_MUTED, fontName="Helvetica", alignment=TA_CENTER),
        "kpi_delta": S("DR_KPIDel", fontSize=7, leading=9,textColor=C_MUTED, fontName="Helvetica", alignment=TA_CENTER),
    }

# quality bar
class QualityBar(Flowable):
    def __init__(self, score: float, width: float = CONTENT_W, height: float = 16):
        super().__init__()
        self.score  = max(0.0, min(score, 100.0))
        self.width  = width
        self.height = height

    def draw(self):
        self.canv.setFillColor(colors.HexColor("#E2E8F0"))
        self.canv.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=0)
        fill_w = self.width * self.score / 100.0
        bar_color = C_DANGER if self.score < 40 else (C_WARNING if self.score < 70 else C_SUCCESS)
        self.canv.setFillColor(bar_color)
        self.canv.roundRect(0, 0, max(fill_w, 4), self.height, 4, fill=1, stroke=0)
        self.canv.setFont("Helvetica-Bold", 7)
        self.canv.setFillColor(C_WHITE)
        self.canv.drawCentredString(self.width / 2, 4, f"{self.score:.1f} / 100")

    def wrap(self, *args):
        return self.width, self.height

# 
def _fig_to_image(fig, width: float = CONTENT_W, dpi: int = 150) -> RLImage:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    img = RLImage(buf)
    scale = width / img.drawWidth
    img.drawWidth  = width
    img.drawHeight = img.drawHeight * scale
    return img

def _section_heading(text: str, styles: dict, number: str = "") -> list:
    label = f"{number}. {text}" if number else text
    return [
        Paragraph(label, styles["section"]),
        HRFlowable(width="100%", thickness=1, color=C_RULE, spaceAfter=6),
    ]

def _kpi_table(kpis: list, styles: dict) -> Table:
    cell_w = CONTENT_W / len(kpis)
    max_val_len = max((len(str(v)) for _, v, _ in kpis), default=1)
    if max_val_len <= 6: val_size, val_leading, row_h = 18, 22, 26
    elif max_val_len <= 9: val_size, val_leading, row_h = 13, 17, 22
    elif max_val_len <= 12: val_size, val_leading, row_h = 10, 13, 18
    else: val_size, val_leading, row_h = 8,  11, 16

    val_style = ParagraphStyle(
        "DR_KPIVal_dyn", parent=styles["kpi_value"],
        fontSize=val_size, leading=val_leading,
    )

    header_row = [Paragraph(l, styles["kpi_label"]) for l, _, _ in kpis]
    value_row  = [Paragraph(v, val_style)           for _, v, _ in kpis]
    delta_row  = [Paragraph(d, styles["kpi_delta"]) for _, _, d in kpis]

    tbl = Table([header_row, value_row, delta_row], colWidths=[cell_w]*len(kpis), rowHeights=[14, row_h, 12])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), C_BG_LIGHT),
        ("BOX", (0,0),(-1,-1), 0.5, C_RULE),
        ("LINEAFTER", (0,0),(-2,-1), 0.4, C_RULE),
        ("TOPPADDING", (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0),(-1,-1), 6),
        ("RIGHTPADDING", (0,0),(-1,-1), 6),
        ("ALIGN", (0,0),(-1,-1), "CENTER"),
        ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
    ]))
    return tbl

def _comparison_table(rows: list, styles: dict) -> Table:
    header = [Paragraph(h, styles["table_header"])
              for h in ["Metric", "Before Cleaning", "After Cleaning", "Change"]]
    data = [header]
    for label, before_val, after_val in rows:
        try:
            bv = float(str(before_val).replace("%",""))
            av = float(str(after_val).replace("%",""))
            ch = av - bv
            if ch < 0:
                ch_str, ch_col = f"▼ {abs(ch):g}", C_SUCCESS 
            elif ch > 0:
                ch_str, ch_col = f"▲ {ch:g}", C_DANGER
            else:
                ch_str, ch_col = "—", C_MUTED
        except (ValueError, TypeError):
            ch_str, ch_col = "—", C_MUTED

        data.append([
            Paragraph(str(label), styles["table_cell"]),
            Paragraph(str(before_val), styles["table_cell_r"]),
            Paragraph(str(after_val), styles["table_cell_r"]),
            Paragraph(ch_str, ParagraphStyle("ch", parent=styles["table_cell_r"], textColor=ch_col, fontName="Helvetica-Bold")),
        ])
    col_w = [CONTENT_W * f for f in (0.38, 0.22, 0.22, 0.18)]
    tbl = Table(data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), C_PRIMARY),
        ("TEXTCOLOR", (0,0),(-1,0), C_WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_BG_LIGHT]),
        ("BOX", (0,0),(-1,-1), 0.5, C_RULE),
        ("LINEBELOW", (0,0),(-1,-1), 0.3, C_RULE),
        ("TOPPADDING", (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0),(-1,-1), 6),
        ("RIGHTPADDING", (0,0),(-1,-1), 6),
        ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
        ("ALIGN", (0,0),(-1, 0), "CENTER"),
    ]))
    return tbl

def _schema_table(df: pd.DataFrame, styles: dict, max_cols: int = 20) -> Table:
    header = [Paragraph(h, styles["table_header"])
              for h in ["Column", "Type", "Missing", "Missing %", "Unique Values"]]
    data = [header]
    sample = df.iloc[:, :max_cols]
    for col in sample.columns:
        miss = int(sample[col].isnull().sum())
        miss_p = f"{miss / max(len(sample),1)*100:.1f}%"
        uniq = int(sample[col].nunique(dropna=True))
        data.append([
            Paragraph(str(col)[:30], styles["table_cell"]),
            Paragraph(str(sample[col].dtype), styles["table_cell"]),
            Paragraph(str(miss), styles["table_cell_r"]),
            Paragraph(miss_p, styles["table_cell_r"]),
            Paragraph(str(uniq), styles["table_cell_r"]),
        ])
    col_w = [CONTENT_W * f for f in (0.35, 0.16, 0.13, 0.16, 0.20)]
    tbl = Table(data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), C_PRIMARY),
        ("TEXTCOLOR", (0,0),(-1,0), C_WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_BG_LIGHT]),
        ("BOX", (0,0),(-1,-1), 0.5, C_RULE),
        ("LINEBELOW", (0,0),(-1,-1), 0.3, C_RULE),
        ("TOPPADDING", (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0),(-1,-1), 5),
        ("RIGHTPADDING", (0,0),(-1,-1), 5),
        ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
        ("ALIGN", (0,0),(-1, 0), "CENTER"),
        ("ALIGN", (1,1),(-1,-1), "RIGHT"),
    ]))
    return tbl

def _stats_table(df: pd.DataFrame, styles: dict) -> Table:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty: return None

    headers = ["Column", "Count", "Mean", "Std Dev", "Min", "25%", "50%", "75%", "Max", "Skewness", "Kurtosis"]
    header_row = [Paragraph(h, styles["table_header"]) for h in headers]
    data = [header_row]

    desc = num_df.describe().T
    def fmt(v):
        try:
            fv = float(v)
            if abs(fv) >= 1e6 or (abs(fv) < 0.001 and fv != 0): return f"{fv:.3e}"
            return f"{fv:.4f}"
        except Exception:
            return str(v)

    for col in desc.index:
        s = num_df[col].dropna()
        data.append([
            Paragraph(str(col)[:22], styles["table_cell"]),
            Paragraph(fmt(desc.loc[col, "count"]),styles["table_cell_r"]),
            Paragraph(fmt(desc.loc[col, "mean"]), styles["table_cell_r"]),
            Paragraph(fmt(desc.loc[col, "std"]), styles["table_cell_r"]),
            Paragraph(fmt(desc.loc[col, "min"]), styles["table_cell_r"]),
            Paragraph(fmt(desc.loc[col, "25%"]), styles["table_cell_r"]),
            Paragraph(fmt(desc.loc[col, "50%"]), styles["table_cell_r"]),
            Paragraph(fmt(desc.loc[col, "75%"]), styles["table_cell_r"]),
            Paragraph(fmt(desc.loc[col, "max"]), styles["table_cell_r"]),
            Paragraph(fmt(s.skew()), styles["table_cell_r"]),
            Paragraph(fmt(s.kurt()), styles["table_cell_r"]),
        ])

    col_w = [CONTENT_W * f for f in (0.14, 0.07, 0.09, 0.09, 0.08, 0.08, 0.08, 0.08, 0.08, 0.11, 0.10)]
    tbl = Table(data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), C_PRIMARY),
        ("TEXTCOLOR", (0,0),(-1,0), C_WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_BG_LIGHT]),
        ("BOX", (0,0),(-1,-1), 0.5, C_RULE),
        ("LINEBELOW", (0,0),(-1,-1), 0.3, C_RULE),
        ("TOPPADDING", (0,0),(-1,-1), 3),
        ("BOTTOMPADDING", (0,0),(-1,-1), 3),
        ("LEFTPADDING", (0,0),(-1,-1), 4),
        ("RIGHTPADDING", (0,0),(-1,-1), 4),
        ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
        ("ALIGN", (0,0),(-1, 0), "CENTER"),
        ("ALIGN", (1,1),(-1,-1), "RIGHT"),
    ]))
    return tbl

def _preview_table(df: pd.DataFrame, styles: dict, max_cols: int = 10) -> Table:
    preview = df.head(5).iloc[:, :max_cols].reset_index(drop=True)
    cols = preview.columns.tolist()

    if not cols:
        data = [
            [Paragraph("#", styles["table_header"]), Paragraph("Message", styles["table_header"])],
            [Paragraph("—", styles["table_cell_r"]), Paragraph("No columns available in this dataset state.", styles["table_cell"])],
        ]
        tbl = Table(data, colWidths=[CONTENT_W * 0.12, CONTENT_W * 0.88], repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_BG_LIGHT]),
            ("BOX", (0, 0), (-1, -1), 0.5, C_RULE),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, C_RULE),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("ALIGN", (0, 1), (0, -1), "RIGHT"),
        ]))
        return tbl

    header_row = [Paragraph("#", styles["table_header"])] + [Paragraph(str(c)[:18], styles["table_header"]) for c in cols]
    data = [header_row]

    def _fmt(v):
        try:
            if pd.isna(v): return "—"
        except Exception: pass
        if isinstance(v, float): return f"{v:.4g}"
        return str(v)[:22]

    for row_num, (_, row) in enumerate(preview.iterrows(), start=0):
        data.append([Paragraph(str(row_num), styles["table_cell_r"])] + [Paragraph(_fmt(row[c]), styles["table_cell"]) for c in cols])

    first_w = 0.06
    rest_w  = (1 - first_w) / len(cols)
    col_w   = [CONTENT_W * first_w] + [CONTENT_W * rest_w] * len(cols)

    tbl = Table(data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), C_PRIMARY),
        ("TEXTCOLOR", (0,0),(-1,0), C_WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_BG_LIGHT]),
        ("BOX", (0,0),(-1,-1), 0.5, C_RULE),
        ("LINEBELOW", (0,0),(-1,-1), 0.3, C_RULE),
        ("TOPPADDING", (0,0),(-1,-1), 3),
        ("BOTTOMPADDING", (0,0),(-1,-1), 3),
        ("LEFTPADDING", (0,0),(-1,-1), 4),
        ("RIGHTPADDING", (0,0),(-1,-1), 4),
        ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
        ("ALIGN", (0,0),(-1, 0), "CENTER"),
        ("ALIGN", (0,1),(0,-1), "RIGHT"),
    ]))
    return tbl


def _style_ax(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor("#F8F9FF")
    ax.tick_params(labelsize=7)

def _chart_missing_bar(df: pd.DataFrame):
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty: return None
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.bar(miss.index, miss.values, color="#4F46E5", width=0.6, edgecolor="white", lw=0.4)
    ax.bar_label(bars, fmt="%d", fontsize=7, padding=2)
    ax.set_title("Missing Values per Column — Before Cleaning", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Column", fontsize=8)
    ax.set_ylabel("Count",  fontsize=8)
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    _style_ax(ax)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig

def _chart_before_after(mb: dict, ma: dict):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, vals, title, ylabel, clrs in [
        (axes[0], [mb["rows"], ma["rows"]], "Row Count", "Rows", ["#4F46E5","#22C55E"]),
        (axes[1], [mb["missing"], ma["missing"]], "Missing Values", "Count", ["#F97316","#22C55E"]),
    ]:
        bars = ax.bar(["Before","After"], vals, color=clrs, width=0.5, edgecolor="white", lw=0.4)
        ax.bar_label(bars, labels=[f"{v:,}" for v in vals], fontsize=7, padding=2)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8)
        _style_ax(ax)
    fig.suptitle("Cleaning Impact Summary", fontsize=10, fontweight="bold", y=1.02)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig

def _chart_dist_shift(raw_df: pd.DataFrame, clean_df: pd.DataFrame):
    num_b = raw_df.select_dtypes(include=[np.number])
    num_a = clean_df.select_dtypes(include=[np.number])
    common = [c for c in num_b.columns if c in num_a.columns]
    if not common: return None
    col = common[0]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(num_b[col].dropna(), bins=30, alpha=0.65, color="#4F46E5", label="Before", edgecolor="white", lw=0.3)
    ax.hist(num_a[col].dropna(), bins=30, alpha=0.65, color="#22C55E", label="After",  edgecolor="white", lw=0.3)
    ax.set_title(f"Distribution Shift — {col}", fontsize=9, fontweight="bold", pad=8)
    ax.set_xlabel(col, fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.legend(fontsize=8)
    _style_ax(ax)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig

def _chart_distributions(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    if num.empty: return None
    top = num.var().sort_values(ascending=False).head(4).index.tolist()
    n_cols = len(top)
    fig, axes = plt.subplots(1, n_cols, figsize=(9, 2.8))
    if n_cols == 1: axes = [axes]
    for ax, col, color in zip(axes, top, _PALETTE):
        s = num[col].dropna()
        ax.hist(s, bins=25, color=color, alpha=0.85, edgecolor="white", lw=0.3)
        ax.axvline(s.mean(), color="#1E293B", linestyle="--", lw=0.9, label="Mean")
        ax.axvline(s.median(), color="#94A3B8", linestyle=":",  lw=0.9, label="Median")
        ax.set_title(col[:16], fontsize=7, fontweight="bold")
        ax.set_xlabel("Value", fontsize=6)
        ax.set_ylabel("Count", fontsize=6)
        _style_ax(ax)
    axes[0].legend(fontsize=5, loc="upper right")
    fig.suptitle("Numeric Distributions — After Cleaning", fontsize=9, fontweight="bold")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig

def _chart_corr(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2: return None
    if num.shape[1] > 12: num = num.iloc[:, :12]
    corr = num.corr().round(2)
    n = len(corr)
    sz = max(5, min(8, n * 0.75))
    fig, ax = plt.subplots(figsize=(sz, sz * 0.8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=40, ha="right", fontsize=6)
    ax.set_yticklabels(corr.columns, fontsize=6)
    for i in range(n):
        for j in range(n):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=5, color="white" if abs(v) > 0.5 else "#1E293B")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    ax.set_title("Correlation Heatmap — After Cleaning", fontsize=9, fontweight="bold", pad=8)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig

def _cleaning_summary_table(history: list, styles: dict) -> Table:
    col_w = [CONTENT_W * f for f in (0.05, 0.28, 0.50, 0.17)]
    header_row = [Paragraph(h, styles["table_header"]) for h in ["#", "Operation", "Steps Applied", "Status"]]
    data = [header_row]

    applied_style = ParagraphStyle("ok", parent=styles["table_cell"], textColor=colors.HexColor("#22C55E"), fontName="Helvetica-Bold")
    source_ai_style = ParagraphStyle("src_ai", parent=styles["table_cell"], textColor=colors.HexColor("#4F46E5"), fontName="Helvetica-Bold", fontSize=7)
    source_rule_style = ParagraphStyle("src_rule", parent=styles["table_cell"], textColor=colors.HexColor("#F97316"), fontName="Helvetica-Bold", fontSize=7)
    substep_style = ParagraphStyle("substep", parent=styles["table_cell"], fontSize=7.5, leading=11, textColor=colors.HexColor("#334155"))

    for idx, item in enumerate(history):
        if isinstance(item, (list, tuple)):
            label  = str(item[0])
            steps  = item[2] if len(item) > 2 else []
            source = str(item[3]) if len(item) > 3 else "Manual"
        else:
            label, steps, source = str(item), [], "Manual"

        if "phi3" in source.lower() or "ollama" in source.lower(): src_para = Paragraph(f"Source: {source}", source_ai_style)
        elif "rule" in source.lower(): src_para = Paragraph(f"Source: {source}", source_rule_style)
        else: src_para = Paragraph(f"Source: {source}", styles["caption"])

        op_cell = [Paragraph(label[:45], styles["body_bold"]), src_para]
        steps_paragraphs = [Paragraph(f"• {str(s)[:80]}", substep_style) for s in steps] if steps else [Paragraph("—", styles["table_cell"])]

        data.append([
            Paragraph(str(idx + 1), styles["table_cell_r"]), op_cell, steps_paragraphs, Paragraph("Applied", applied_style),
        ])

    tbl = Table(data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_BG_LIGHT]),
        ("BOX", (0, 0), (-1, -1), 0.5, C_RULE),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, C_RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "RIGHT"),
        ("ALIGN", (3, 1), (3, -1), "CENTER"),
    ]))
    return tbl

def _on_page(canvas_obj, doc, file_name: str):
    canvas_obj.saveState()
    w, h = A4
    canvas_obj.setFillColor(C_PRIMARY)
    canvas_obj.rect(0, h - 8*mm, w, 8*mm, fill=1, stroke=0)
    canvas_obj.setFont("Helvetica-Bold", 8.5)
    canvas_obj.setFillColor(C_WHITE)
    canvas_obj.drawString(MARGIN, h - 5.5*mm, "DataRefine — AI Data Cleaner Report")
    canvas_obj.drawRightString(w - MARGIN, h - 5.5*mm, f"Page {doc.page}")

    canvas_obj.setStrokeColor(C_RULE)
    canvas_obj.setLineWidth(0.5)
    canvas_obj.line(MARGIN, 14*mm, w - MARGIN, 14*mm)
    canvas_obj.setFont("Helvetica", 7)
    canvas_obj.setFillColor(C_MUTED)
    canvas_obj.drawString(MARGIN, 9*mm, f"Source: {file_name}")
    canvas_obj.drawCentredString(w / 2, 9*mm, f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')}")
    canvas_obj.drawRightString(w - MARGIN, 9*mm, "Confidential — DataRefine")
    canvas_obj.restoreState()

# pdf report builder
def build_pdf_report(
    raw_df: Optional[pd.DataFrame],
    clean_df: Optional[pd.DataFrame],
    file_name: str  = "-",
    cleaning_history: list = None,
) -> bytes:
    if raw_df is None: raise ValueError("raw_df is required to build the report.")
    if clean_df is None or clean_df.empty: clean_df = raw_df

    styles = _build_styles()
    mb, ma = compute_basic_metrics(raw_df), compute_basic_metrics(clean_df)
    qb, qa = compute_quality_score(raw_df), compute_quality_score(clean_df)
    improvement = qa - qb
    improvement_pct = (improvement / qb * 100) if qb > 0 else 0.0

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=2.5*cm, bottomMargin=2.2*cm,
        title="DataRefine Report", author="DataRefine AI",
    )
    on_page = lambda c, d: _on_page(c, d, file_name)
    story: list = []

    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("DataRefine", styles["title"]))
    story.append(Paragraph("AI-Powered Data Cleaning Report", styles["subtitle"]))
    story.append(Paragraph(f"File: <b>{file_name}</b> &nbsp;|&nbsp; Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", styles["meta"]))
    story.append(Spacer(1, 0.35*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=C_PRIMARY, spaceAfter=10))

    story += _section_heading("Dataset Summary", styles, "1")
    def _d(b, a):
        d = a - b
        if d < 0: return f"▼ {abs(d):,}"
        if d > 0: return f"▲ {d:,}" 
        return "—"

    story.append(_kpi_table([
        ("Rows (Before)", f"{mb['rows']:,}", ""), ("Rows (After)", f"{ma['rows']:,}", _d(mb['rows'], ma['rows'])),
        ("Columns", f"{mb['cols']}", ""), ("Missing (Before)", f"{mb['missing']:,}", ""),
        ("Missing (After)", f"{ma['missing']:,}", _d(mb['missing'], ma['missing'])),
        ("Duplicates Removed", f"{mb['duplicates']-ma['duplicates']:,}", ""),
    ], styles))
    story.append(Spacer(1, 0.4*cm))

    story += _section_heading("Data Quality Score", styles, "2")
    story.append(_kpi_table([
        ("Score Before", f"{qb:.1f}", "/ 100"), ("Score After", f"{qa:.1f}", "/ 100"),
        ("Improvement", f"{improvement:+.1f}", f"({improvement_pct:+.1f}%)"),
    ], styles))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Quality after cleaning:", styles["body_bold"]))
    story.append(Spacer(1, 2*mm))
    story.append(QualityBar(qa, width=CONTENT_W, height=16))
    story.append(Spacer(1, 0.4*cm))

    story += _section_heading("Before vs After Comparison", styles, "3")
    def _pct(n, d): return f"{round(n/max(d,1)*100, 2):.2f}%"
    story.append(_comparison_table([
        ("Total Rows", mb["rows"], ma["rows"]), ("Total Columns", mb["cols"], ma["cols"]),
        ("Missing Values", mb["missing"], ma["missing"]), ("Duplicate Rows", mb["duplicates"], ma["duplicates"]),
        ("Missing % of Cells", _pct(mb["missing"], mb["rows"]*mb["cols"]), _pct(ma["missing"], ma["rows"]*ma["cols"])),
        ("Duplicate % of Rows", _pct(mb["duplicates"], mb["rows"]), _pct(ma["duplicates"], ma["rows"])),
        ("Data Quality Score", f"{qb:.2f} / 100", f"{qa:.2f} / 100"),
    ], styles))
    story.append(Spacer(1, 0.4*cm))
    
    story += _section_heading("Column Schema (After Cleaning)", styles, "4")
    n_shown = min(clean_df.shape[1], 20)
    story.append(Paragraph(f"Showing {n_shown} of {clean_df.shape[1]} columns.", styles["caption"]))
    story.append(_schema_table(clean_df, styles))
    story.append(Spacer(1, 0.3*cm))

    story += _section_heading("Cleaning Summary", styles, "5")
    history = cleaning_history or []
    if not history: story.append(Paragraph("No cleaning steps were recorded for this session.", styles["body"]))
    else:
        story.append(Paragraph(f"{len(history)} cleaning step(s) applied to the dataset.", styles["caption"]))
        story.append(_cleaning_summary_table(history, styles))
    story.append(Spacer(1, 0.4*cm))

    story += _section_heading("Statistical Summary (After Cleaning)", styles, "6")
    num_cols_count = len(clean_df.select_dtypes(include=[np.number]).columns)
    if num_cols_count == 0: story.append(Paragraph("No numerical columns found in the cleaned dataset.", styles["body"]))
    else:
        story.append(Paragraph(f"Descriptive statistics for all {num_cols_count} numerical column(s) after cleaning.", styles["caption"]))
        stats_tbl = _stats_table(clean_df, styles)
        if stats_tbl is not None: story.append(stats_tbl)
    story.append(Spacer(1, 0.3*cm))

    story += _section_heading("Data Preview — Before & After Cleaning", styles, "7")
    red_label = ParagraphStyle("RedLabel", parent=styles["body_bold"], textColor=colors.HexColor("#EF4444"), fontSize=10, spaceBefore=6, spaceAfter=4)
    green_label = ParagraphStyle("GreenLabel", parent=styles["body_bold"], textColor=colors.HexColor("#22C55E"), fontSize=10, spaceBefore=10, spaceAfter=4)

    raw_cols, clean_cols = raw_df.shape[1], clean_df.shape[1]
    shown_raw, shown_clean = min(raw_cols,10), min(clean_cols,10)

    story.append(Paragraph("Original Data (Before Cleaning)", red_label))
    story.append(Paragraph(f"First 5 rows — showing {shown_raw} of {raw_cols} column(s).", styles["caption"]))
    story.append(_preview_table(raw_df, styles))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("Cleaned Data (After Cleaning)", green_label))
    story.append(Paragraph(f"First 5 rows — showing {shown_clean} of {clean_cols} column(s).", styles["caption"]))
    story.append(_preview_table(clean_df, styles))
    story.append(Spacer(1, 0.3*cm))
    story.append(PageBreak())

    story += _section_heading("Key Visuals", styles, "8")
    charts = [
        ("8.1 — Missingness Overview (Before Cleaning)", _chart_missing_bar(raw_df)),
        ("8.2 — Cleaning Impact (Rows & Missing Values)", _chart_before_after(mb, ma)),
        ("8.3 — Distribution Shift (First Numeric Column)", _chart_dist_shift(raw_df, clean_df)),
        ("8.4 — Numeric Distributions (After Cleaning)", _chart_distributions(clean_df)),
        ("8.5 — Correlation Heatmap (After Cleaning)", _chart_corr(clean_df)),
    ]

    for caption, fig in charts:
        if fig is None: continue
        story.append(KeepTogether([
            Paragraph(caption, styles["body_bold"]), Spacer(1, 2*mm),
            _fig_to_image(fig, CONTENT_W), Spacer(1, 0.45*cm),
        ]))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    buffer.seek(0)
    return buffer.read()