from __future__ import annotations

from io import BytesIO
from typing import Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_pdf_report(raw_df: pd.DataFrame, clean_df: pd.DataFrame, history: list, file_name: str = "dataset.csv") -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph("DataRefine Report", styles["Title"]))
    story.append(Paragraph(f"File: {file_name}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    rows_before = len(raw_df)
    rows_after = len(clean_df)
    missing_before = int(raw_df.isna().sum().sum())
    missing_after = int(clean_df.isna().sum().sum())
    duplicate_before = int(raw_df.duplicated().sum())
    duplicate_after = int(clean_df.duplicated().sum())

    data = [
        ["Metric", "Before", "After"],
        ["Rows", str(rows_before), str(rows_after)],
        ["Missing values", str(missing_before), str(missing_after)],
        ["Duplicate rows", str(duplicate_before), str(duplicate_after)],
        ["Columns", str(len(raw_df.columns)), str(len(clean_df.columns))],
    ]
    table = Table(data, colWidths=[180, 130, 130])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.8, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Cleaning history", styles["Heading2"]))
    if not history:
        story.append(Paragraph("No cleaning steps were recorded.", styles["BodyText"]))
    else:
        for idx, item in enumerate(history[:10], start=1):
            label = item.get("label", "Operation") if isinstance(item, dict) else str(item)
            story.append(Paragraph(f"{idx}. {label}", styles["BodyText"]))

    doc.build(story)
    return buffer.getvalue()
