from __future__ import annotations

from io import BytesIO
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def csv_download(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def excel_download(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="Report")
    return buffer.getvalue()


def excel_workbook_download(tables: dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, frame in tables.items():
            frame.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return buffer.getvalue()


def pdf_report_download(
    title: str,
    dataset_name: str,
    sections: dict[str, pd.DataFrame],
    disclaimer: str,
) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=title,
    )
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "ReportCell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7,
        leading=8,
        wordWrap="CJK",
    )
    header_style = ParagraphStyle(
        "ReportHeaderCell",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )
    story = [
        Paragraph(escape(_pdf_safe(title)), styles["Title"]),
        Paragraph(
            escape(
                f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | Dataset: {_pdf_safe(dataset_name)}"
            ),
            styles["Normal"],
        ),
        Spacer(1, 8 * mm),
    ]
    for heading, frame in sections.items():
        story.append(Paragraph(escape(_pdf_safe(heading)), styles["Heading2"]))
        if frame.empty:
            story.append(Paragraph("No data available for this section.", styles["Normal"]))
        else:
            printable = frame.fillna("")
            data = [[
                Paragraph(escape(_pdf_cell(str(column))), header_style)
                for column in printable.columns
            ]]
            data.extend(
                [
                    [Paragraph(escape(_pdf_cell(value)), body_style) for value in row]
                    for row in printable.astype(str).values.tolist()
                ]
            )
            table = Table(
                data,
                colWidths=[document.width / len(printable.columns)] * len(printable.columns),
                repeatRows=1,
                hAlign="LEFT",
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17264a")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#dce2ed")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6fb")]),
                        ("FONTSIZE", (0, 0), (-1, -1), 7),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.append(table)
        story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(escape(_pdf_safe(disclaimer)), styles["Italic"]))
    document.build(story)
    return buffer.getvalue()


def _pdf_cell(value: str, limit: int = 60) -> str:
    value = _pdf_safe(value)
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _pdf_safe(value: str) -> str:
    return str(value).encode("cp1252", "replace").decode("cp1252")
