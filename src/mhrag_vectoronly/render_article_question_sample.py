"""Render one full-corpus adaptive-question example as a readable PDF."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
URL = "https://www.foxnews.com/lifestyle/5-biggest-mistakes-parents-make-christmastime-parenting-expert"
OUTPUT = ROOT / "report" / "mhrag_adaptive_question_article_sample.pdf"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(18 * mm, 10 * mm, "MultiHop-RAG adaptive-question example")
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def main() -> None:
    corpus = json.loads((ROOT / "data/raw/multihoprag/corpus.json").read_text())
    article = next(row for row in corpus if row["url"] == URL)
    chunks = [
        row
        for row in read_jsonl(
            ROOT / "data/processed/mhrag_adaptive_questions_full/chunks.jsonl"
        )
        if row["document_id"] == URL
    ]
    generations = {
        row["chunk_id"]: row
        for row in read_jsonl(
            ROOT
            / "data/processed/mhrag_adaptive_questions_full/adaptive_generations.jsonl"
        )
    }

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "TitleCenter",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#172033"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#475569"),
        )
    )
    styles.add(
        ParagraphStyle(
            "Question",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=13,
            leftIndent=8,
            bulletIndent=0,
            spaceAfter=5,
        )
    )
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="MultiHop-RAG adaptive generated questions — article example",
    )
    story = [
        Paragraph(
            "Adaptive generated questions: one article example", styles["TitleCenter"]
        ),
        Paragraph(f"<b>{escape(article.get('title', ''))}</b>", styles["Heading2"]),
        Paragraph(f"Source: {escape(URL)}", styles["Small"]),
        Spacer(1, 5 * mm),
        Paragraph("Original article text", styles["Heading2"]),
    ]
    for paragraph in article.get("body", "").split("\n"):
        if paragraph.strip():
            story.extend(
                [
                    Paragraph(escape(paragraph.strip()), styles["BodyText"]),
                    Spacer(1, 2 * mm),
                ]
            )

    for chunk in chunks:
        generation = generations[chunk["chunk_id"]]
        story.extend(
            [
                PageBreak(),
                Paragraph(
                    f"Chunk {escape(chunk['chunk_id'])}: extracted atomic facts",
                    styles["Heading2"],
                ),
                Paragraph(
                    f"{chunk['n_tokens']} tokens. The LLM extracted {len(generation['facts'])} "
                    "deduplicated facts and assigned importance and distinctiveness scores.",
                    styles["Small"],
                ),
                Spacer(1, 3 * mm),
            ]
        )
        fact_rows = [
            [
                Paragraph("ID", styles["Small"]),
                Paragraph("Atomic fact", styles["Small"]),
                Paragraph("Imp.", styles["Small"]),
                Paragraph("Dist.", styles["Small"]),
            ]
        ]
        for index, fact in enumerate(generation["facts"]):
            fact_rows.append(
                [
                    str(index),
                    Paragraph(escape(fact["fact"]), styles["Small"]),
                    str(fact["importance"]),
                    str(fact["distinctiveness"]),
                ]
            )
        table = Table(
            fact_rows, colWidths=[10 * mm, 132 * mm, 13 * mm, 13 * mm], repeatRows=1
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#94a3b8")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("ALIGN", (-2, 1), (-1, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.extend([table, Spacer(1, 5 * mm)])
        for label, field in (
            ("Bounded 5–20 questions", "bounded_questions"),
            ("Unbounded questions", "unbounded_questions"),
        ):
            story.append(Paragraph(label, styles["Heading2"]))
            for number, question in enumerate(generation[field], 1):
                ids = ", ".join(map(str, question["source_fact_ids"]))
                story.append(
                    Paragraph(
                        f"{number}. {escape(question['question'])} "
                        f"<font color='#64748b'>(source fact IDs: {ids})</font>",
                        styles["Question"],
                    )
                )

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
