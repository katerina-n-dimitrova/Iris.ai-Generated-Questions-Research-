#!/usr/bin/env python3
"""Create a concise provenance and sample PDF for the Yettel benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "yettel_bg"
OUTPUT = ROOT / "report" / "yettel_dataset_creation_and_sample.pdf"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def first_jsonl(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


def esc(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Arial", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(
        18 * mm, 10 * mm, "Yettel Bulgaria synthetic RAG benchmark — provenance summary"
    )
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def build() -> None:
    pdfmetrics.registerFont(TTFont("Arial", FONT))
    pdfmetrics.registerFont(TTFont("Arial-Bold", FONT_BOLD))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    q_manifest = json.loads(
        (DATA / "question_manifest.json").read_text(encoding="utf-8")
    )
    document = first_jsonl(DATA / "documents.jsonl")
    chunk = first_jsonl(DATA / "chunks_1024.jsonl")
    query = first_jsonl(DATA / "questions.jsonl")

    base = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title",
        parent=base["Title"],
        fontName="Arial-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#111827"),
        alignment=TA_CENTER,
        spaceAfter=8 * mm,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontName="Arial-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#5B21B6"),
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
    )
    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="Arial",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=2.2 * mm,
    )
    small = ParagraphStyle("Small", parent=body, fontSize=8.2, leading=11)
    label = ParagraphStyle(
        "Label",
        parent=small,
        fontName="Arial-Bold",
        textColor=colors.HexColor("#374151"),
    )
    note = ParagraphStyle(
        "Note",
        parent=small,
        backColor=colors.HexColor("#F3F4F6"),
        borderColor=colors.HexColor("#D1D5DB"),
        borderWidth=0.5,
        borderPadding=7,
        spaceBefore=3 * mm,
        spaceAfter=3 * mm,
    )

    document_pdf = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=17 * mm,
        title="How the Yettel Bulgaria synthetic RAG dataset was created",
        author="RAG research project",
    )
    frame = Frame(
        document_pdf.leftMargin,
        document_pdf.bottomMargin,
        document_pdf.width,
        document_pdf.height,
        id="normal",
    )
    document_pdf.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=footer))

    story = [
        Paragraph("How the Yettel Bulgaria dataset was created", title),
        Paragraph(
            "<b>Short answer:</b> We built a synthetic Bulgarian multi-hop RAG benchmark "
            "from publicly available Yettel Bulgaria corporate web pages. The documents "
            "are real public prose; the evaluation questions, answers, and gold evidence "
            "links were generated with GPT-5.4-mini and then automatically validated.",
            body,
        ),
        Paragraph("Simple pipeline", h2),
    ]

    steps = [
        "1. <b>Discover:</b> read the public Yettel Bulgaria corporate sitemap and download each listed HTML page.",
        "2. <b>Clean:</b> extract the main article prose; remove navigation, headers, footers, tables, figures, forms, media, and scripts.",
        "3. <b>Filter:</b> retain unique Bulgarian pages containing 1,500–5,000 cl100k_base tokens, then deterministically select 340 documents.",
        "4. <b>Chunk:</b> split each document into 1,024-token chunks with 128-token overlap, producing 963 chunks with stable IDs.",
        "5. <b>Create evaluation QA:</b> group related documents and ask GPT-5.4-mini to create Bulgarian inference, comparison, temporal, and unanswerable questions.",
        "6. <b>Validate and label:</b> require verbatim evidence spans, map each span to canonical document/chunk IDs, reject invalid or duplicate outputs, and save the gold labels.",
    ]
    for step in steps:
        story.append(Paragraph(step, body))

    stats = [
        [
            Paragraph("Documents", label),
            Paragraph(f"{manifest['document_count']:,}", body),
            Paragraph("Chunks", label),
            Paragraph(f"{manifest['chunk_count']:,}", body),
        ],
        [
            Paragraph("Questions", label),
            Paragraph(f"{q_manifest['question_count']:,}", body),
            Paragraph("Evidence-bearing", label),
            Paragraph(
                f"{q_manifest['question_count'] - q_manifest['question_types']['null_query']:,}",
                body,
            ),
        ],
        [
            Paragraph("Null questions", label),
            Paragraph(f"{q_manifest['question_types']['null_query']:,}", body),
            Paragraph("Language", label),
            Paragraph("Bulgarian", body),
        ],
    ]
    stats_table = Table(stats, colWidths=[31 * mm, 36 * mm, 34 * mm, 52 * mm])
    stats_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFAFA")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend(
        [
            Spacer(1, 2 * mm),
            stats_table,
            Paragraph(
                "<b>Important:</b> This is a research dataset created from Yettel's public pages; "
                "it is not an official dataset released or endorsed by Yettel. The date field is "
                "the sitemap's <i>last-modified</i> value, not necessarily the publication date.",
                note,
            ),
            Paragraph("Small sample of the dataset", h2),
        ]
    )

    sample_rows = [
        [
            Paragraph("Document ID", label),
            Paragraph(esc(document["document_id"]), small),
        ],
        [Paragraph("Title", label), Paragraph(esc(document["title"]), small)],
        [
            Paragraph("Category / language", label),
            Paragraph(
                f"{esc(document['category'])} / {esc(document['language'])}", small
            ),
        ],
        [Paragraph("Source URL", label), Paragraph(esc(document["url"]), small)],
        [
            Paragraph("Length", label),
            Paragraph(f"{document['token_count']:,} tokens", small),
        ],
        [
            Paragraph("Body excerpt", label),
            Paragraph(esc(document["body"][:620]) + "…", small),
        ],
    ]
    sample_table = Table(sample_rows, colWidths=[33 * mm, 120 * mm], repeatRows=0)
    sample_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F4F6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([sample_table, Spacer(1, 3 * mm)])

    chunk_rows = [
        [Paragraph("Chunk ID", label), Paragraph(esc(chunk["chunk_id"]), small)],
        [
            Paragraph("Token range", label),
            Paragraph(
                f"{chunk['token_start']}–{chunk['token_end']} ({chunk['token_count']} tokens)",
                small,
            ),
        ],
        [
            Paragraph("Chunk excerpt", label),
            Paragraph(esc(chunk["text"][:420]) + "…", small),
        ],
    ]
    chunk_table = Table(chunk_rows, colWidths=[33 * mm, 120 * mm])
    chunk_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F4F6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([chunk_table, Spacer(1, 3 * mm)])

    evidence = "<br/>".join(
        f"• {esc(cid)} — “{esc(fact)}”"
        for cid, fact in zip(query["gold_chunk_ids"], query["gold_evidence"])
    )
    qa_rows = [
        [
            Paragraph("Query ID / type", label),
            Paragraph(
                f"{esc(query['query_id'])} / {esc(query['question_type'])}", small
            ),
        ],
        [Paragraph("Question", label), Paragraph(esc(query["query"]), small)],
        [Paragraph("Answer", label), Paragraph(esc(query["answer"]), small)],
        [Paragraph("Gold evidence", label), Paragraph(evidence, small)],
    ]
    qa_table = Table(qa_rows, colWidths=[33 * mm, 120 * mm])
    qa_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F4F6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend(
        [
            KeepTogether([Paragraph("Example evaluation record", h2), qa_table]),
            Paragraph(
                "The PDF shows shortened excerpts for readability. The machine-readable files "
                "retain the full document text, complete chunks, metadata, answers, and evidence mappings.",
                note,
            ),
        ]
    )
    document_pdf.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
