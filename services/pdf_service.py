"""
PDF Service
Handles PDF parsing (extraction) and PDF generation using pypdf and reportlab.
"""

import io
import os
from typing import List

from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT
from langchain.schema import Document


class PDFService:
    def parse_pdf(self, file_bytes: bytes, source_name: str = "uploaded.pdf") -> List[Document]:
        """
        Extract text from a PDF file (as bytes).
        Returns a list of LangChain Document objects (one per page).
        """
        reader = PdfReader(io.BytesIO(file_bytes))
        documents = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": source_name, "page": page_num + 1},
                    )
                )
        return documents

    def get_pdf_metadata(self, file_bytes: bytes) -> dict:
        """Extract metadata from a PDF file."""
        reader = PdfReader(io.BytesIO(file_bytes))
        meta = reader.metadata or {}
        return {
            "total_pages": len(reader.pages),
            "title": meta.get("/Title", ""),
            "author": meta.get("/Author", ""),
            "subject": meta.get("/Subject", ""),
            "creator": meta.get("/Creator", ""),
        }

    def generate_pdf(self, title: str, content: str) -> bytes:
        """
        Generate a PDF from a title and plain text content.
        Returns raw PDF bytes.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=20 * mm,
            leftMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontSize=18,
            spaceAfter=12,
        )
        body_style = ParagraphStyle(
            "CustomBody",
            parent=styles["Normal"],
            fontSize=11,
            leading=16,
            alignment=TA_LEFT,
        )

        story = [
            Paragraph(title, title_style),
            Spacer(1, 6 * mm),
            Paragraph(content.replace("\n", "<br/>"), body_style),
        ]

        doc.build(story)
        buffer.seek(0)
        return buffer.read()
