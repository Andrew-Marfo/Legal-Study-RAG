"""Shared test fixtures.

Sample documents are generated at test time (PyMuPDF for the PDF, python-pptx
for the deck) so the repository carries no binary fixtures and the expected
page/slide content is visible right here in the source.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from pptx import Presentation
from pptx.util import Inches

# Page 1 is deliberately long enough to split into several chunks.
PDF_PAGES: list[str] = [
    (
        "CONTRACT LAW - OFFER AND ACCEPTANCE\n\n"
        "An offer is an expression of willingness to contract on specified terms, "
        "made with the intention that it shall become binding as soon as it is "
        "accepted by the person to whom it is addressed. "
        + (
            "An invitation to treat is merely an indication that a party is willing "
            "to enter negotiations; it is not an offer capable of acceptance. "
            "The display of goods in a shop window is an invitation to treat. "
        )
        * 6
    ),
    (
        "CONSIDERATION\n\n"
        "Consideration must move from the promisee but need not move to the "
        "promisor. Past consideration is generally not good consideration."
    ),
    (
        "PRIVITY OF CONTRACT\n\n"
        "The doctrine of privity provides that a contract cannot confer rights or "
        "impose obligations on any person except the parties to it."
    ),
]

PPTX_SLIDES: list[dict[str, str]] = [
    {
        "title": "Torts: Negligence",
        "body": "Duty of care\nBreach of duty\nCausation\nRemoteness of damage",
        "notes": "Remember the neighbour principle from Donoghue v Stevenson.",
    },
    {
        "title": "Standard of Care",
        "body": "The standard is that of the reasonable person in the circumstances.",
        "notes": "",
    },
]


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A three-page text PDF with known per-page content."""
    path = tmp_path_factory.mktemp("docs") / "contracts.pdf"
    doc = pymupdf.open()
    for body in PDF_PAGES:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(40, 40, 555, 800), body, fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture(scope="session")
def sample_pptx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A two-slide deck with a title, body text, speaker notes and a table."""
    path = tmp_path_factory.mktemp("docs") / "torts-lecture.pptx"
    presentation = Presentation()
    blank_layout = presentation.slide_layouts[6]
    title_body_layout = presentation.slide_layouts[1]

    for spec in PPTX_SLIDES:
        slide = presentation.slides.add_slide(title_body_layout)
        slide.shapes.title.text = spec["title"]
        slide.placeholders[1].text = spec["body"]
        if spec["notes"]:
            slide.notes_slide.notes_text_frame.text = spec["notes"]

    # A third slide whose only content is a table, to prove table extraction.
    table_slide = presentation.slides.add_slide(blank_layout)
    table = table_slide.shapes.add_table(
        rows=2, cols=2, left=Inches(1), top=Inches(1), width=Inches(6), height=Inches(1)
    ).table
    table.cell(0, 0).text = "Case"
    table.cell(0, 1).text = "Principle"
    table.cell(1, 0).text = "Caparo v Dickman"
    table.cell(1, 1).text = "Threefold duty of care test"

    # An intentionally empty slide: it must be skipped, not indexed.
    presentation.slides.add_slide(blank_layout)

    presentation.save(str(path))
    return path


@pytest.fixture(scope="session")
def empty_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A PDF with a page but no extractable text (stands in for a scan)."""
    path = tmp_path_factory.mktemp("docs") / "scanned.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(path))
    doc.close()
    return path


DOCX_SECTIONS: list[dict[str, object]] = [
    {
        "heading": "Formation of Contract",
        "paragraphs": [
            "A contract requires offer, acceptance, consideration and an "
            "intention to create legal relations.",
            "Each element must be present; the absence of any one is fatal to "
            "the formation of a binding agreement.",
        ],
    },
    {
        "heading": "Promissory Estoppel",
        "paragraphs": [
            "Where one party makes a clear promise intended to be relied upon, "
            "and the other party relies on it to their detriment, the promisor "
            "may be estopped from resiling from that promise.",
        ],
    },
]


@pytest.fixture(scope="session")
def sample_docx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A Word document with a title, two headed sections and a table."""
    from docx import Document as WordDocument

    path = tmp_path_factory.mktemp("docs") / "contract-notes.docx"
    document = WordDocument()
    document.add_heading("Contract Law Revision Notes", level=0)  # Title style
    document.add_paragraph(
        "These notes summarise the core doctrines covered in the first term."
    )

    for section in DOCX_SECTIONS:
        document.add_heading(str(section["heading"]), level=1)
        for paragraph in section["paragraphs"]:  # type: ignore[union-attr]
            document.add_paragraph(paragraph)

    # A table under the final heading, to prove tables stay with their section.
    document.add_heading("Key Authorities", level=1)
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Case"
    table.cell(0, 1).text = "Principle"
    table.cell(1, 0).text = "Central London Property v High Trees House"
    table.cell(1, 1).text = "Promissory estoppel in English law"

    document.save(str(path))
    return path


@pytest.fixture(scope="session")
def headingless_docx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A Word document with no headings at all - the fallback path."""
    from docx import Document as WordDocument

    path = tmp_path_factory.mktemp("docs") / "plain-notes.docx"
    document = WordDocument()
    for i in range(6):
        document.add_paragraph(
            f"Paragraph {i + 1}: consideration must be sufficient but need not "
            "be adequate, and the courts will not weigh the bargain."
        )
    document.save(str(path))
    return path


@pytest.fixture(scope="session")
def empty_docx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A Word document containing no text."""
    from docx import Document as WordDocument

    path = tmp_path_factory.mktemp("docs") / "blank.docx"
    WordDocument().save(str(path))
    return path
