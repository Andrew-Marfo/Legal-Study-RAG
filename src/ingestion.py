"""Document ingestion: parse PDFs and PowerPoint decks into metadata-rich chunks.

The unit of retrieval is a :class:`Chunk`: a span of text small enough to embed
well, carrying the provenance a law student needs to verify it - which file it
came from, and which page (PDF) or slide (PPTX).

Pipeline::

    file -> parse_pdf / parse_pptx -> [DocumentUnit]  (one per page/slide)
         -> clean_text -> RecursiveCharacterTextSplitter -> [Chunk]
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import pymupdf
from docx import Document as WordDocument
from docx.oxml.ns import qn
from docx.table import Table as WordTable
from docx.text.paragraph import Paragraph as WordParagraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pptx import Presentation

from src import config

DocType = Literal["pdf", "pptx", "docx"]

# What one unit of a document is called, per format. PDFs and decks have a
# fixed, visible location. Word does not: pagination is decided by whatever
# renders the file, so python-docx cannot report a page number. Headings are
# the stable anchor a reader can actually search for, so they are the unit.
LOCATION_KEYS: dict[str, str] = {"pdf": "page", "pptx": "slide", "docx": "section"}

# Close an unlabelled Word section after this many characters, so a document
# with no headings still yields section numbers that mean something.
DOCX_SECTION_CHAR_BUDGET = 4000

# Stable namespace so the same (source, location, index) always yields the same
# point id. Re-indexing an unchanged document overwrites rather than duplicates.
_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


class UnsupportedFileTypeError(ValueError):
    """Raised for a file extension outside :data:`config.SUPPORTED_EXTENSIONS`."""


class DocumentParseError(RuntimeError):
    """Raised when a file cannot be opened or read as its declared type."""


@dataclass(frozen=True)
class DocumentUnit:
    """One page of a PDF, one slide of a deck, or one section of a Word file."""

    text: str
    index: int  # 1-based page, slide or section number
    doc_type: DocType
    label: str = ""  # Word heading text, when the section has one

    @property
    def location_key(self) -> str:
        return LOCATION_KEYS[self.doc_type]


def format_citation(
    source: str, doc_type: str, index: int, label: str = ""
) -> str:
    """The human-readable source reference shown under an answer."""
    if doc_type == "pdf":
        return f"{source} - p. {index}"
    if doc_type == "pptx":
        return f"{source} - slide {index}"
    if label:
        return f"{source} - {label}"
    return f"{source} - section {index}"


@dataclass(frozen=True)
class Chunk:
    """An embeddable span of text plus the provenance needed to cite it."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return str(self.metadata["chunk_id"])

    @property
    def citation(self) -> str:
        """Human-readable source reference, e.g. ``contracts.pdf - p. 12``."""
        stored = self.metadata.get("citation")
        if stored:
            return str(stored)

        source = str(self.metadata.get("source", "unknown source"))
        page = self.metadata.get("page")
        slide = self.metadata.get("slide")
        section = self.metadata.get("section")
        if page is not None:
            return format_citation(source, "pdf", int(page))
        if slide is not None:
            return format_citation(source, "pptx", int(slide))
        if section is not None:
            return format_citation(
                source, "docx", int(section), str(self.metadata.get("heading", ""))
            )
        return source


# --- Text cleaning ----------------------------------------------------------

# "contrac-\ntual" -> "contractual". Common in justified legal PDFs.
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
# Collapse runs of 3+ newlines down to a paragraph break.
_EXCESS_NEWLINES = re.compile(r"\n{3,}")
# Runs of spaces/tabs, but never newlines.
_EXCESS_SPACES = re.compile(r"[ \t ]{2,}")
# Soft hyphen and zero-width characters that PyMuPDF sometimes surfaces.
_INVISIBLES = re.compile(r"[­​‌‍﻿]")


def clean_text(text: str) -> str:
    """Normalise extracted text without destroying paragraph structure."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLES.sub("", text)
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _EXCESS_SPACES.sub(" ", text)
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


# --- Parsers ----------------------------------------------------------------


def parse_pdf(path: str | Path) -> list[DocumentUnit]:
    """Extract one :class:`DocumentUnit` per non-empty PDF page (1-based)."""
    path = Path(path)
    units: list[DocumentUnit] = []
    try:
        with pymupdf.open(path) as doc:
            for page_number, page in enumerate(doc, start=1):
                text = clean_text(page.get_text("text"))
                if text:
                    units.append(
                        DocumentUnit(text=text, index=page_number, doc_type="pdf")
                    )
    except Exception as exc:  # PyMuPDF raises a broad family of errors
        raise DocumentParseError(f"Could not read PDF {path.name!r}: {exc}") from exc
    return units


def _shape_text(shape: Any) -> list[str]:
    """Recursively pull text out of a shape, including groups and tables."""
    parts: list[str] = []

    # Grouped shapes nest arbitrarily deep.
    if hasattr(shape, "shapes"):
        for child in shape.shapes:
            parts.extend(_shape_text(child))
        return parts

    if getattr(shape, "has_text_frame", False):
        for paragraph in shape.text_frame.paragraphs:
            line = "".join(run.text for run in paragraph.runs).strip()
            if line:
                parts.append(line)

    if getattr(shape, "has_table", False):
        for row in shape.table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            line = " | ".join(c for c in cells if c)
            if line:
                parts.append(line)

    return parts


def _notes_text(slide: Any) -> str:
    """Speaker notes for a slide, or an empty string."""
    try:
        if not slide.has_notes_slide:
            return ""
        notes_frame = slide.notes_slide.notes_text_frame
        return (notes_frame.text or "").strip() if notes_frame is not None else ""
    except Exception:
        # A malformed notes part should never sink the whole deck.
        return ""


def parse_pptx(path: str | Path) -> list[DocumentUnit]:
    """Extract one :class:`DocumentUnit` per non-empty slide (1-based).

    Slide body text, tables and speaker notes are all captured; notes are
    appended under a ``Speaker notes:`` heading so the LLM can tell them apart.
    """
    path = Path(path)
    units: list[DocumentUnit] = []
    try:
        presentation = Presentation(str(path))
        for slide_number, slide in enumerate(presentation.slides, start=1):
            parts: list[str] = []
            for shape in slide.shapes:
                parts.extend(_shape_text(shape))
            body = "\n".join(parts)

            notes = _notes_text(slide)
            if notes:
                body = (
                    f"{body}\n\nSpeaker notes: {notes}"
                    if body
                    else f"Speaker notes: {notes}"
                )

            text = clean_text(body)
            if text:
                units.append(
                    DocumentUnit(text=text, index=slide_number, doc_type="pptx")
                )
    except Exception as exc:
        raise DocumentParseError(f"Could not read PPTX {path.name!r}: {exc}") from exc
    return units


def _iter_block_items(document):
    """Yield a Word document's paragraphs and tables in reading order.

    ``document.paragraphs`` and ``document.tables`` are separate collections,
    so neither preserves the order content appears in. Walking the body XML
    does, which is what keeps a table attached to the heading above it.
    """
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield WordParagraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield WordTable(child, document)


def _heading_text(paragraph: Any) -> str | None:
    """The paragraph's text when it is a heading, else None."""
    style = paragraph.style
    name = (getattr(style, "name", "") or "") if style is not None else ""
    is_heading = name == "Title" or name.startswith("Heading")
    if not is_heading:
        return None
    text = (paragraph.text or "").strip()
    return text or None


def _table_lines(table: Any) -> list[str]:
    lines: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        line = " | ".join(c for c in cells if c)
        if line:
            lines.append(line)
    return lines


def parse_docx(path: str | Path) -> list[DocumentUnit]:
    """Extract one :class:`DocumentUnit` per section of a Word document.

    A section runs from one heading to the next. Word has no page numbers to
    cite - they are produced by the renderer, not stored in the file - so the
    heading is the anchor, and an unheaded run of text is closed off by a
    character budget so its section number still localises the quote.
    """
    path = Path(path)
    try:
        document = WordDocument(str(path))
    except Exception as exc:
        raise DocumentParseError(f"Could not read Word file {path.name!r}: {exc}") from exc

    units: list[DocumentUnit] = []
    label = ""
    lines: list[str] = []

    def flush() -> None:
        nonlocal lines
        text = clean_text("\n".join(lines))
        if text:
            units.append(
                DocumentUnit(
                    text=text, index=len(units) + 1, doc_type="docx", label=label
                )
            )
        lines = []

    try:
        for block in _iter_block_items(document):
            if isinstance(block, WordTable):
                lines.extend(_table_lines(block))
                continue

            heading = _heading_text(block)
            if heading is not None:
                flush()
                label = heading
                # Keep the heading in the body text so it is embedded with the
                # content it introduces and can be matched by a query.
                lines = [heading]
                continue

            text = (block.text or "").strip()
            if text:
                lines.append(text)

            if sum(len(line) for line in lines) >= DOCX_SECTION_CHAR_BUDGET:
                flush()
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(f"Could not read Word file {path.name!r}: {exc}") from exc

    flush()
    return units


# --- Chunking ---------------------------------------------------------------


def _build_splitter(
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Splitter tuned for dense legal prose.

    The separator list prefers paragraph, then line, then sentence boundaries,
    so a holding or definition is far less likely to be cut mid-thought.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator=True,
        length_function=len,
    )


def _make_chunk_id(source: str, location_key: str, location: int, index: int) -> str:
    seed = f"{source}|{location_key}:{location}|{index}"
    return str(uuid.uuid5(_ID_NAMESPACE, seed))


def chunk_units(
    units: Iterable[DocumentUnit],
    *,
    source: str,
    subject: str = "General",
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    min_chars: int = config.MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Split document units into chunks, attaching provenance to each.

    Every chunk carries ``source``, ``doc_type``, ``subject``, ``chunk_index``,
    ``chunk_id``, ``citation`` and exactly one of ``page`` / ``slide``.
    """
    splitter = _build_splitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    running_index = 0

    for unit in units:
        for piece in splitter.split_text(unit.text):
            text = piece.strip()
            if len(text) < min_chars:
                continue

            citation = format_citation(
                source, unit.doc_type, unit.index, unit.label
            )
            metadata: dict[str, Any] = {
                "source": source,
                "doc_type": unit.doc_type,
                "subject": subject,
                unit.location_key: unit.index,
                "chunk_index": running_index,
                "chunk_id": _make_chunk_id(
                    source, unit.location_key, unit.index, running_index
                ),
                "citation": citation,
                "n_chars": len(text),
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            }
            if unit.label:
                metadata["heading"] = unit.label

            chunks.append(Chunk(text=text, metadata=metadata))
            running_index += 1

    return chunks


# --- Orchestration ----------------------------------------------------------


def detect_doc_type(filename: str) -> DocType:
    """Map a filename to its parser, or raise :class:`UnsupportedFileTypeError`."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".pptx":
        return "pptx"
    if suffix == ".docx":
        return "docx"

    # The legacy binary formats are a different file format entirely, not a
    # variant of the modern one, so say what to do rather than just refusing.
    legacy = {".doc": "Word", ".ppt": "PowerPoint"}
    if suffix in legacy:
        app = legacy[suffix]
        modern = ".docx" if suffix == ".doc" else ".pptx"
        raise UnsupportedFileTypeError(
            f"{filename!r} is in the old {app} format, which cannot be read. "
            f'Open it in {app} and use "Save As" to save a {modern} copy, '
            "then upload that."
        )

    supported = ", ".join(config.SUPPORTED_EXTENSIONS)
    raise UnsupportedFileTypeError(
        f"{filename!r} is not a supported file type. Supported: {supported}."
    )


def ingest_file(
    path: str | Path,
    *,
    subject: str = "General",
    source_name: str | None = None,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Parse and chunk a single file.

    ``source_name`` overrides the filename recorded in metadata - needed when
    the file on disk is a temporary copy of a browser upload.
    """
    path = Path(path)
    source = source_name or path.name
    doc_type = detect_doc_type(source)

    parsers = {"pdf": parse_pdf, "pptx": parse_pptx, "docx": parse_docx}
    units = parsers[doc_type](path)
    if not units:
        raise DocumentParseError(
            f"No extractable text found in {source!r}. If it is a scanned PDF, "
            "it needs OCR before it can be indexed."
        )

    return chunk_units(
        units,
        source=source,
        subject=subject,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
