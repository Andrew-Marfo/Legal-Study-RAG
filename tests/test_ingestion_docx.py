"""Word document ingestion.

Word files carry no page numbers - pagination is produced by whatever renders
the document, not stored in it - so the citable unit is the heading-delimited
section. These tests pin down that contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from src.ingestion import (
    DocumentParseError,
    UnsupportedFileTypeError,
    chunk_units,
    detect_doc_type,
    format_citation,
    ingest_file,
    parse_docx,
)


# --- Parsing ----------------------------------------------------------------


def test_sections_are_split_at_headings(sample_docx: Path):
    units = parse_docx(sample_docx)
    labels = [u.label for u in units]
    assert "Formation of Contract" in labels
    assert "Promissory Estoppel" in labels
    assert "Key Authorities" in labels


def test_sections_are_numbered_from_one_and_contiguous(sample_docx: Path):
    units = parse_docx(sample_docx)
    assert [u.index for u in units] == list(range(1, len(units) + 1))
    assert all(u.doc_type == "docx" for u in units)


def test_the_title_opens_the_first_section(sample_docx: Path):
    units = parse_docx(sample_docx)
    assert units[0].label == "Contract Law Revision Notes"
    assert "core doctrines" in units[0].text


def test_body_text_stays_with_its_heading(sample_docx: Path):
    units = {u.label: u.text for u in parse_docx(sample_docx)}
    assert "intention to create legal relations" in units["Formation of Contract"]
    assert "detriment" in units["Promissory Estoppel"]
    # Content must not bleed across the heading boundary.
    assert "detriment" not in units["Formation of Contract"]


def test_the_heading_is_included_in_its_section_text(sample_docx: Path):
    """So the heading's own wording is embedded and searchable."""
    units = {u.label: u.text for u in parse_docx(sample_docx)}
    assert units["Promissory Estoppel"].startswith("Promissory Estoppel")


def test_tables_are_captured_under_their_heading(sample_docx: Path):
    units = {u.label: u.text for u in parse_docx(sample_docx)}
    authorities = units["Key Authorities"]
    assert "Central London Property v High Trees House" in authorities
    assert "Promissory estoppel in English law" in authorities


def test_a_document_without_headings_still_yields_sections(headingless_docx: Path):
    units = parse_docx(headingless_docx)
    assert units
    assert all(u.label == "" for u in units)
    assert units[0].index == 1
    assert "consideration must be sufficient" in units[0].text


def test_a_document_with_no_text_yields_no_units(empty_docx: Path):
    assert parse_docx(empty_docx) == []


def test_a_corrupt_word_file_reports_rather_than_crashes(tmp_path: Path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"not a zip archive at all")
    with pytest.raises(DocumentParseError, match="broken.docx"):
        parse_docx(path)


# --- Metadata and citations -------------------------------------------------


def test_chunks_carry_section_and_heading_not_page_or_slide(sample_docx: Path):
    chunks = chunk_units(
        parse_docx(sample_docx), source="contract-notes.docx", subject="Contracts"
    )
    assert chunks
    for chunk in chunks:
        meta = chunk.metadata
        assert meta["doc_type"] == "docx"
        assert isinstance(meta["section"], int) and meta["section"] >= 1
        assert "page" not in meta
        assert "slide" not in meta


def test_a_cited_section_names_its_heading(sample_docx: Path):
    chunks = chunk_units(parse_docx(sample_docx), source="contract-notes.docx")
    citations = {c.metadata["citation"] for c in chunks}
    assert "contract-notes.docx - Promissory Estoppel" in citations


def test_an_unheaded_section_is_cited_by_number(headingless_docx: Path):
    chunks = chunk_units(parse_docx(headingless_docx), source="plain-notes.docx")
    assert chunks[0].metadata["citation"] == "plain-notes.docx - section 1"
    assert "heading" not in chunks[0].metadata


def test_citation_property_matches_stored_citation(sample_docx: Path):
    chunks = chunk_units(parse_docx(sample_docx), source="contract-notes.docx")
    for chunk in chunks:
        assert chunk.citation == chunk.metadata["citation"]


@pytest.mark.parametrize(
    "doc_type,index,label,expected",
    [
        ("pdf", 12, "", "notes.pdf - p. 12"),
        ("pptx", 4, "", "notes.pdf - slide 4"),
        ("docx", 3, "Consideration", "notes.pdf - Consideration"),
        ("docx", 3, "", "notes.pdf - section 3"),
    ],
)
def test_format_citation_per_document_type(
    doc_type: str, index: int, label: str, expected: str
):
    assert format_citation("notes.pdf", doc_type, index, label) == expected


# --- File type detection ----------------------------------------------------


@pytest.mark.parametrize("filename", ["notes.docx", "NOTES.DOCX", "Notes.DocX"])
def test_docx_is_detected_case_insensitively(filename: str):
    assert detect_doc_type(filename) == "docx"


def test_docx_is_an_advertised_supported_extension():
    assert ".docx" in config.SUPPORTED_EXTENSIONS


def test_legacy_doc_is_rejected_with_instructions():
    with pytest.raises(UnsupportedFileTypeError, match="Save As"):
        detect_doc_type("old-notes.doc")


def test_legacy_ppt_is_rejected_with_instructions():
    with pytest.raises(UnsupportedFileTypeError, match="Save As"):
        detect_doc_type("old-deck.ppt")


# --- End to end -------------------------------------------------------------


def test_ingest_file_handles_word_documents(sample_docx: Path):
    chunks = ingest_file(sample_docx, subject="Contracts")
    assert chunks
    assert {c.metadata["source"] for c in chunks} == {"contract-notes.docx"}
    assert {c.metadata["doc_type"] for c in chunks} == {"docx"}
    assert all(c.metadata["subject"] == "Contracts" for c in chunks)


def test_word_chunk_ids_are_unique_and_deterministic(sample_docx: Path):
    first = ingest_file(sample_docx)
    second = ingest_file(sample_docx)
    ids = [c.metadata["chunk_id"] for c in first]
    assert len(set(ids)) == len(ids)
    assert ids == [c.metadata["chunk_id"] for c in second]


def test_a_text_free_word_document_reports_that_it_cannot_be_indexed(
    empty_docx: Path,
):
    with pytest.raises(DocumentParseError, match="No extractable text"):
        ingest_file(empty_docx)
