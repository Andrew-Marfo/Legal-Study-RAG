"""Tests for parsing, chunking and metadata provenance."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from src.ingestion import (
    Chunk,
    DocumentParseError,
    DocumentUnit,
    UnsupportedFileTypeError,
    chunk_units,
    clean_text,
    detect_doc_type,
    ingest_file,
    parse_pdf,
    parse_pptx,
)


# --- clean_text -------------------------------------------------------------


def test_clean_text_rejoins_hyphenated_line_breaks():
    assert clean_text("contrac-\ntual obligation") == "contractual obligation"


def test_clean_text_collapses_runs_of_spaces_but_keeps_paragraphs():
    assert clean_text("a     b\n\nc") == "a b\n\nc"


def test_clean_text_collapses_excessive_blank_lines():
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_clean_text_strips_invisible_characters():
    assert clean_text("fore​see­able") == "foreseeable"


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""
    assert clean_text("   \n  \n ") == ""


# --- PDF parsing ------------------------------------------------------------


def test_parse_pdf_returns_one_unit_per_page_numbered_from_one(sample_pdf: Path):
    units = parse_pdf(sample_pdf)
    assert len(units) == 3
    assert [u.index for u in units] == [1, 2, 3]
    assert all(u.doc_type == "pdf" for u in units)


def test_parse_pdf_keeps_page_content_with_its_page(sample_pdf: Path):
    units = parse_pdf(sample_pdf)
    assert "OFFER AND ACCEPTANCE" in units[0].text
    assert "CONSIDERATION" in units[1].text
    assert "PRIVITY" in units[2].text


def test_parse_pdf_skips_pages_without_text(empty_pdf: Path):
    assert parse_pdf(empty_pdf) == []


def test_parse_pdf_raises_a_clear_error_for_a_non_pdf(tmp_path: Path):
    bogus = tmp_path / "not-really.pdf"
    bogus.write_text("this is plain text, not a PDF", encoding="utf-8")
    with pytest.raises(DocumentParseError, match="not-really.pdf"):
        parse_pdf(bogus)


# --- PPTX parsing -----------------------------------------------------------


def test_parse_pptx_numbers_slides_from_one_and_skips_empty_slides(sample_pptx: Path):
    units = parse_pptx(sample_pptx)
    # Four slides exist; the fourth is blank and must be dropped.
    assert [u.index for u in units] == [1, 2, 3]
    assert all(u.doc_type == "pptx" for u in units)


def test_parse_pptx_captures_title_and_body(sample_pptx: Path):
    units = parse_pptx(sample_pptx)
    assert "Torts: Negligence" in units[0].text
    assert "Causation" in units[0].text


def test_parse_pptx_captures_speaker_notes(sample_pptx: Path):
    units = parse_pptx(sample_pptx)
    assert "Speaker notes:" in units[0].text
    assert "Donoghue v Stevenson" in units[0].text


def test_parse_pptx_captures_table_cells(sample_pptx: Path):
    units = parse_pptx(sample_pptx)
    table_slide = units[2]
    assert "Caparo v Dickman" in table_slide.text
    assert "Threefold duty of care test" in table_slide.text


def test_parse_pptx_raises_a_clear_error_for_a_non_pptx(tmp_path: Path):
    bogus = tmp_path / "broken.pptx"
    bogus.write_bytes(b"not a zip archive")
    with pytest.raises(DocumentParseError, match="broken.pptx"):
        parse_pptx(bogus)


# --- Chunking ---------------------------------------------------------------


def _units(*texts: str, doc_type: str = "pdf") -> list[DocumentUnit]:
    return [
        DocumentUnit(text=text, index=i, doc_type=doc_type)  # type: ignore[arg-type]
        for i, text in enumerate(texts, start=1)
    ]


def test_every_chunk_carries_the_required_metadata(sample_pdf: Path):
    chunks = chunk_units(parse_pdf(sample_pdf), source="contracts.pdf", subject="Contracts")
    assert chunks
    for chunk in chunks:
        meta = chunk.metadata
        assert meta["source"] == "contracts.pdf"
        assert meta["subject"] == "Contracts"
        assert meta["doc_type"] == "pdf"
        assert isinstance(meta["page"], int) and meta["page"] >= 1
        assert "slide" not in meta
        assert isinstance(meta["chunk_index"], int)
        assert meta["chunk_id"]
        assert meta["citation"].startswith("contracts.pdf - p. ")
        assert meta["n_chars"] == len(chunk.text)


def test_pptx_chunks_carry_slide_not_page(sample_pptx: Path):
    chunks = chunk_units(parse_pptx(sample_pptx), source="torts.pptx", subject="Torts")
    assert chunks
    for chunk in chunks:
        assert "slide" in chunk.metadata
        assert "page" not in chunk.metadata
        assert chunk.metadata["citation"].startswith("torts.pptx - slide ")


def test_chunk_index_is_contiguous_from_zero_across_the_document(sample_pdf: Path):
    chunks = chunk_units(parse_pdf(sample_pdf), source="contracts.pdf")
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_unique_within_a_document(sample_pdf: Path):
    chunks = chunk_units(parse_pdf(sample_pdf), source="contracts.pdf")
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(set(ids)) == len(ids)


def test_chunk_ids_are_deterministic_across_runs(sample_pdf: Path):
    first = chunk_units(parse_pdf(sample_pdf), source="contracts.pdf")
    second = chunk_units(parse_pdf(sample_pdf), source="contracts.pdf")
    assert [c.metadata["chunk_id"] for c in first] == [
        c.metadata["chunk_id"] for c in second
    ]


def test_chunk_ids_differ_between_sources(sample_pdf: Path):
    a = chunk_units(parse_pdf(sample_pdf), source="contracts.pdf")
    b = chunk_units(parse_pdf(sample_pdf), source="torts.pdf")
    assert {c.metadata["chunk_id"] for c in a}.isdisjoint(
        {c.metadata["chunk_id"] for c in b}
    )


def test_a_long_page_splits_into_several_overlapping_chunks(sample_pdf: Path):
    page_one = parse_pdf(sample_pdf)[:1]
    chunks = chunk_units(page_one, source="contracts.pdf")
    assert len(chunks) > 1
    # Every chunk from one page keeps that page number.
    assert {c.metadata["page"] for c in chunks} == {1}


def test_chunks_respect_the_configured_size_budget():
    long_text = "The reasonable person standard applies here. " * 200
    chunks = chunk_units(_units(long_text), source="x.pdf", chunk_size=400, chunk_overlap=80)
    assert len(chunks) > 1
    # Allow a small overshoot: the splitter will not break an atomic token.
    assert max(len(c.text) for c in chunks) <= 400 + 50


def test_consecutive_chunks_share_overlapping_text():
    long_text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa. " * 60
    chunks = chunk_units(_units(long_text), source="x.pdf", chunk_size=400, chunk_overlap=120)
    assert len(chunks) >= 2
    tail = chunks[0].text[-60:]
    assert any(word in chunks[1].text for word in tail.split() if len(word) > 3)


def test_chunks_below_the_minimum_length_are_dropped():
    chunks = chunk_units(_units("Too short."), source="x.pdf")
    assert chunks == []


def test_subject_defaults_to_general():
    chunks = chunk_units(_units("x" * 300), source="x.pdf")
    assert chunks[0].metadata["subject"] == "General"


def test_chunk_citation_property_formats_page_and_slide():
    pdf_chunk = Chunk(text="t", metadata={"source": "a.pdf", "page": 7})
    pptx_chunk = Chunk(text="t", metadata={"source": "b.pptx", "slide": 3})
    bare_chunk = Chunk(text="t", metadata={"source": "c.pdf"})
    assert pdf_chunk.citation == "a.pdf - p. 7"
    assert pptx_chunk.citation == "b.pptx - slide 3"
    assert bare_chunk.citation == "c.pdf"


# --- detect_doc_type --------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("notes.pdf", "pdf"),
        ("NOTES.PDF", "pdf"),
        ("lecture.pptx", "pptx"),
        ("Lecture.PPTX", "pptx"),
    ],
)
def test_detect_doc_type_is_case_insensitive(filename: str, expected: str):
    assert detect_doc_type(filename) == expected


@pytest.mark.parametrize("filename", ["notes.docx", "notes.ppt", "notes.txt", "notes"])
def test_detect_doc_type_rejects_unsupported_extensions(filename: str):
    with pytest.raises(UnsupportedFileTypeError):
        detect_doc_type(filename)


# --- ingest_file ------------------------------------------------------------


def test_ingest_file_end_to_end_for_pdf(sample_pdf: Path):
    chunks = ingest_file(sample_pdf, subject="Contracts")
    assert chunks
    assert {c.metadata["source"] for c in chunks} == {"contracts.pdf"}
    assert {c.metadata["page"] for c in chunks} == {1, 2, 3}


def test_ingest_file_end_to_end_for_pptx(sample_pptx: Path):
    chunks = ingest_file(sample_pptx, subject="Torts")
    assert chunks
    assert {c.metadata["source"] for c in chunks} == {"torts-lecture.pptx"}
    assert {c.metadata["slide"] for c in chunks} == {1, 2, 3}


def test_ingest_file_honours_source_name_override(tmp_path: Path, sample_pdf: Path):
    # Browser uploads land in a temp file; the displayed name must survive.
    temp_copy = tmp_path / "tmp123.bin"
    temp_copy.write_bytes(sample_pdf.read_bytes())
    chunks = ingest_file(temp_copy, source_name="Contracts Week 1.pdf")
    assert {c.metadata["source"] for c in chunks} == {"Contracts Week 1.pdf"}


def test_ingest_file_rejects_unsupported_types(tmp_path: Path):
    path = tmp_path / "essay.docx"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError):
        ingest_file(path)


def test_ingest_file_reports_a_document_with_no_extractable_text(empty_pdf: Path):
    with pytest.raises(DocumentParseError, match="OCR"):
        ingest_file(empty_pdf)


# --- config sanity ----------------------------------------------------------


def test_chunk_overlap_is_smaller_than_chunk_size():
    assert 0 <= config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_embedding_dimensions_match_the_known_models():
    assert config.embedding_dimensions("BAAI/bge-base-en-v1.5") == 768
    assert config.embedding_dimensions("sentence-transformers/all-MiniLM-L6-v2") == 384
    # The configured model must have a declared size, whatever it is set to.
    assert config.embedding_dimensions() > 0


def test_embedding_dimensions_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown embedding model"):
        config.embedding_dimensions("not/a-real-model")
