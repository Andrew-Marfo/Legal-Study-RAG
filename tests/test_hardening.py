"""Phase 7: bad input must produce a message, never a crash."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from src.ingestion import (
    DocumentParseError,
    UnsupportedFileTypeError,
    clean_text,
    ingest_file,
    parse_pptx,
)


class FakeUpload:
    """Mimics the attributes app._validate_upload reads off an UploadedFile."""

    def __init__(self, name: str, size: int) -> None:
        self.name = name
        self.size = size


def _validate(name: str, size: int) -> str | None:
    """Mirror of app._validate_upload, importable without starting Streamlit."""
    from app import _validate_upload  # imported lazily: pulls in streamlit

    return _validate_upload(FakeUpload(name, size))


# --- Upload validation ------------------------------------------------------


def test_a_good_file_passes_validation():
    assert _validate("notes.pdf", 1024) is None


@pytest.mark.parametrize("name", ["notes.txt", "notes.odt", "notes.pages", "scan.jpg"])
def test_an_unsupported_extension_is_rejected(name: str):
    message = _validate(name, 1024)
    assert message and "unsupported file type" in message


@pytest.mark.parametrize("name", ["notes.pdf", "deck.pptx", "essay.docx"])
def test_every_advertised_extension_passes_validation(name: str):
    assert _validate(name, 1024) is None


def test_an_oversized_file_is_rejected_with_its_size():
    size = (config.MAX_FILE_MB + 5) * 1024 * 1024
    message = _validate("huge.pdf", size)
    assert message and f"{config.MAX_FILE_MB} MB limit" in message


def test_an_empty_file_is_rejected():
    message = _validate("empty.pdf", 0)
    assert message and "empty" in message


def test_a_file_at_exactly_the_limit_is_accepted():
    assert _validate("edge.pdf", config.MAX_FILE_MB * 1024 * 1024) is None


# --- Corrupt and odd documents ----------------------------------------------


def test_a_truncated_pdf_reports_rather_than_crashes(tmp_path: Path):
    path = tmp_path / "truncated.pdf"
    path.write_bytes(b"%PDF-1.7\n%garbage, nothing else here")
    with pytest.raises(DocumentParseError):
        ingest_file(path)


def test_a_pptx_that_is_not_a_zip_reports_rather_than_crashes(tmp_path: Path):
    path = tmp_path / "broken.pptx"
    path.write_bytes(b"definitely not a zip archive")
    with pytest.raises(DocumentParseError):
        ingest_file(path)


def test_a_missing_file_reports_rather_than_crashes(tmp_path: Path):
    with pytest.raises(DocumentParseError):
        ingest_file(tmp_path / "does-not-exist.pdf")


def test_a_legacy_word_file_is_rejected_with_a_conversion_hint(tmp_path: Path):
    path = tmp_path / "old-notes.doc"
    path.write_bytes(bytes([0xD0, 0xCF, 0x11, 0xE0]))  # OLE2 magic: real .doc
    with pytest.raises(UnsupportedFileTypeError, match="Save As"):
        ingest_file(path)


def test_an_extensionless_file_is_rejected(tmp_path: Path):
    path = tmp_path / "README"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError):
        ingest_file(path)


def test_a_deck_with_no_text_reports_rather_than_crashes(tmp_path: Path):
    from pptx import Presentation

    path = tmp_path / "blank.pptx"
    presentation = Presentation()
    presentation.slides.add_slide(presentation.slide_layouts[6])
    presentation.save(str(path))

    assert parse_pptx(path) == []
    with pytest.raises(DocumentParseError, match="No extractable text"):
        ingest_file(path)


# --- Text cleaning robustness -----------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "\n\n\n",
        "\x00\x00",
        "日本語のテキスト",
        "emoji 📄 and symbols §§ ¶¶",
    ],
)
def test_clean_text_never_raises(raw: str):
    clean_text(raw)


def test_clean_text_handles_a_very_long_run_of_text():
    # Built inside the test rather than parametrised: a 100k-character test id
    # overflows the Windows environment-variable limit that pytest writes
    # PYTEST_CURRENT_TEST into.
    assert clean_text("a" * 100_000) == "a" * 100_000


def test_a_filename_with_unusual_characters_survives_as_metadata(
    tmp_path: Path, sample_pdf: Path
):
    odd = "Contracts & Torts (2026) — week #1.pdf"
    copy = tmp_path / "tmp.pdf"
    copy.write_bytes(sample_pdf.read_bytes())
    chunks = ingest_file(copy, source_name=odd)
    assert {c.metadata["source"] for c in chunks} == {odd}
    assert chunks[0].citation.startswith(odd)


# --- Configuration guards ---------------------------------------------------


def test_missing_secrets_lists_what_is_needed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "QDRANT_URL", "")
    monkeypatch.setattr(config, "QDRANT_API_KEY", "")
    monkeypatch.setattr(config, "USE_LOCAL_STORE", False)
    assert set(config.missing_secrets()) == {
        "GROQ_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
    }
    assert config.is_configured() is False


def test_local_store_mode_does_not_require_qdrant_credentials(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(config, "GROQ_API_KEY", "set")
    monkeypatch.setattr(config, "QDRANT_URL", "")
    monkeypatch.setattr(config, "QDRANT_API_KEY", "")
    monkeypatch.setattr(config, "USE_LOCAL_STORE", True)
    assert config.missing_secrets() == []


def test_whitespace_only_secrets_count_as_missing():
    assert config._env("A_NAME_THAT_IS_NOT_SET") == ""


def test_supported_extensions_match_the_uploader_filter():
    assert set(config.SUPPORTED_EXTENSIONS) == {".pdf", ".pptx", ".docx"}
