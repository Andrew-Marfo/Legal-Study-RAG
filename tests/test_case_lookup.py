"""Named-case lookup tests.

The casebook here mirrors the real failure: a land-law compilation where the
case is *cited* inside an earlier judgment, then reported under its own
upper-case heading, then followed by the next report. Embedded Qdrant, fake
vectors - the lookup is text-based and never embeds anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src import case_lookup, rag
from src.case_lookup import (
    _strip_overlap,
    extract_case_reference,
    heading_score,
    looks_like_new_case,
    read_case,
)
from src.ingestion import DocumentUnit, chunk_units
from src.vector_store import VectorStore

DIM = 8
SOURCE = "Land Law Cases.pdf"

JUDGMENT = (
    "The plaintiff claimed declaration of title to a portion of stool land "
    "which he had cleared and farmed for many years. "
) * 8 + (
    "The stool holds the absolute title as trustee for its subjects, and the "
    "subjects are entitled to the usufruct of the portions they occupy. "
) * 8

PAGES = [
    # Page 1: an earlier case that merely CITES Ohimen v Adjei.
    "MENSAH V. ASAMOAH (1975) 1 GLR 225\nCOURT OF APPEAL\n"
    + "The court considered the authorities on stool land at length. " * 6
    + "\nCASES REFERRED TO\n(3) Ohimen v. Adjei (1957) 2 W.A.L.R. 275.\n"
    + "Ohimen v Adjei (1957) 2 W.A.L.R. 275, S.C.; Thompson v. Mensah cited.\n",
    # Page 2: the report itself.
    "OHIMEN V. ADJEI AND ANOTHER. (1957) 2 WALR 275\n"
    "SUPREME COURT OF GHANA, LAND COURT, CAPE COAST\nOLLENU J.\n" + JUDGMENT,
    # Page 3: the judgment ends and the next report begins.
    "Judgment for the plaintiff on the usufructuary claim.\n"
    "THOMPSON V. MENSAH (1957) 3 WALR 240\nCOURT OF APPEAL\n"
    + "An unrelated dispute about a lease of family land. " * 10,
]


def _vector() -> list[float]:
    return [1.0] + [0.0] * (DIM - 1)


@pytest.fixture
def casebook(tmp_path: Path) -> VectorStore:
    store = VectorStore(
        client=QdrantClient(path=str(tmp_path / "store")),
        collection="casebook",
        dimensions=DIM,
    )
    units = [
        DocumentUnit(text=text, doc_type="pdf", index=i)
        for i, text in enumerate(PAGES, start=1)
    ]
    chunks = chunk_units(
        units, source=SOURCE, subject="Land Law", chunk_size=400, chunk_overlap=80
    )
    store.upsert_chunks(chunks, [_vector() for _ in chunks])
    return store


# --- recognising a case name in a question ---------------------------------------


@pytest.mark.parametrize(
    "question, first, second",
    [
        ('summarize the case "Ohimen vs Adjei"', "ohimen", "adjei"),
        ("What was held in Ohimen v. Adjei?", "ohimen", "adjei"),
        ("Golightly and Ors. v. Ashrifi", "golightly", "ashrifi"),
        ("Bradford Corporation v Pickles", "bradford", "pickles"),
        ("explain donoghue versus stevenson", "donoghue", "stevenson"),
    ],
)
def test_extracts_party_words(question, first, second):
    reference = extract_case_reference(question)
    assert reference is not None
    assert (reference.first_party, reference.second_party) == (first, second)


def test_single_letter_party_is_not_searched_on():
    reference = extract_case_reference("What did R v Woollin decide?")
    assert reference is not None
    assert reference.tokens == ["woollin"]


@pytest.mark.parametrize(
    "question",
    ["What is privity of contract?", "Explain consideration", "", "v"],
)
def test_no_case_named(question):
    assert extract_case_reference(question) is None


# --- telling a report heading from a citation -----------------------------------


def _ref():
    return extract_case_reference("Ohimen v Adjei")


def test_report_heading_scores_high():
    lines = [
        "OHIMEN V. ADJEI AKD ANOTHER.   (1957) 2 WALR 257",
        "SUPREME COURT OF GHANA, CENTRAL JUDICIAL DIVISION, LAND COURT",
    ]
    assert heading_score(lines, 0, _ref()) >= 3


@pytest.mark.parametrize(
    "line",
    [
        "(3) Ohimen v. Adjei (1957) 2 W.A.L.R. 275.",
        "Ohimen v Adjei (1957) 3 W.A.L.R. 275, S.C.; Thompson v. Mensah cited.",
        "Ohimen v. Adjei (1957) 2 W.A.L.R. 275 at pp. 279-280 the court held",
        "see Ohimen v. Adjei (1957) 2 W.A.L.R. 275.",
    ],
)
def test_citations_are_not_headings(line):
    assert heading_score([line, "and the learned judge went on"], 0, _ref()) < 3


def test_next_report_is_recognised_but_not_the_same_case():
    assert looks_like_new_case("THOMPSON V. MENSAH (1957) 3 WALR 240", _ref())
    assert not looks_like_new_case("OHIMEN V. ADJEI (1957) 2 WALR 275", _ref())
    assert not looks_like_new_case("KWASI T. OPPONG-KYEKYEKU, LAND LAW CASES", _ref())


def test_strip_overlap_removes_repeated_text():
    previous = "The stool holds the absolute title as trustee for its subjects."
    current = "as trustee for its subjects. The subjects hold the usufruct."
    assert _strip_overlap(previous, current) == "The subjects hold the usufruct."
    assert _strip_overlap("unrelated", current) == current


# --- reading the judgment --------------------------------------------------------


def test_reads_the_report_not_the_citation(casebook):
    hits = read_case(casebook, _ref())
    assert hits, "the report should be found"
    assert hits[0].text.startswith("OHIMEN V. ADJEI AND ANOTHER.")
    assert hits[0].metadata["page"] == 2
    assert all(h.metadata["matched_by"] == "case name" for h in hits)


def test_reads_in_order_and_stops_at_the_next_case(casebook):
    hits = read_case(casebook, _ref())
    combined = "\n".join(h.text for h in hits)
    assert "usufruct" in combined
    assert "Judgment for the plaintiff" in combined
    assert "THOMPSON V. MENSAH" not in combined
    assert "unrelated dispute" not in combined
    indexes = [h.metadata["chunk_index"] for h in hits]
    assert indexes == sorted(indexes)


def test_respects_the_character_budget(casebook):
    full = sum(len(h.text) for h in read_case(casebook, _ref()))
    capped = read_case(casebook, _ref(), max_chars=600)
    assert full > 600
    assert capped and sum(len(h.text) for h in capped) <= 600 + len(" [...]")


def test_mixed_case_report_title_is_a_heading():
    lines = ["Ohimen v Adjei and Another (1957) 2 WALR 275", "Ollennu J."]
    assert heading_score(lines, 0, _ref()) >= 3


def test_scope_filters_apply(casebook):
    assert read_case(casebook, _ref(), sources=["Other.pdf"]) == []
    assert read_case(casebook, _ref(), subject="Torts") == []
    assert read_case(casebook, _ref(), sources=[SOURCE], subject="Land Law")


def test_cited_only_case_is_not_found(casebook):
    # Thompson v Mensah is reported here, but Golightly v Ashrifi never is.
    reference = extract_case_reference("Golightly v Ashrifi")
    assert read_case(casebook, reference) == []


def test_lookup_swallows_store_errors(casebook):
    casebook.client.close()
    assert case_lookup.lookup("summarise Ohimen v Adjei", casebook) == []


# --- wired into retrieval --------------------------------------------------------


def test_retrieve_uses_the_lookup_for_named_cases(casebook, monkeypatch):
    def no_embedding(_question):
        raise AssertionError("semantic search should not run for a named case")

    monkeypatch.setattr(rag.embeddings, "embed_query", no_embedding)
    hits = rag.retrieve('summarize the case "Ohimen vs Adjei"', store=casebook)
    assert hits[0].text.startswith("OHIMEN V. ADJEI")


def test_retrieve_falls_back_to_semantic_search(casebook, monkeypatch):
    calls = []

    def fake_embedding(question):
        calls.append(question)
        return _vector()

    monkeypatch.setattr(rag.embeddings, "embed_query", fake_embedding)
    rag.retrieve("Who holds the absolute title to stool land?", store=casebook)
    rag.retrieve("Summarise Golightly v Ashrifi", store=casebook)  # cited only
    assert len(calls) == 2
