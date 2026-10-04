"""End-to-end pipeline: file -> chunks -> real embeddings -> store -> search.

This is the Phase 2 acceptance criterion expressed as a test. It uses the real
embedding model against an embedded store, so it exercises everything except
the LLM call.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src import config, embeddings, prompts, rag
from src.ingestion import ingest_file
from src.vector_store import VectorStore

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def indexed_store(tmp_path_factory, sample_pdf: Path, sample_pptx: Path) -> VectorStore:
    """A store holding one indexed PDF (Contracts) and one deck (Torts)."""
    client = QdrantClient(path=str(tmp_path_factory.mktemp("kb") / "store"))
    store = VectorStore(
        client=client,
        collection="integration",
        dimensions=config.embedding_dimensions(),
    )
    store.ensure_collection()

    for path, subject in ((sample_pdf, "Contracts"), (sample_pptx, "Torts")):
        chunks = ingest_file(path, subject=subject)
        store.upsert_chunks(chunks, embeddings.embed_chunks(chunks))

    return store


def test_both_documents_are_in_the_knowledge_base(indexed_store: VectorStore):
    summaries = {s.source: s for s in indexed_store.list_sources()}
    assert set(summaries) == {"contracts.pdf", "torts-lecture.pptx"}
    assert summaries["contracts.pdf"].subject == "Contracts"
    assert summaries["torts-lecture.pptx"].doc_type == "pptx"
    assert indexed_store.count() == sum(s.chunk_count for s in summaries.values())


def test_a_question_retrieves_the_right_passage_with_intact_metadata(
    indexed_store: VectorStore,
):
    hits = rag.retrieve("What is an offer?", store=indexed_store, top_k=3)
    assert hits, "expected at least one hit above the similarity threshold"

    top = hits[0]
    assert "offer" in top.text.lower()
    assert top.metadata["source"] == "contracts.pdf"
    assert top.metadata["page"] == 1
    assert top.metadata["subject"] == "Contracts"
    assert top.citation == "contracts.pdf - p. 1"


def test_retrieval_reaches_slide_content_with_slide_numbers(
    indexed_store: VectorStore,
):
    hits = rag.retrieve("What are the elements of negligence?", store=indexed_store)
    assert hits
    top = hits[0]
    assert top.metadata["source"] == "torts-lecture.pptx"
    assert top.metadata["slide"] == 1
    assert "slide" in top.citation


def test_speaker_notes_are_retrievable(indexed_store: VectorStore):
    hits = rag.retrieve("the neighbour principle", store=indexed_store, top_k=5)
    assert any("Donoghue v Stevenson" in hit.text for hit in hits)


def test_the_subject_filter_confines_retrieval(indexed_store: VectorStore):
    hits = rag.retrieve(
        "What is an offer?", store=indexed_store, subject="Torts", top_k=5
    )
    assert all(hit.metadata["subject"] == "Torts" for hit in hits)


@pytest.mark.parametrize(
    "question",
    [
        "What is the best recipe for sourdough bread?",
        "How do I change a car tyre?",
        "What is the capital of Mongolia?",
    ],
)
def test_clearly_unrelated_questions_retrieve_nothing(
    indexed_store: VectorStore, question: str
):
    assert rag.retrieve(question, store=indexed_store) == []


def test_a_legal_question_outside_the_materials_still_retrieves_chunks(
    indexed_store: VectorStore,
):
    """Documents a known limit of similarity thresholding.

    bge rates any legal prose as broadly similar, so a question about law the
    student has not uploaded clears the floor and reaches the LLM. The system
    prompt is what must decline in that case - hence the assertion that it
    carries the instruction to do so.
    """
    hits = rag.retrieve(
        "Explain the rule against perpetuities", store=indexed_store, top_k=3
    )
    assert hits, "expected the threshold alone not to catch this"
    assert prompts.INSUFFICIENT_CONTEXT_PHRASE in prompts.SYSTEM_PROMPT


def test_an_unrelated_question_short_circuits_before_the_llm(
    indexed_store: VectorStore, monkeypatch: pytest.MonkeyPatch
):
    def explode(messages):
        raise AssertionError("LLM must not be called with no retrieved context")

    monkeypatch.setattr(rag, "_complete", explode)
    result = rag.ask("What is the best recipe for sourdough bread?", store=indexed_store)
    assert result.grounded is False


def test_reindexing_a_document_replaces_rather_than_duplicates(
    indexed_store: VectorStore, sample_pdf: Path
):
    before = indexed_store.count()
    chunks = ingest_file(sample_pdf, subject="Contracts")
    indexed_store.upsert_chunks(chunks, embeddings.embed_chunks(chunks))
    assert indexed_store.count() == before


def test_deleting_a_document_removes_it_from_retrieval(
    tmp_path_factory, sample_pdf: Path
):
    client = QdrantClient(path=str(tmp_path_factory.mktemp("kb2") / "store"))
    store = VectorStore(
        client=client, collection="deltest", dimensions=config.embedding_dimensions()
    )
    store.ensure_collection()
    chunks = ingest_file(sample_pdf, subject="Contracts")
    store.upsert_chunks(chunks, embeddings.embed_chunks(chunks))

    assert rag.retrieve("What is an offer?", store=store)
    store.delete_source("contracts.pdf")
    assert rag.retrieve("What is an offer?", store=store) == []


# --- Word documents ---------------------------------------------------------


@pytest.fixture(scope="module")
def word_store(tmp_path_factory, sample_docx: Path) -> VectorStore:
    client = QdrantClient(path=str(tmp_path_factory.mktemp("kb3") / "store"))
    store = VectorStore(
        client=client,
        collection="word_integration",
        dimensions=config.embedding_dimensions(),
    )
    store.ensure_collection()
    chunks = ingest_file(sample_docx, subject="Contracts")
    store.upsert_chunks(chunks, embeddings.embed_chunks(chunks))
    return store


def test_a_word_document_is_retrievable_and_cites_its_heading(
    word_store: VectorStore,
):
    hits = rag.retrieve("What is promissory estoppel?", store=word_store, top_k=3)
    assert hits
    top = hits[0]
    assert top.metadata["source"] == "contract-notes.docx"
    assert top.metadata["doc_type"] == "docx"
    assert top.metadata["heading"] == "Promissory Estoppel"
    assert top.citation == "contract-notes.docx - Promissory Estoppel"


def test_word_retrieval_resolves_to_the_right_section(word_store: VectorStore):
    hits = rag.retrieve(
        "What are the elements needed to form a contract?",
        store=word_store,
        top_k=3,
    )
    assert hits
    assert hits[0].metadata["heading"] == "Formation of Contract"


def test_a_table_inside_a_word_document_is_retrievable(word_store: VectorStore):
    hits = rag.retrieve("High Trees House", store=word_store, top_k=5)
    assert any("Central London Property" in hit.text for hit in hits)


def test_the_word_knowledge_base_lists_the_document(word_store: VectorStore):
    summaries = word_store.list_sources()
    assert [s.source for s in summaries] == ["contract-notes.docx"]
    assert summaries[0].doc_type == "docx"
