"""Vector store tests.

These run against qdrant-client's embedded on-disk mode in a temp directory -
no cloud credentials, no network. Vectors are deterministic fakes so the suite
stays fast and does not depend on the embedding model being downloaded; the
real model is exercised in ``test_embeddings.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src.ingestion import Chunk
from src.vector_store import (
    DimensionMismatchError,
    SearchHit,
    VectorStore,
)

DIM = 8


def _unit(*values: float) -> list[float]:
    """L2-normalise a vector, padded to DIM, as the real embedder would."""
    vec = list(values) + [0.0] * (DIM - len(values))
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _chunk(chunk_id: str, text: str, **meta) -> Chunk:
    base = {
        "source": "contracts.pdf",
        "doc_type": "pdf",
        "subject": "Contracts",
        "page": 1,
        "chunk_index": 0,
        "chunk_id": chunk_id,
        "citation": "contracts.pdf - p. 1",
    }
    base.update(meta)
    return Chunk(text=text, metadata=base)


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    client = QdrantClient(path=str(tmp_path / "store"))
    store = VectorStore(client=client, collection="test_collection", dimensions=DIM)
    store.ensure_collection()
    return store


# --- collection lifecycle ---------------------------------------------------


def test_ensure_collection_is_idempotent(store: VectorStore):
    store.ensure_collection()
    store.ensure_collection()
    assert store.collection_exists()


def test_dimension_mismatch_is_reported_clearly(store: VectorStore):
    mismatched = VectorStore(
        client=store.client, collection=store.collection, dimensions=DIM + 1
    )
    with pytest.raises(DimensionMismatchError, match="recreate the collection"):
        mismatched.ensure_collection()


def test_recreate_collection_clears_everything(store: VectorStore):
    store.upsert_chunks([_chunk("11111111-1111-4111-8111-111111111111", "x" * 60)], [_unit(1)])
    assert store.count() == 1
    store.recreate_collection()
    assert store.count() == 0


# --- writes -----------------------------------------------------------------


def test_upsert_returns_the_number_written(store: VectorStore):
    chunks = [
        _chunk("11111111-1111-4111-8111-111111111111", "offer and acceptance"),
        _chunk("22222222-2222-4222-8222-222222222222", "consideration", page=2),
    ]
    assert store.upsert_chunks(chunks, [_unit(1), _unit(0, 1)]) == 2
    assert store.count() == 2


def test_upsert_rejects_mismatched_vector_count(store: VectorStore):
    with pytest.raises(ValueError, match="these must match"):
        store.upsert_chunks([_chunk("11111111-1111-4111-8111-111111111111", "a")], [])


def test_upsert_of_nothing_is_a_no_op(store: VectorStore):
    assert store.upsert_chunks([], []) == 0


def test_reindexing_the_same_document_does_not_duplicate(store: VectorStore):
    chunk = _chunk("11111111-1111-4111-8111-111111111111", "offer and acceptance")
    store.upsert_chunks([chunk], [_unit(1)])
    store.upsert_chunks([chunk], [_unit(1)])
    assert store.count() == 1


# --- search -----------------------------------------------------------------


def test_search_returns_the_nearest_chunk_first(store: VectorStore):
    store.upsert_chunks(
        [
            _chunk("11111111-1111-4111-8111-111111111111", "offer and acceptance"),
            _chunk("22222222-2222-4222-8222-222222222222", "consideration", page=2),
        ],
        [_unit(1), _unit(0, 1)],
    )
    hits = store.search(_unit(0, 1), top_k=2, score_threshold=None)
    assert hits[0].text == "consideration"
    assert hits[0].score > hits[1].score


def test_search_preserves_metadata_and_strips_text_from_it(store: VectorStore):
    store.upsert_chunks(
        [_chunk("11111111-1111-4111-8111-111111111111", "offer and acceptance")],
        [_unit(1)],
    )
    hit = store.search(_unit(1), top_k=1, score_threshold=None)[0]
    assert hit.text == "offer and acceptance"
    assert "text" not in hit.metadata
    assert hit.metadata["source"] == "contracts.pdf"
    assert hit.metadata["page"] == 1
    assert hit.citation == "contracts.pdf - p. 1"


def test_search_filters_by_subject(store: VectorStore):
    store.upsert_chunks(
        [
            _chunk("11111111-1111-4111-8111-111111111111", "offer", subject="Contracts"),
            _chunk(
                "22222222-2222-4222-8222-222222222222",
                "duty of care",
                subject="Torts",
                source="torts.pptx",
            ),
        ],
        [_unit(1), _unit(1)],
    )
    hits = store.search(_unit(1), top_k=5, subject="Torts", score_threshold=None)
    assert [h.text for h in hits] == ["duty of care"]


def test_search_filters_by_source(store: VectorStore):
    store.upsert_chunks(
        [
            _chunk("11111111-1111-4111-8111-111111111111", "offer"),
            _chunk(
                "22222222-2222-4222-8222-222222222222",
                "duty of care",
                source="torts.pptx",
            ),
        ],
        [_unit(1), _unit(1)],
    )
    hits = store.search(
        _unit(1), top_k=5, sources=["torts.pptx"], score_threshold=None
    )
    assert [h.text for h in hits] == ["duty of care"]


def test_search_honours_the_score_threshold(store: VectorStore):
    store.upsert_chunks(
        [_chunk("11111111-1111-4111-8111-111111111111", "offer")], [_unit(1)]
    )
    # Orthogonal query vector -> cosine 0, below any positive threshold.
    assert store.search(_unit(0, 1), top_k=5, score_threshold=0.5) == []


def test_search_on_a_missing_collection_returns_empty(tmp_path: Path):
    client = QdrantClient(path=str(tmp_path / "empty"))
    empty = VectorStore(client=client, collection="never_created", dimensions=DIM)
    assert empty.search(_unit(1)) == []
    assert empty.count() == 0
    assert empty.list_sources() == []


# --- knowledge-base listing -------------------------------------------------


def test_list_sources_aggregates_chunk_counts(store: VectorStore):
    store.upsert_chunks(
        [
            _chunk("11111111-1111-4111-8111-111111111111", "a"),
            _chunk("22222222-2222-4222-8222-222222222222", "b", page=2),
            _chunk(
                "33333333-3333-4333-8333-333333333333",
                "c",
                source="torts.pptx",
                subject="Torts",
                doc_type="pptx",
            ),
        ],
        [_unit(1), _unit(1), _unit(1)],
    )
    summaries = {s.source: s for s in store.list_sources()}
    assert summaries["contracts.pdf"].chunk_count == 2
    assert summaries["contracts.pdf"].subject == "Contracts"
    assert summaries["torts.pptx"].chunk_count == 1
    assert summaries["torts.pptx"].doc_type == "pptx"


def test_list_subjects_is_sorted_and_deduplicated(store: VectorStore):
    store.upsert_chunks(
        [
            _chunk("11111111-1111-4111-8111-111111111111", "a"),
            _chunk(
                "33333333-3333-4333-8333-333333333333",
                "c",
                source="torts.pptx",
                subject="Torts",
            ),
        ],
        [_unit(1), _unit(1)],
    )
    assert store.list_subjects() == ["Contracts", "Torts"]


def test_delete_source_removes_only_that_document(store: VectorStore):
    store.upsert_chunks(
        [
            _chunk("11111111-1111-4111-8111-111111111111", "a"),
            _chunk(
                "33333333-3333-4333-8333-333333333333",
                "c",
                source="torts.pptx",
            ),
        ],
        [_unit(1), _unit(1)],
    )
    store.delete_source("contracts.pdf")
    assert [s.source for s in store.list_sources()] == ["torts.pptx"]


def test_delete_source_on_a_missing_collection_is_a_no_op(tmp_path: Path):
    client = QdrantClient(path=str(tmp_path / "empty2"))
    empty = VectorStore(client=client, collection="never_created", dimensions=DIM)
    empty.delete_source("anything.pdf")  # must not raise


# --- SearchHit --------------------------------------------------------------


def test_search_hit_citation_falls_back_when_not_stored():
    assert SearchHit("t", 1.0, {"source": "a.pdf", "page": 3}).citation == "a.pdf - p. 3"
    assert (
        SearchHit("t", 1.0, {"source": "b.pptx", "slide": 2}).citation
        == "b.pptx - slide 2"
    )
    assert SearchHit("t", 1.0, {"source": "c.pdf"}).citation == "c.pdf"
