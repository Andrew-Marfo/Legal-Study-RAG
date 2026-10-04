"""Embedding tests against the real model.

Marked ``slow`` because they load bge-base (~440MB, downloaded on first run).
Skip them with ``-m "not slow"``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src import config, embeddings
from src.ingestion import ingest_file

pytestmark = pytest.mark.slow


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_document_vectors_have_the_configured_dimensions():
    vectors = embeddings.embed_documents(["An offer is an expression of willingness."])
    assert len(vectors) == 1
    assert len(vectors[0]) == config.embedding_dimensions()


def test_vectors_are_l2_normalised():
    [vector] = embeddings.embed_documents(["consideration must move from the promisee"])
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-4)


def test_query_vectors_match_document_dimensions():
    assert len(embeddings.embed_query("what is an offer?")) == (
        config.embedding_dimensions()
    )


def test_embedding_nothing_returns_nothing():
    assert embeddings.embed_documents([]) == []


def test_batching_produces_the_same_vectors_as_one_pass():
    texts = [f"Legal principle number {i} concerning contractual liability." for i in range(5)]
    single = embeddings.embed_documents(texts, batch_size=64)
    batched = embeddings.embed_documents(texts, batch_size=2)
    for a, b in zip(single, batched):
        assert _cosine(a, b) > 0.999


def test_progress_callback_reports_completion():
    seen: list[tuple[int, int]] = []
    texts = ["a" * 20, "b" * 20, "c" * 20]
    embeddings.embed_documents(texts, batch_size=2, on_progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (3, 3)


def test_the_bge_query_prefix_is_applied_to_queries_only(monkeypatch):
    captured: list[str] = []

    class Recorder:
        def encode(self, text, **kwargs):
            captured.append(text if isinstance(text, str) else text[0])
            import numpy as np

            size = config.embedding_dimensions()
            if isinstance(text, str):
                return np.zeros(size)
            return np.zeros((len(text), size))

    monkeypatch.setattr(embeddings, "get_model", lambda name=None: Recorder())
    embeddings.embed_query("what is an offer?")
    embeddings.embed_documents(["an offer is..."])

    assert captured[0].startswith(config.BGE_QUERY_PREFIX)
    assert not captured[1].startswith(config.BGE_QUERY_PREFIX)


def test_retrieval_ranks_the_relevant_passage_first():
    """The real quality check: does a question find its answer?"""
    passages = [
        "An offer is an expression of willingness to contract on specified terms, "
        "made with the intention that it shall become binding on acceptance.",
        "The doctrine of privity provides that a contract cannot confer rights on "
        "a person who is not a party to it.",
        "A testator must have testamentary capacity at the time the will is made.",
    ]
    vectors = embeddings.embed_documents(passages)
    query = embeddings.embed_query("What makes something an offer?")

    scores = [_cosine(query, vector) for vector in vectors]
    assert scores.index(max(scores)) == 0


def test_an_unrelated_question_scores_below_the_threshold():
    [vector] = embeddings.embed_documents(
        ["An offer is an expression of willingness to contract on specified terms."]
    )
    query = embeddings.embed_query("How do I bake sourdough bread at home?")
    assert _cosine(query, vector) < config.SCORE_THRESHOLD


def test_chunks_from_a_real_document_embed_end_to_end(sample_pdf: Path):
    chunks = ingest_file(sample_pdf, subject="Contracts")
    vectors = embeddings.embed_chunks(chunks)
    assert len(vectors) == len(chunks)
    assert all(len(v) == config.embedding_dimensions() for v in vectors)
