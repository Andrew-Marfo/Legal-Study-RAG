"""Retrieval + grounded generation tests.

The embedding model and the Groq API are both stubbed: these tests are about
the control flow that keeps answers grounded, not about third-party services.
The single most important assertion here is that the LLM is never called when
retrieval comes back empty.
"""

from __future__ import annotations

from typing import Any

import pytest

from src import prompts, rag
from src.vector_store import SearchHit


def _hit(text: str, citation: str, score: float = 0.8, **meta: Any) -> SearchHit:
    metadata = {"source": citation.split(" - ")[0], "citation": citation}
    metadata.update(meta)
    return SearchHit(text=text, score=score, metadata=metadata)


class FakeStore:
    """Stands in for a VectorStore; records how it was searched."""

    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self.hits = hits or []
        self.calls: list[dict[str, Any]] = []

    def search(self, query_vector, **kwargs):  # noqa: ANN001 - test double
        self.calls.append(kwargs)
        return self.hits


@pytest.fixture(autouse=True)
def stub_embedding(monkeypatch: pytest.MonkeyPatch):
    """Avoid loading the real sentence-transformer in unit tests."""
    monkeypatch.setattr(rag.embeddings, "embed_query", lambda text, **kw: [0.1] * 8)


@pytest.fixture
def captured_llm(monkeypatch: pytest.MonkeyPatch) -> list[list[dict[str, str]]]:
    """Capture messages sent to the LLM and return a canned answer."""
    calls: list[list[dict[str, str]]] = []

    def fake_complete(messages):
        calls.append(list(messages))
        return "An offer is an expression of willingness to contract [1]."

    monkeypatch.setattr(rag, "_complete", fake_complete)
    return calls


@pytest.fixture
def forbidden_llm(monkeypatch: pytest.MonkeyPatch):
    """Make any LLM call an immediate test failure."""

    def explode(messages):
        raise AssertionError("The LLM must not be called without retrieved context.")

    monkeypatch.setattr(rag, "_complete", explode)
    monkeypatch.setattr(rag, "_complete_stream", explode)


# --- The fail-closed guarantee ----------------------------------------------


def test_empty_retrieval_returns_the_honest_message_without_calling_the_llm(
    forbidden_llm,
):
    result = rag.ask("What is promissory estoppel?", store=FakeStore([]))
    assert result.answer == prompts.NO_CONTEXT_MESSAGE
    assert result.answer.startswith(prompts.INSUFFICIENT_CONTEXT_PHRASE)
    assert result.hits == []
    assert result.grounded is False


def test_blank_question_retrieves_nothing_and_calls_nothing(forbidden_llm):
    store = FakeStore([_hit("text", "a.pdf - p. 1")])
    assert rag.retrieve("   ", store=store) == []
    assert store.calls == []


def test_streaming_also_fails_closed(forbidden_llm):
    hits, stream = rag.ask_stream("Unknown topic", store=FakeStore([]))
    assert hits == []
    assert "".join(stream) == prompts.NO_CONTEXT_MESSAGE


# --- Prompt assembly --------------------------------------------------------


def test_the_prompt_carries_the_excerpts_and_their_citations(captured_llm):
    store = FakeStore(
        [
            _hit("An offer is an expression of willingness.", "contracts.pdf - p. 3"),
            _hit("Consideration must move from the promisee.", "contracts.pdf - p. 7"),
        ]
    )
    rag.ask("What is an offer?", store=store)

    system, user = captured_llm[0]
    assert system["role"] == "system"
    assert "ONLY the supplied excerpts" in system["content"]
    assert "NEVER invent" in system["content"]

    assert "[1] Source: contracts.pdf - p. 3" in user["content"]
    assert "[2] Source: contracts.pdf - p. 7" in user["content"]
    assert "An offer is an expression of willingness." in user["content"]
    assert "What is an offer?" in user["content"]
    assert prompts.INSUFFICIENT_CONTEXT_PHRASE in user["content"]


def test_format_context_numbers_excerpts_from_one():
    block = prompts.format_context(
        [_hit("alpha", "a.pdf - p. 1"), _hit("beta", "b.pptx - slide 2")]
    )
    assert block.index("[1]") < block.index("[2]")
    assert "a.pdf - p. 1" in block
    assert "b.pptx - slide 2" in block


# --- Filters are passed through ---------------------------------------------


def test_subject_and_source_filters_reach_the_store(captured_llm):
    store = FakeStore([_hit("text", "torts.pptx - slide 1")])
    rag.ask(
        "Duty of care?", store=store, subject="Torts", sources=["torts.pptx"], top_k=3
    )
    call = store.calls[0]
    assert call["subject"] == "Torts"
    assert call["sources"] == ["torts.pptx"]
    assert call["top_k"] == 3


def test_retrieval_applies_the_configured_score_threshold(captured_llm):
    store = FakeStore([_hit("text", "a.pdf - p. 1")])
    rag.retrieve("anything", store=store)
    assert store.calls[0]["score_threshold"] == rag.config.SCORE_THRESHOLD


# --- RagAnswer --------------------------------------------------------------


def test_sources_are_deduplicated_in_retrieval_order(captured_llm):
    store = FakeStore(
        [
            _hit("a", "contracts.pdf - p. 3"),
            _hit("b", "contracts.pdf - p. 3"),
            _hit("c", "torts.pptx - slide 1"),
        ]
    )
    result = rag.ask("q", store=store)
    assert result.sources == ["contracts.pdf - p. 3", "torts.pptx - slide 1"]


def test_an_answer_with_evidence_is_grounded(captured_llm):
    result = rag.ask("q", store=FakeStore([_hit("a", "contracts.pdf - p. 3")]))
    assert result.grounded is True


def test_an_answer_that_declines_is_not_grounded(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        rag, "_complete", lambda m: prompts.INSUFFICIENT_CONTEXT_PHRASE + " ..."
    )
    result = rag.ask("q", store=FakeStore([_hit("a", "contracts.pdf - p. 3")]))
    assert result.grounded is False


# --- Study tasks ------------------------------------------------------------


def test_study_task_builds_a_task_prompt_over_retrieved_excerpts(captured_llm):
    store = FakeStore([_hit("Negligence has four elements.", "torts.pptx - slide 2")])
    result = rag.run_study_task(
        prompts.SUMMARISE_INSTRUCTION, "Negligence", store=store
    )
    user = captured_llm[0][1]["content"]
    assert "Task target: Negligence" in user
    assert "Negligence has four elements." in user
    assert "Key principles" in user
    assert result.hits


def test_study_task_fails_closed_too(forbidden_llm):
    result = rag.run_study_task(
        prompts.DEFINE_INSTRUCTION, "Estoppel", store=FakeStore([])
    )
    assert result.answer == prompts.NO_CONTEXT_MESSAGE


# --- Error messages ---------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected_fragment",
    [
        ("Error code 429: rate limit exceeded", "rate-limited"),
        ("401 authentication_error: invalid api key", "rejected the API key"),
        ("model `x` not found", "does not recognise the model"),
        ("some unexpected socket failure", "language model call failed"),
    ],
)
def test_provider_errors_are_translated_for_the_student(raw: str, expected_fragment: str):
    assert expected_fragment in rag._friendly_llm_error(Exception(raw))


def test_missing_api_key_is_reported_not_crashed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rag.config, "GROQ_API_KEY", "")
    rag._groq_client.cache_clear()
    with pytest.raises(rag.LLMError, match="GROQ_API_KEY is not set"):
        rag._groq_client()
    rag._groq_client.cache_clear()
