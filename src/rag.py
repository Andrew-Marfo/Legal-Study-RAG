"""Retrieval-augmented generation: question in, grounded cited answer out.

The pipeline is deliberately fail-closed. If retrieval returns nothing above
the similarity threshold, the LLM is never called - the honest "not in your
materials" answer is returned directly. A model asked to answer with no context
will answer from training data, which is exactly the hallucinated-law failure
this app exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Iterator, Sequence

from src import config, embeddings, prompts
from src.vector_store import SearchHit, VectorStore, VectorStoreError, get_store


class LLMError(RuntimeError):
    """Raised when the generation call fails or is not configured."""


@dataclass
class RagAnswer:
    """A generated answer together with the evidence behind it."""

    answer: str
    hits: list[SearchHit] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """False when the answer is the honest 'not found' response."""
        return bool(self.hits) and not self.answer.startswith(
            prompts.INSUFFICIENT_CONTEXT_PHRASE
        )

    @property
    def sources(self) -> list[str]:
        """Distinct citations behind the answer, in retrieval order."""
        seen: list[str] = []
        for hit in self.hits:
            if hit.citation not in seen:
                seen.append(hit.citation)
        return seen


@lru_cache(maxsize=1)
def _groq_client():
    """Lazily build the Groq client so an unconfigured app still boots."""
    if not config.GROQ_API_KEY:
        raise LLMError(
            "GROQ_API_KEY is not set. Add it to .env locally, or as a "
            "Repository Secret on Hugging Face Spaces."
        )
    from groq import Groq

    return Groq(api_key=config.GROQ_API_KEY)


# --- Retrieval --------------------------------------------------------------


def retrieve(
    question: str,
    *,
    store: VectorStore | None = None,
    subject: str | None = None,
    sources: Iterable[str] | None = None,
    top_k: int = config.TOP_K,
    score_threshold: float | None = config.SCORE_THRESHOLD,
) -> list[SearchHit]:
    """Embed the question and fetch the most similar chunks."""
    question = (question or "").strip()
    if not question:
        return []

    target = store or get_store()
    query_vector = embeddings.embed_query(question)
    return target.search(
        query_vector,
        top_k=top_k,
        subject=subject,
        sources=sources,
        score_threshold=score_threshold,
    )


# --- Generation -------------------------------------------------------------


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _complete(messages: Sequence[dict[str, str]]) -> str:
    """One non-streaming completion."""
    try:
        response = _groq_client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=list(messages),
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(_friendly_llm_error(exc)) from exc
    content = (response.choices[0].message.content or "").strip()
    return prompts.normalise_citation_markers(content)


def _complete_stream(messages: Sequence[dict[str, str]]) -> Iterator[str]:
    """Token stream for the same completion."""
    try:
        stream = _groq_client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=list(messages),
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            stream=True,
        )
        for event in stream:
            piece = event.choices[0].delta.content
            if piece:
                yield prompts.normalise_citation_markers(piece)
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(_friendly_llm_error(exc)) from exc


def _friendly_llm_error(exc: Exception) -> str:
    """Turn provider errors into something a student can act on."""
    text = str(exc)
    lowered = text.lower()
    if "rate" in lowered and "limit" in lowered:
        return (
            "The free Groq tier is rate-limited and has had too many requests "
            "just now. Wait a minute and ask again."
        )
    if "authentication" in lowered or "api key" in lowered or "401" in lowered:
        return "Groq rejected the API key. Check GROQ_API_KEY is correct and active."
    if "model" in lowered and any(
        marker in lowered
        for marker in ("not found", "not_found", "does not exist", "decommission")
    ):
        return (
            f"Groq does not recognise the model {config.LLM_MODEL!r} - it has "
            "probably been retired. Set LLM_MODEL in .env to a current model "
            "(for example openai/gpt-oss-120b)."
        )
    return f"The language model call failed: {text}"


# --- Public API -------------------------------------------------------------


def ask(
    question: str,
    *,
    store: VectorStore | None = None,
    subject: str | None = None,
    sources: Iterable[str] | None = None,
    top_k: int = config.TOP_K,
) -> RagAnswer:
    """Answer a question from the knowledge base. Never calls the LLM blind."""
    hits = retrieve(
        question, store=store, subject=subject, sources=sources, top_k=top_k
    )
    if not hits:
        return RagAnswer(answer=prompts.NO_CONTEXT_MESSAGE, hits=[])

    answer = _complete(
        _messages(prompts.SYSTEM_PROMPT, prompts.build_question_prompt(question, hits))
    )
    return RagAnswer(answer=answer, hits=hits)


def ask_stream(
    question: str,
    *,
    store: VectorStore | None = None,
    subject: str | None = None,
    sources: Iterable[str] | None = None,
    top_k: int = config.TOP_K,
) -> tuple[list[SearchHit], Iterator[str]]:
    """Streaming variant: retrieve first, then yield answer tokens.

    Returning the hits up front lets the UI render citations as soon as they
    are known, rather than after generation completes.
    """
    hits = retrieve(
        question, store=store, subject=subject, sources=sources, top_k=top_k
    )
    if not hits:
        return [], iter([prompts.NO_CONTEXT_MESSAGE])

    stream = _complete_stream(
        _messages(prompts.SYSTEM_PROMPT, prompts.build_question_prompt(question, hits))
    )
    return hits, stream


def run_study_task(
    instruction: str,
    target: str,
    *,
    store: VectorStore | None = None,
    subject: str | None = None,
    sources: Iterable[str] | None = None,
    top_k: int = config.TOP_K * 2,
    query: str | None = None,
) -> RagAnswer:
    """Run a Phase 5 study task (summarise / practice questions / define).

    ``target`` names what the task is about and is also the retrieval query
    unless ``query`` overrides it.
    """
    hits = retrieve(
        query or target,
        store=store,
        subject=subject,
        sources=sources,
        top_k=top_k,
    )
    if not hits:
        return RagAnswer(answer=prompts.NO_CONTEXT_MESSAGE, hits=[])

    answer = _complete(
        _messages(
            prompts.SYSTEM_PROMPT, prompts.build_task_prompt(instruction, target, hits)
        )
    )
    return RagAnswer(answer=answer, hits=hits)


def health_check() -> tuple[bool, str]:
    """Quick readiness probe for the UI: store reachable, key present."""
    try:
        store = get_store()
        count = store.count()
    except VectorStoreError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"Vector store unavailable: {exc}"

    if not config.GROQ_API_KEY:
        return False, "GROQ_API_KEY is not set, so questions cannot be answered."

    return True, f"Ready - {count} chunks indexed."
