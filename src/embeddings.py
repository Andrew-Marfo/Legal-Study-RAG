"""Local sentence-transformer embeddings.

The model is loaded once per process and reused. It runs on CPU, needs no API
key and has no rate limit, which is what keeps the whole app free to operate.

``bge`` models are asymmetric: queries are prefixed with a short instruction,
passages are not. Getting this wrong quietly degrades retrieval, so the two
directions are separate functions rather than one with a flag the caller may
forget to set.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Iterable, Sequence

from src import config

ProgressCallback = Callable[[int, int], None]


@lru_cache(maxsize=1)
def get_model(model_name: str | None = None):
    """Load (and memoise) the sentence-transformer model.

    The import is deferred so that modules which only need chunking - and the
    test suite - do not pay the multi-second torch import cost.
    """
    from sentence_transformers import SentenceTransformer

    name = model_name or config.EMBED_MODEL
    return SentenceTransformer(name, device="cpu")


def _needs_query_prefix(model_name: str) -> bool:
    """Only the bge family uses the retrieval instruction prefix."""
    return "bge" in model_name.lower()


def embed_documents(
    texts: Sequence[str],
    *,
    model_name: str | None = None,
    batch_size: int = config.EMBED_BATCH_SIZE,
    on_progress: ProgressCallback | None = None,
) -> list[list[float]]:
    """Embed passages for storage.

    Vectors are L2-normalised, so a dot product equals cosine similarity and
    scores land in a predictable 0-1 range for thresholding.
    """
    if not texts:
        return []

    model = get_model(model_name)
    total = len(texts)
    vectors: list[list[float]] = []

    for start in range(0, total, batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = model.encode(
            batch,
            batch_size=len(batch),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vectors.extend(vec.tolist() for vec in encoded)
        if on_progress is not None:
            on_progress(min(start + batch_size, total), total)

    return vectors


def embed_query(text: str, *, model_name: str | None = None) -> list[float]:
    """Embed a question, applying the bge retrieval instruction prefix."""
    name = model_name or config.EMBED_MODEL
    model = get_model(name)
    prepared = (
        f"{config.BGE_QUERY_PREFIX}{text}" if _needs_query_prefix(name) else text
    )
    vector = model.encode(
        prepared,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vector.tolist()


def embed_chunks(
    chunks: Iterable,
    *,
    model_name: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> list[list[float]]:
    """Convenience wrapper: embed the ``text`` of each :class:`~src.ingestion.Chunk`."""
    texts = [chunk.text for chunk in chunks]
    return embed_documents(texts, model_name=model_name, on_progress=on_progress)


def warm_up(model_name: str | None = None) -> None:
    """Force the model download/load now rather than on the user's first query."""
    get_model(model_name).encode("warm up", show_progress_bar=False)
