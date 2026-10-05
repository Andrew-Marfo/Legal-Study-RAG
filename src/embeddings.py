"""Local embeddings, run on CPU via ONNX.

No API key and no rate limit, which is what keeps the app free to operate.

The runtime is fastembed (onnxruntime) rather than sentence-transformers
(PyTorch). Both serve the same ``BAAI/bge-base-en-v1.5`` weights and produce
the same 768-dimensional vectors - measured agreement is cosine 1.0000, and
retrieval ranking is unchanged when one encoder's queries are run against the
other's passages - but the ONNX path peaks near 566MB of RSS against 1381MB
for PyTorch. That difference is what lets the app run inside a 1GB free-tier
host, and it is why switching did not require re-indexing anything.

``bge`` models are asymmetric: queries carry a short instruction prefix,
passages do not. fastembed's ``query_embed`` does NOT apply that prefix for
this model - it is identical to ``embed`` - so the prefix is applied here
explicitly. Leaving it off measurably degrades retrieval, and silently.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Iterable, Sequence

from src import config

ProgressCallback = Callable[[int, int], None]


@lru_cache(maxsize=1)
def get_model(model_name: str | None = None):
    """Load (and memoise) the ONNX embedding model.

    The import is deferred so that modules which only need chunking - and the
    fast tests - do not pay the onnxruntime import cost.
    """
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name or config.EMBED_MODEL)


def _needs_query_prefix(model_name: str) -> bool:
    """Only the bge family uses the retrieval instruction prefix."""
    return "bge" in model_name.lower()


def _encode(model, texts: Sequence[str]) -> list[list[float]]:
    """Run the model and return plain lists of floats.

    fastembed yields numpy arrays lazily; materialising them here keeps numpy
    types out of the rest of the codebase, including the Qdrant payloads.
    """
    return [[float(x) for x in vector] for vector in model.embed(list(texts))]


def embed_documents(
    texts: Sequence[str],
    *,
    model_name: str | None = None,
    batch_size: int = config.EMBED_BATCH_SIZE,
    on_progress: ProgressCallback | None = None,
) -> list[list[float]]:
    """Embed passages for storage.

    Vectors are L2-normalised by the model, so a dot product equals cosine
    similarity and scores land in a predictable 0-1 range for thresholding.
    """
    if not texts:
        return []

    model = get_model(model_name)
    total = len(texts)
    vectors: list[list[float]] = []

    for start in range(0, total, batch_size):
        vectors.extend(_encode(model, texts[start : start + batch_size]))
        if on_progress is not None:
            on_progress(min(start + batch_size, total), total)

    return vectors


def embed_query(text: str, *, model_name: str | None = None) -> list[float]:
    """Embed a question, applying the bge retrieval instruction prefix."""
    name = model_name or config.EMBED_MODEL
    prepared = (
        f"{config.BGE_QUERY_PREFIX}{text}" if _needs_query_prefix(name) else text
    )
    return _encode(get_model(name), [prepared])[0]


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
    _encode(get_model(model_name), ["warm up"])
