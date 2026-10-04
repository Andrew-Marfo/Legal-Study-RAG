"""Vector storage and similarity search.

Qdrant is used in both deployment modes, which keeps a single code path:

* **Hosted** (default) - a Qdrant Cloud cluster. It lives outside the Hugging
  Face Space, so the knowledge base survives the Space sleeping or rebuilding.
* **Local** (``USE_LOCAL_STORE=true``) - qdrant-client's embedded on-disk mode,
  for offline development. No server, no extra dependency.

The collection is the source of truth for the app; raw uploads are transient.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from qdrant_client import QdrantClient, models

from src import config
from src.ingestion import Chunk, format_citation


class VectorStoreError(RuntimeError):
    """Raised for configuration or connectivity problems with the store."""


class DimensionMismatchError(VectorStoreError):
    """The existing collection was built with a different embedding model."""


@dataclass(frozen=True)
class SearchHit:
    """One retrieved chunk plus its similarity score."""

    text: str
    score: float
    metadata: dict[str, Any]

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown source"))

    @property
    def citation(self) -> str:
        """``file.pdf - p. 12`` / ``deck.pptx - slide 4``."""
        stored = self.metadata.get("citation")
        if stored:
            return str(stored)
        page = self.metadata.get("page")
        slide = self.metadata.get("slide")
        section = self.metadata.get("section")
        if page is not None:
            return format_citation(self.source, "pdf", int(page))
        if slide is not None:
            return format_citation(self.source, "pptx", int(slide))
        if section is not None:
            return format_citation(
                self.source, "docx", int(section), str(self.metadata.get("heading", ""))
            )
        return self.source


@dataclass(frozen=True)
class SourceSummary:
    """A document in the knowledge base, as shown in the UI."""

    source: str
    subject: str
    doc_type: str
    chunk_count: int


def _build_client() -> QdrantClient:
    if config.USE_LOCAL_STORE:
        path = Path(config.LOCAL_STORE_PATH)
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))

    if not config.QDRANT_URL or not config.QDRANT_API_KEY:
        raise VectorStoreError(
            "QDRANT_URL and QDRANT_API_KEY must be set to use Qdrant Cloud. "
            "Set USE_LOCAL_STORE=true to work offline against a local store "
            "instead."
        )
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=60,
    )


class VectorStore:
    """Thin, typed wrapper over a Qdrant collection."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self.client = client or _build_client()
        self.collection = collection or config.QDRANT_COLLECTION
        self.dimensions = dimensions or config.embedding_dimensions()

    # --- collection lifecycle ---------------------------------------------

    def collection_exists(self) -> bool:
        try:
            return self.client.collection_exists(self.collection)
        except Exception as exc:
            raise VectorStoreError(f"Could not reach the vector store: {exc}") from exc

    def ensure_collection(self) -> None:
        """Create the collection if absent; verify its vector size if present."""
        if self.collection_exists():
            self._assert_dimensions()
            return

        try:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dimensions,
                    distance=models.Distance.COSINE,
                ),
            )
        except Exception as exc:
            # A concurrent creator is fine; anything else is not.
            if not self.collection_exists():
                raise VectorStoreError(
                    f"Could not create collection {self.collection!r}: {exc}"
                ) from exc

        self._ensure_payload_indexes()

    def _assert_dimensions(self) -> None:
        """Fail loudly when the stored vector size disagrees with the model."""
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        size = (
            vectors.size
            if isinstance(vectors, models.VectorParams)
            else next(iter(vectors.values())).size  # named-vector config
        )
        if size != self.dimensions:
            raise DimensionMismatchError(
                f"Collection {self.collection!r} stores {size}-dimensional vectors "
                f"but {config.EMBED_MODEL} produces {self.dimensions}. The "
                "embedding model changed - recreate the collection (this deletes "
                "the indexed documents, which must then be re-uploaded)."
            )

    def _ensure_payload_indexes(self) -> None:
        """Index the fields used as retrieval filters.

        Embedded mode ignores payload indexes (and warns about it); filtering
        still works there, just without the index. Neither case is fatal.
        """
        for field in ("subject", "source"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    self.client.create_payload_index(
                        collection_name=self.collection,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
            except Exception:
                pass

    def recreate_collection(self) -> None:
        """Drop and rebuild the collection. Destroys every indexed document."""
        if self.collection_exists():
            try:
                self.client.delete_collection(self.collection)
            except Exception as exc:
                raise VectorStoreError(
                    f"Could not drop collection {self.collection!r}: {exc}"
                ) from exc

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.dimensions,
                distance=models.Distance.COSINE,
            ),
        )
        self._ensure_payload_indexes()

        # qdrant-client's embedded mode can retain points across a
        # delete/create cycle, which would silently mix vectors from two
        # different embedding models. Make the reset unconditional.
        if self.count():
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(filter=models.Filter()),
                wait=True,
            )

    # --- writes ------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        batch_size: int = 64,
    ) -> int:
        """Upsert chunks with their vectors. Returns the number written.

        Point ids are the deterministic ``chunk_id`` values, so re-indexing an
        unchanged document overwrites in place instead of duplicating it.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"{len(chunks)} chunks but {len(vectors)} vectors - these must match."
            )
        if not chunks:
            return 0

        self.ensure_collection()

        written = 0
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            points = [
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector=list(vector),
                    payload={"text": chunk.text, **chunk.metadata},
                )
                for chunk, vector in zip(batch_chunks, batch_vectors)
            ]
            try:
                self.client.upsert(
                    collection_name=self.collection, points=points, wait=True
                )
            except Exception as exc:
                raise VectorStoreError(f"Failed to write to the vector store: {exc}") from exc
            written += len(points)

        return written

    def delete_source(self, source: str) -> None:
        """Remove every chunk belonging to one document."""
        if not self.collection_exists():
            return
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source", match=models.MatchValue(value=source)
                            )
                        ]
                    )
                ),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not delete {source!r}: {exc}") from exc

    # --- reads -------------------------------------------------------------

    @staticmethod
    def _build_filter(
        subject: str | None = None, sources: Iterable[str] | None = None
    ) -> models.Filter | None:
        conditions: list[models.Condition] = []
        if subject:
            conditions.append(
                models.FieldCondition(
                    key="subject", match=models.MatchValue(value=subject)
                )
            )
        source_list = [s for s in (sources or []) if s]
        if source_list:
            conditions.append(
                models.FieldCondition(
                    key="source", match=models.MatchAny(any=source_list)
                )
            )
        return models.Filter(must=conditions) if conditions else None

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = config.TOP_K,
        subject: str | None = None,
        sources: Iterable[str] | None = None,
        score_threshold: float | None = config.SCORE_THRESHOLD,
    ) -> list[SearchHit]:
        """Cosine similarity search, optionally scoped by subject and source."""
        if not self.collection_exists():
            return []

        query_filter = self._build_filter(subject, sources)
        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=list(query_vector),
                limit=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )
            points = response.points
        except Exception as exc:
            raise VectorStoreError(f"Search failed: {exc}") from exc

        hits: list[SearchHit] = []
        for point in points:
            payload = dict(point.payload or {})
            text = payload.pop("text", "")
            hits.append(
                SearchHit(text=text, score=float(point.score), metadata=payload)
            )
        return hits

    def count(self) -> int:
        """Total number of indexed chunks."""
        if not self.collection_exists():
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    def list_sources(self) -> list[SourceSummary]:
        """Distinct indexed documents, with their chunk counts."""
        if not self.collection_exists():
            return []

        aggregates: dict[str, dict[str, Any]] = {}
        offset: Any = None
        while True:
            try:
                records, offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=256,
                    offset=offset,
                    with_payload=["source", "subject", "doc_type"],
                    with_vectors=False,
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Could not list indexed documents: {exc}"
                ) from exc

            for record in records:
                payload = record.payload or {}
                source = str(payload.get("source", "unknown"))
                entry = aggregates.setdefault(
                    source,
                    {
                        "subject": str(payload.get("subject", "General")),
                        "doc_type": str(payload.get("doc_type", "")),
                        "count": 0,
                    },
                )
                entry["count"] += 1

            if offset is None:
                break

        return sorted(
            (
                SourceSummary(
                    source=source,
                    subject=entry["subject"],
                    doc_type=entry["doc_type"],
                    chunk_count=entry["count"],
                )
                for source, entry in aggregates.items()
            ),
            key=lambda s: s.source.lower(),
        )

    def list_subjects(self) -> list[str]:
        """Subjects currently present in the knowledge base."""
        return sorted({s.subject for s in self.list_sources()})


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    """Process-wide store singleton (one client, reused across Streamlit reruns)."""
    return VectorStore()
