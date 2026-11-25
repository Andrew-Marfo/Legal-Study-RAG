"""Landmark case examples matched to a student's question.

The answer itself comes only from the student's own materials. Alongside it,
the app suggests one or two cases from a curated library (``data/cases.json``)
that illustrate the same principle - the way a tutor would say "see Letang v
Cooper" after explaining trespass.

The library is curated rather than generated on purpose. An LLM asked to name
a relevant case will sometimes invent one, with a plausible citation, and a
fabricated authority is the worst failure a legal study tool can have. Every
case shown here is one a person put in the file.

Matching is plain vector similarity between the question and each case's
principle, topics and facts, using the same embedding model as the documents.
The cases live in their own Qdrant collection, rebuilt automatically whenever
the JSON file or the embedding model changes.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from qdrant_client import QdrantClient, models

from src import config

EmbedFn = Callable[[Sequence[str]], list[list[float]]]

_REQUIRED_TEXT_FIELDS = ("name", "court", "jurisdiction", "subject", "principle", "facts", "held")


class CaseLibraryError(RuntimeError):
    """The case file is malformed, or the case collection is unreachable."""


@dataclass(frozen=True)
class Case:
    """One case brief, as shown in the UI."""

    name: str
    court: str
    year: int
    jurisdiction: str
    subject: str
    principle: str
    facts: str
    held: str
    citation: str = ""
    topics: tuple[str, ...] = field(default_factory=tuple)

    @property
    def case_id(self) -> str:
        """Stable point id, derived from the name, so re-syncs overwrite in place."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"legal-study-rag/case/{self.name}"))

    @property
    def search_text(self) -> str:
        """What gets embedded. Principle and topics first: they carry the law."""
        topics = ", ".join(self.topics)
        return (
            f"{self.name}. {self.subject}: {topics}. "
            f"{self.principle} {self.facts}"
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["topics"] = list(self.topics)
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Case":
        return cls(
            name=str(payload["name"]),
            court=str(payload["court"]),
            year=int(payload["year"]),
            jurisdiction=str(payload["jurisdiction"]),
            subject=str(payload["subject"]),
            principle=str(payload["principle"]),
            facts=str(payload["facts"]),
            held=str(payload["held"]),
            citation=str(payload.get("citation") or ""),
            topics=tuple(str(t) for t in payload.get("topics") or ()),
        )


@dataclass(frozen=True)
class CaseMatch:
    """A case and how closely it matched the question."""

    case: Case
    score: float


def load_cases(path: Path | str = config.CASES_PATH) -> list[Case]:
    """Read and validate the case file. Fails loudly on a bad entry."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CaseLibraryError(f"Case library not found at {path}.") from exc
    except json.JSONDecodeError as exc:
        raise CaseLibraryError(f"Case library {path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, list):
        raise CaseLibraryError("The case library must be a JSON list of cases.")

    cases: list[Case] = []
    seen: set[str] = set()
    for position, entry in enumerate(raw, start=1):
        label = entry.get("name") if isinstance(entry, dict) else None
        where = f"case #{position}" + (f" ({label})" if label else "")
        if not isinstance(entry, dict):
            raise CaseLibraryError(f"{where} is not an object.")
        missing = [f for f in _REQUIRED_TEXT_FIELDS if not str(entry.get(f) or "").strip()]
        if missing:
            raise CaseLibraryError(f"{where} is missing: {', '.join(missing)}.")
        try:
            case = Case.from_payload(entry)
        except (KeyError, TypeError, ValueError) as exc:
            raise CaseLibraryError(f"{where} is invalid: {exc}") from exc
        if case.name in seen:
            raise CaseLibraryError(f"{where} is a duplicate.")
        seen.add(case.name)
        cases.append(case)
    return cases


def library_version(cases: Sequence[Case], model_name: str = config.EMBED_MODEL) -> str:
    """Fingerprint of the cases and the model that embeds them."""
    digest = hashlib.sha256(model_name.encode())
    for case in cases:
        digest.update(json.dumps(case.to_payload(), sort_keys=True).encode())
    return digest.hexdigest()[:16]


def _default_embed(texts: Sequence[str]) -> list[list[float]]:
    from src import embeddings

    return embeddings.embed_documents(list(texts))


class CaseLibrary:
    """The case collection: kept in sync with the JSON file, and searchable."""

    def __init__(
        self,
        client: QdrantClient,
        *,
        cases: Sequence[Case] | None = None,
        collection: str = config.CASES_COLLECTION,
        dimensions: int | None = None,
        model_name: str = config.EMBED_MODEL,
    ) -> None:
        self.client = client
        self.cases = list(cases) if cases is not None else load_cases()
        self.collection = collection
        self.dimensions = dimensions or config.embedding_dimensions(model_name)
        self.version = library_version(self.cases, model_name)

    # --- sync ----------------------------------------------------------------

    def is_current(self) -> bool:
        """True when the collection holds exactly this library, embedded by this model."""
        try:
            if not self.client.collection_exists(self.collection):
                return False
            if self.client.count(self.collection, exact=True).count != len(self.cases):
                return False
            records, _ = self.client.scroll(
                collection_name=self.collection,
                limit=1,
                with_payload=["library_version"],
                with_vectors=False,
            )
        except Exception as exc:
            raise CaseLibraryError(f"Could not reach the case library: {exc}") from exc
        return bool(records) and (records[0].payload or {}).get(
            "library_version"
        ) == self.version

    def sync(self, embed: EmbedFn | None = None) -> bool:
        """Rebuild the collection if it is stale. Returns True if it rebuilt."""
        if self.is_current():
            return False

        vectors = (embed or _default_embed)([case.search_text for case in self.cases])
        if len(vectors) != len(self.cases):
            raise CaseLibraryError("The embedder returned the wrong number of vectors.")

        try:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dimensions, distance=models.Distance.COSINE
                ),
            )
            # Embedded mode can retain points across delete/create (see
            # VectorStore.recreate_collection), which would keep removed cases.
            if self.client.count(self.collection, exact=True).count:
                self.client.delete(
                    collection_name=self.collection,
                    points_selector=models.FilterSelector(filter=models.Filter()),
                    wait=True,
                )
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=case.case_id,
                        vector=list(vector),
                        payload={**case.to_payload(), "library_version": self.version},
                    )
                    for case, vector in zip(self.cases, vectors)
                ],
                wait=True,
            )
        except Exception as exc:
            raise CaseLibraryError(f"Could not build the case library: {exc}") from exc
        return True

    # --- search --------------------------------------------------------------

    def find(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = config.CASE_TOP_K,
        min_score: float = config.CASE_MIN_SCORE,
        margin: float = config.CASE_SCORE_MARGIN,
        exclude: Callable[[Case], bool] | None = None,
    ) -> list[CaseMatch]:
        """Best-matching cases above ``min_score``.

        Runners-up are kept only within ``margin`` of the best match, so one
        clearly relevant case is not padded out with a loosely related one.
        Cases for which ``exclude`` returns True are skipped before the
        margin is measured.
        """
        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=list(query_vector),
                limit=top_k + (1 if exclude else 0),
                score_threshold=min_score,
                with_payload=True,
            )
        except Exception as exc:
            raise CaseLibraryError(f"Case search failed: {exc}") from exc

        matches = [
            CaseMatch(case=Case.from_payload(point.payload or {}), score=float(point.score))
            for point in response.points
        ]
        if exclude:
            matches = [m for m in matches if not exclude(m.case)]
        matches = matches[:top_k]
        if not matches:
            return []
        best = matches[0].score
        return [m for m in matches if best - m.score <= margin]


def match_question(question: str, library: CaseLibrary) -> list[CaseMatch]:
    """Cases that illustrate the law behind ``question``.

    When the question names a case ("summarise Ohimen v Adjei"), that case is
    the answer itself, so it is not suggested again as a related case.
    """
    question = (question or "").strip()
    if not question:
        return []
    from src import embeddings
    from src.case_lookup import extract_case_reference

    reference = extract_case_reference(question)
    exclude = None
    if reference is not None:
        def exclude(case: Case) -> bool:
            words = set(re.findall(r"[^\W_]+", case.name.lower()))
            return all(token in words for token in reference.tokens)

    return library.find(embeddings.embed_query(question), exclude=exclude)
