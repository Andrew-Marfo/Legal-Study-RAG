"""Find a named case in the student's materials and read its judgment in order.

Semantic search cannot answer "summarise Ohimen v Adjei". In a casebook every
chunk is about the same subject, so a name-only question scores almost the same
against all of them, and the party names - rare proper nouns - barely move an
embedding. Worse, a reported judgment names itself once, in its heading; the
dozens of chunks that follow never repeat it, while later cases that cite it
do. Measured on a 313-page land-law casebook: the six retrieved chunks scored
0.634-0.642, and the heading of the actual judgment was not in the top 60.

So when a question names a case, this module does what a student would do with
the book: look the name up as text, find the place where it is the *heading*
(not a citation in someone else's judgment), and read on from there until the
next case begins or the context budget runs out.

Every failure here falls back to ordinary semantic retrieval; it never stops a
question being answered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from qdrant_client import models

from src import config
from src.vector_store import SearchHit, VectorStore

# "X v Y", "X v. Y", "X vs Y", "X vs. Y", "X versus Y". Up to six words a side.
_WORD = r"[A-Za-z][\w.'&-]*"
_CASE_PATTERN = re.compile(
    rf"({_WORD}(?:\s+{_WORD}){{0,5}})\s+(?:v|vs|versus)\.?\s+({_WORD}(?:\s+{_WORD}){{0,5}})",
    re.IGNORECASE,
)

# Words that sit next to "v" but do not identify a party.
_GENERIC = {
    "a", "an", "and", "another", "anor", "case", "co", "company", "corporation",
    "describe", "discuss", "ex", "explain", "in", "limited", "ltd", "of", "ors",
    "others", "parte", "plc", "re", "summarise", "summarize", "the", "what",
}

_YEAR = re.compile(r"[\(\[]\s*1[6-9]\d\d\s*[\)\]]|\b(?:18|19|20)\d\d\b")
_V_TOKEN = re.compile(r"\bV(?:S)?\.?\s", re.IGNORECASE)
# The line under a report title names the court or the judge.
_COURT_LINE = re.compile(
    r"^(in the )?(supreme|high|privy|full|land|divisional|west african|house of lords|"
    r"court|circuit|district)\b|\b[A-Z][A-Za-z'-]+,? (J|JA|JSC|CJ|C\.J|J\.A|J\.S\.C)\.?$",
)


@dataclass(frozen=True)
class CaseReference:
    """A case named in a question, reduced to one identifying word per party."""

    raw: str
    first_party: str
    second_party: str

    @property
    def tokens(self) -> list[str]:
        """Party words long enough to search on (drops the 'R' in 'R v Woollin')."""
        return [t for t in (self.first_party, self.second_party) if len(t) >= 3]


def _words(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[^\W_]+", text)]


def extract_case_reference(question: str) -> CaseReference | None:
    """The case named in ``question``, or None if it names none."""
    match = _CASE_PATTERN.search(question or "")
    if not match:
        return None
    left = [w for w in _words(match.group(1)) if w not in _GENERIC]
    right = [w for w in _words(match.group(2)) if w not in _GENERIC]
    if not left or not right:
        return None
    reference = CaseReference(
        raw=match.group(0).strip(" \"'"),
        first_party=left[-1],  # the party word nearest the "v"
        second_party=right[0],
    )
    return reference if reference.tokens else None


# --- Recognising headings ------------------------------------------------------


def _mostly_upper(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 6 and sum(c.isupper() for c in letters) / len(letters) >= 0.8


def _is_court_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 150:
        return False
    if _mostly_upper(stripped):
        return True
    return bool(_COURT_LINE.search(stripped) or _COURT_LINE.search(stripped.lower()))


def heading_score(lines: list[str], index: int, reference: CaseReference) -> int:
    """How strongly ``lines[index]`` looks like the title of the referenced case.

    0 means not a heading. A citation inside another judgment ("(3) Ohimen v.
    Adjei (1957)...", "...; Ohimen v Adjei cited") scores low; a report title
    ("OHIMEN V. ADJEI AND ANOTHER. (1957) 2 WALR 257") followed by the court
    scores high.
    """
    line = lines[index].strip()
    words = _words(line)
    if not words or not all(token in words for token in reference.tokens):
        return 0
    if line[0] in "([" or line[0].isdigit():
        return 0  # numbered "cases referred to" entry or footnote
    if not _V_TOKEN.search(line + " "):
        return 0

    score = 0
    upper = _mostly_upper(line)
    if upper:
        score += 3
    if words[0] == reference.first_party:
        score += 2
    if _YEAR.search(line):
        score += 1
    court_follows = any(_is_court_line(l) for l in lines[index + 1 : index + 3])
    if court_follows:
        score += 2
    if ";" in line or "cited" in line.lower():
        score -= 3
    # Mixed-case lines are usually citations in running text; a mixed-case
    # title counts only if it starts with the party and the court follows.
    if not upper and not (court_follows and words[0] == reference.first_party):
        return 0
    return max(score, 0)


def looks_like_new_case(line: str, reference: CaseReference) -> bool:
    """A report title for a *different* case: where the judgment being read ends."""
    stripped = line.strip()
    if len(stripped) > 200 or not _mostly_upper(stripped):
        return False
    if not _V_TOKEN.search(stripped + " ") or not _YEAR.search(stripped):
        return False
    words = _words(stripped)
    return not all(token in words for token in reference.tokens)


# --- Reading the judgment --------------------------------------------------------


def _scope_conditions(
    subject: str | None, sources: Iterable[str] | None
) -> list[models.Condition]:
    conditions: list[models.Condition] = []
    if subject:
        conditions.append(
            models.FieldCondition(key="subject", match=models.MatchValue(value=subject))
        )
    source_list = [s for s in (sources or []) if s]
    if source_list:
        conditions.append(
            models.FieldCondition(key="source", match=models.MatchAny(any=source_list))
        )
    return conditions


def _strip_overlap(previous: str, current: str, longest: int = 400) -> str:
    """Drop the start of ``current`` that repeats the end of ``previous``.

    Consecutive chunks overlap by CHUNK_OVERLAP characters; sending the overlap
    twice would spend the context budget on duplicate text.
    """
    for size in range(min(len(previous), len(current), longest), 19, -1):
        if previous.endswith(current[:size]):
            return current[size:].lstrip()
    return current


def find_heading(
    store: VectorStore,
    reference: CaseReference,
    *,
    subject: str | None = None,
    sources: Iterable[str] | None = None,
) -> tuple[dict[str, Any], int] | None:
    """The chunk payload where the case's report begins, and the heading's line number."""
    conditions = _scope_conditions(subject, sources) + [
        models.FieldCondition(
            key="text", match=models.MatchText(text=" ".join(reference.tokens))
        )
    ]
    records, _ = store.client.scroll(
        collection_name=store.collection,
        scroll_filter=models.Filter(must=conditions),
        limit=200,
        with_payload=True,
        with_vectors=False,
    )

    best: tuple[int, int, dict[str, Any], int] | None = None  # (score, -chunk, payload, line)
    for record in records:
        payload = dict(record.payload or {})
        lines = str(payload.get("text", "")).splitlines()
        for number in range(len(lines)):
            score = heading_score(lines, number, reference)
            if score < 3:
                continue
            rank = (score, -int(payload.get("chunk_index", 0)))
            if best is None or rank > best[:2]:
                best = (rank[0], rank[1], payload, number)
    if best is None:
        return None
    return best[2], best[3]


def read_case(
    store: VectorStore,
    reference: CaseReference,
    *,
    subject: str | None = None,
    sources: Iterable[str] | None = None,
    max_chars: int = config.CASE_READ_MAX_CHARS,
    max_chunks: int = config.CASE_READ_MAX_CHUNKS,
) -> list[SearchHit]:
    """The judgment's chunks in reading order, from its heading, within budget.

    Returns [] when the materials contain no report of the case (only
    citations of it), so the caller can fall back to semantic search.
    """
    found = find_heading(store, reference, subject=subject, sources=sources)
    if found is None:
        return []
    heading_payload, heading_line = found
    source = heading_payload["source"]
    start = int(heading_payload.get("chunk_index", 0))

    records, _ = store.client.scroll(
        collection_name=store.collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="source", match=models.MatchValue(value=source)),
                models.FieldCondition(
                    key="chunk_index",
                    range=models.Range(gte=start, lt=start + max_chunks),
                ),
            ]
        ),
        limit=max_chunks,
        with_payload=True,
        with_vectors=False,
    )
    ordered = sorted(
        (dict(r.payload or {}) for r in records), key=lambda p: int(p.get("chunk_index", 0))
    )

    hits: list[SearchHit] = []
    used = 0
    previous = ""
    for payload in ordered:
        original = str(payload.pop("text", ""))
        reached_next_case = False
        if int(payload.get("chunk_index", 0)) == start:
            # Text above the heading belongs to the previous case.
            text = "\n".join(original.splitlines()[heading_line:])
        else:
            lines = _strip_overlap(previous, original).splitlines()
            ends_at = next(
                (i for i, line in enumerate(lines) if looks_like_new_case(line, reference)),
                None,
            )
            if ends_at is not None:
                lines = lines[:ends_at]
                reached_next_case = True
            text = "\n".join(lines)
        previous = original

        if text.strip():
            if used + len(text) > max_chars:
                remaining = max_chars - used
                if remaining > 200:
                    hits.append(_hit(text[:remaining].rstrip() + " [...]", payload))
                break
            hits.append(_hit(text, payload))
            used += len(text)
        if reached_next_case:
            break
    return hits


def _hit(text: str, payload: dict[str, Any]) -> SearchHit:
    metadata = {k: v for k, v in payload.items() if k != "text"}
    metadata["matched_by"] = "case name"
    return SearchHit(text=text, score=1.0, metadata=metadata)


def lookup(
    question: str,
    store: VectorStore,
    *,
    subject: str | None = None,
    sources: Iterable[str] | None = None,
) -> list[SearchHit]:
    """The named case's judgment, or [] to fall back to semantic retrieval."""
    reference = extract_case_reference(question)
    if reference is None:
        return []
    try:
        return read_case(store, reference, subject=subject, sources=sources)
    except Exception:
        # A missing text index or a transient error must not block the answer.
        return []
