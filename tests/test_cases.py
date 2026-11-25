"""Case library tests.

The starter library is validated as data. Sync and matching run against
qdrant-client's embedded mode with deterministic fake vectors, like
``test_vector_store.py``; the real-model calibration behind
``config.CASE_MIN_SCORE`` is recorded in the comment above that setting.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src import config
from src.cases import Case, CaseLibrary, CaseLibraryError, load_cases, match_question

DIM = 8
# Each fake "concept" is one axis, so similarity is easy to reason about.
AXES = ["privity", "assault", "battery", "frustration", "duress"]


def _unit(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def _concept_vector(text: str) -> list[float]:
    lowered = text.lower()
    vec = [1.0 if axis in lowered else 0.0 for axis in AXES]
    vec += [0.0] * (DIM - len(vec))
    vec[-1] = 0.1  # never the zero vector
    return _unit(vec)


def fake_embed(texts):
    return [_concept_vector(t) for t in texts]


def _case(name: str, topic: str, **overrides) -> Case:
    fields = dict(
        name=name,
        court="Court of Appeal",
        year=1900,
        jurisdiction="England & Wales",
        subject="Contracts",
        principle=f"A principle about {topic}.",
        facts="Some facts.",
        held="Some holding.",
        topics=(topic,),
    )
    fields.update(overrides)
    return Case(**fields)


SMALL_LIBRARY = [
    _case("Tweddle v Atkinson", "privity"),
    _case("Stephens v Myers", "assault", subject="Torts"),
    _case("Taylor v Caldwell", "frustration"),
]


@pytest.fixture
def client(tmp_path: Path) -> QdrantClient:
    return QdrantClient(path=str(tmp_path / "cases"))


def _library(client: QdrantClient, cases=SMALL_LIBRARY) -> CaseLibrary:
    return CaseLibrary(client, cases=cases, collection="test_cases", dimensions=DIM)


# --- the starter library itself ----------------------------------------------


def test_starter_library_is_valid():
    cases = load_cases()
    assert len(cases) >= 50
    assert len({c.name for c in cases}) == len(cases)
    for case in cases:
        assert 1600 <= case.year <= 2030, case.name
        assert case.topics, f"{case.name} has no topics"
        assert case.principle.endswith("."), case.name


def test_starter_library_covers_the_core_subjects():
    subjects = {c.subject for c in load_cases()}
    assert {"Torts", "Contracts", "Criminal Law", "Constitutional Law"} <= subjects


# --- loading -----------------------------------------------------------------


def _write(tmp_path: Path, payload) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_rejects_missing_fields(tmp_path: Path):
    entry = _case("X v Y", "privity").to_payload()
    del entry["held"]
    with pytest.raises(CaseLibraryError, match="missing: held"):
        load_cases(_write(tmp_path, [entry]))


def test_load_rejects_duplicates(tmp_path: Path):
    entry = _case("X v Y", "privity").to_payload()
    with pytest.raises(CaseLibraryError, match="duplicate"):
        load_cases(_write(tmp_path, [entry, entry]))


def test_load_rejects_non_list(tmp_path: Path):
    with pytest.raises(CaseLibraryError, match="JSON list"):
        load_cases(_write(tmp_path, {"name": "X v Y"}))


def test_load_reports_missing_file(tmp_path: Path):
    with pytest.raises(CaseLibraryError, match="not found"):
        load_cases(tmp_path / "nope.json")


# --- sync --------------------------------------------------------------------


def test_sync_builds_once_then_is_current(client):
    library = _library(client)
    assert not library.is_current()
    assert library.sync(embed=fake_embed) is True
    assert library.is_current()
    assert library.sync(embed=fake_embed) is False
    assert client.count("test_cases", exact=True).count == len(SMALL_LIBRARY)


def test_sync_rebuilds_when_the_library_changes(client):
    _library(client).sync(embed=fake_embed)

    edited = SMALL_LIBRARY[:2] + [_case("Krell v Henry", "frustration")]
    library = _library(client, cases=edited)
    assert not library.is_current()
    assert library.sync(embed=fake_embed) is True

    names = {
        r.payload["name"]
        for r in client.scroll("test_cases", limit=10, with_payload=True)[0]
    }
    assert names == {"Tweddle v Atkinson", "Stephens v Myers", "Krell v Henry"}


def test_version_changes_with_the_embedding_model():
    from src.cases import library_version

    assert library_version(SMALL_LIBRARY, "model-a") != library_version(
        SMALL_LIBRARY, "model-b"
    )


# --- matching ----------------------------------------------------------------


def test_find_returns_the_matching_case(client):
    library = _library(client)
    library.sync(embed=fake_embed)
    matches = library.find(_concept_vector("privity"), min_score=0.5)
    assert [m.case.name for m in matches] == ["Tweddle v Atkinson"]
    assert matches[0].case.topics == ("privity",)


def test_find_returns_nothing_below_threshold(client):
    library = _library(client)
    library.sync(embed=fake_embed)
    assert library.find(_concept_vector("duress"), min_score=0.5) == []


def test_find_drops_runner_up_outside_margin(client):
    cases = [
        _case("Strong", "privity"),
        _case("Weaker", "privity assault"),  # half privity
    ]
    library = _library(client, cases=cases)
    library.sync(embed=fake_embed)
    query = _concept_vector("privity")

    tight = library.find(query, top_k=2, min_score=0.1, margin=0.05)
    assert [m.case.name for m in tight] == ["Strong"]

    loose = library.find(query, top_k=2, min_score=0.1, margin=1.0)
    assert [m.case.name for m in loose] == ["Strong", "Weaker"]


def test_named_case_is_not_suggested_back(client, monkeypatch):
    from src import embeddings

    cases = [
        _case("Ohimen v Adjei", "privity"),
        _case("Tweddle v Atkinson", "privity assault"),
    ]
    library = _library(client, cases=cases)
    library.sync(embed=fake_embed)
    monkeypatch.setattr(embeddings, "embed_query", lambda q: _concept_vector("privity"))

    # Without the name in the question, the case is the best match...
    assert match_question("What is privity?", library)[0].case.name == "Ohimen v Adjei"
    # ...but when the question asks about it, the next case is suggested instead.
    matches = match_question("Summarise Ohimen vs Adjei", library)
    assert [m.case.name for m in matches] == ["Tweddle v Atkinson"]


def test_match_question_ignores_blank_questions(client):
    library = _library(client)
    assert match_question("   ", library) == []


def test_unreachable_store_raises_library_error(client):
    library = _library(client)
    client.close()
    with pytest.raises(CaseLibraryError):
        library.is_current()


def test_defaults_come_from_config():
    assert config.CASE_TOP_K >= 1
    assert 0.0 < config.CASE_MIN_SCORE < 1.0
    assert config.CASES_PATH.name == "cases.json"
