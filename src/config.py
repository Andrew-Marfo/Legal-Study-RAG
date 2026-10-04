"""Central configuration for the Legal Study RAG assistant.

All tunables live here. Values are read once at import time from the
environment (a local ``.env`` during development, Repository Secrets on
Hugging Face Spaces). Nothing in this module raises on import: a missing
secret is reported through :func:`missing_secrets` so the UI can show a
friendly message instead of crashing at startup.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root if present. Real deployments (HF Spaces)
# inject the same names as real environment variables, which take precedence.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(name: str, default: str = "") -> str:
    """Read an env var, trimming whitespace and treating blanks as unset."""
    return (os.getenv(name) or default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# --- Application ------------------------------------------------------------

APP_TITLE = "Legal Study Assistant"
APP_ICON = "⚖️"
APP_TAGLINE = "Ask questions about your own course materials — with sources."

DISCLAIMER = (
    "**Study aid — not legal advice.** Answers are generated from the documents "
    "you upload and may be incomplete or wrong. Always verify every citation "
    "against the primary source before relying on it."
)

# --- LLM (generation) -------------------------------------------------------

GROQ_API_KEY = _env("GROQ_API_KEY")

# Groq retires models regularly - llama-3.3-70b-versatile was decommissioned
# and now 404s. Check `client.models.list()` if generation starts failing with
# model_not_found, and set LLM_MODEL to a current one without touching code.
# Known good alternatives: openai/gpt-oss-20b, qwen/qwen3.8-27b.
LLM_MODEL = _env("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1536

# --- Embeddings -------------------------------------------------------------

EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

# Vector dimensions per supported embedding model. Used to size the Qdrant
# collection; a mismatch here silently breaks search, so keep it explicit.
EMBED_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-MiniLM-L6-v2": 384,
}

# bge models are trained with an instruction prefix on the *query* side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

EMBED_BATCH_SIZE = 32

# --- Chunking ---------------------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# Chunks shorter than this are dropped (page numbers, headers, empty slides).
MIN_CHUNK_CHARS = 50

# --- Vector store -----------------------------------------------------------

QDRANT_URL = _env("QDRANT_URL")
QDRANT_API_KEY = _env("QDRANT_API_KEY")
QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "legal_study_rag")

# Offline development mode. The local store is qdrant-client's embedded
# on-disk mode rather than Chroma: it is the same API as the cloud path, so
# there is one implementation to keep correct, and no extra dependency.
# USE_LOCAL_CHROMA is accepted as an alias for backwards compatibility.
USE_LOCAL_STORE = _env_bool("USE_LOCAL_STORE", False) or _env_bool(
    "USE_LOCAL_CHROMA", False
)
LOCAL_STORE_PATH = _env("LOCAL_STORE_PATH") or _env(
    "CHROMA_PATH", str(PROJECT_ROOT / ".local_store")
)

# --- Retrieval --------------------------------------------------------------

TOP_K = 6

# Cosine similarity floor, calibrated against bge-base-en-v1.5 on legal text.
# Measured top-1 scores for an indexed contracts document:
#
#   on-topic questions            0.63 - 0.80
#   off-topic, non-legal          0.23 - 0.36  ("how do I change a car tyre")
#   off-topic but legal           0.51 - 0.55  ("rule against perpetuities")
#
# 0.45 sits in the gap: it discards clearly unrelated text without costing
# recall on genuine questions. Note that no threshold separates the third row
# from the first - bge rates any legal prose as broadly similar - so a question
# about law the student has not uploaded WILL retrieve chunks. That case is
# caught by the system prompt, which must decline when the excerpts do not
# actually answer the question. Threshold and prompt are both load-bearing.
SCORE_THRESHOLD = 0.45

# --- Uploads ----------------------------------------------------------------

SUPPORTED_EXTENSIONS = (".pdf", ".pptx")
MAX_FILE_MB = 50

# Subjects offered in the UI. "General" is the fallback when none is chosen.
DEFAULT_SUBJECTS = [
    "General",
    "Constitutional Law",
    "Contracts",
    "Criminal Law",
    "Equity & Trusts",
    "Evidence",
    "Family Law",
    "Jurisprudence",
    "Land Law",
    "Torts",
]


# --- Helpers ----------------------------------------------------------------


def embedding_dimensions(model_name: str | None = None) -> int:
    """Vector size for ``model_name``.

    Raises a clear error for an unknown model rather than letting a wrong
    collection size fail obscurely at search time.
    """
    name = model_name or EMBED_MODEL
    try:
        return EMBED_DIMENSIONS[name]
    except KeyError as exc:
        known = ", ".join(sorted(EMBED_DIMENSIONS))
        raise ValueError(
            f"Unknown embedding model {name!r}. Add its vector size to "
            f"EMBED_DIMENSIONS in src/config.py. Known models: {known}"
        ) from exc


def missing_secrets() -> list[str]:
    """Names of required secrets that are not set, given the current mode."""
    missing: list[str] = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not USE_LOCAL_STORE:
        if not QDRANT_URL:
            missing.append("QDRANT_URL")
        if not QDRANT_API_KEY:
            missing.append("QDRANT_API_KEY")
    return missing


def is_configured() -> bool:
    """True when every secret needed for a full query round-trip is present."""
    return not missing_secrets()
