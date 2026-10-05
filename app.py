"""Legal Study RAG - Streamlit entry point.

Two loops over one vector store: upload/index on the left, ask/answer in the
middle. Every answer is rendered with the file and page or slide it came from.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Sequence

import streamlit as st

# Streamlit Community Cloud supplies configuration through st.secrets, while
# local development and most other hosts use environment variables. Copy one
# into the other before src.config is imported, so there is a single way to
# read configuration everywhere. setdefault means a real environment variable
# still wins, which keeps local overrides working.
try:  # pragma: no cover - depends on the host, not on our logic
    for _key, _value in st.secrets.items():
        if isinstance(_value, (str, int, float, bool)):
            os.environ.setdefault(_key, str(_value))
except Exception:
    # No secrets.toml and no Cloud secrets configured: fall back to the
    # environment alone. config.missing_secrets() reports anything absent.
    pass

from src import config, embeddings, prompts, rag  # noqa: E402
from src.ingestion import (  # noqa: E402
    Chunk,
    DocumentParseError,
    UnsupportedFileTypeError,
    ingest_file,
)
from src.vector_store import (  # noqa: E402
    DimensionMismatchError,
    SearchHit,
    SourceSummary,
    VectorStore,
    VectorStoreError,
)

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Cached resources -------------------------------------------------------


@st.cache_resource(show_spinner="Connecting to the knowledge base...")
def get_store() -> VectorStore:
    store = VectorStore()
    store.ensure_collection()
    return store


@st.cache_resource(show_spinner="Loading the embedding model (first run is slow)...")
def load_embedding_model():
    return embeddings.get_model()


@st.cache_data(show_spinner=False, ttl=300)
def list_sources_cached() -> list[SourceSummary]:
    """Indexed documents, cached until a write invalidates it.

    Deliberately not keyed on a session counter: ``st.cache_data`` is shared
    across sessions, so a session-scoped key would serve one session a result
    cached under another session's counter. Writes call ``.clear()`` instead.

    Errors deliberately propagate. Returning an empty list on failure would
    make a store outage look exactly like an empty library, and a student
    would reasonably conclude her documents had been lost and re-upload them.
    """
    return get_store().list_sources()


def invalidate_kb_cache() -> None:
    """Call after any write so the sidebar reflects it immediately."""
    list_sources_cached.clear()


# --- Header -----------------------------------------------------------------


def render_header() -> None:
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.caption(config.APP_TAGLINE)
    st.warning(config.DISCLAIMER, icon="⚠️")


def render_setup_help(missing: Sequence[str]) -> None:
    st.error(
        "**Not configured yet.** These settings are missing: "
        + ", ".join(f"`{name}`" for name in missing),
        icon="🔑",
    )
    st.markdown(
        """
Create a `.env` file next to `app.py` (copy `.env.example`) with:

```
GROQ_API_KEY=...        # free, no card: https://console.groq.com
QDRANT_URL=...          # free 1GB cluster: https://cloud.qdrant.io
QDRANT_API_KEY=...
```

On Hugging Face Spaces, add the same names under
**Settings → Repository secrets** instead.

To work offline without a Qdrant cluster, set `USE_LOCAL_STORE=true` - a
local on-disk store is used instead. A `GROQ_API_KEY` is still needed to
generate answers.
        """
    )


# --- Sidebar: upload and knowledge base -------------------------------------


def _subject_options(sources: Sequence[SourceSummary]) -> list[str]:
    known = {s.subject for s in sources if s.subject}
    return sorted(known | set(config.DEFAULT_SUBJECTS))


def _validate_upload(upload) -> str | None:
    """Return an error message, or None when the file is acceptable."""
    suffix = Path(upload.name).suffix.lower()
    if suffix not in config.SUPPORTED_EXTENSIONS:
        return (
            f"{upload.name}: unsupported file type "
            f"(need {' or '.join(config.SUPPORTED_EXTENSIONS)})."
        )
    size_mb = upload.size / (1024 * 1024)
    if size_mb > config.MAX_FILE_MB:
        return (
            f"{upload.name}: {size_mb:.1f} MB exceeds the "
            f"{config.MAX_FILE_MB} MB limit."
        )
    if upload.size == 0:
        return f"{upload.name}: the file is empty."
    return None


def _index_one(upload, subject: str, store: VectorStore, progress) -> int:
    """Parse, embed and upsert a single upload. Returns chunks written."""
    suffix = Path(upload.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(upload.getbuffer())
        temp_path = Path(handle.name)

    try:
        chunks: list[Chunk] = ingest_file(
            temp_path, subject=subject, source_name=upload.name
        )
        progress(f"{upload.name}: {len(chunks)} chunks - embedding...")

        vectors = embeddings.embed_chunks(chunks)

        # Replace rather than merge, so a re-upload of an edited document does
        # not leave orphaned chunks from the previous version behind.
        store.delete_source(upload.name)
        return store.upsert_chunks(chunks, vectors)
    finally:
        temp_path.unlink(missing_ok=True)


def render_upload_panel(store: VectorStore, sources: Sequence[SourceSummary]) -> None:
    st.subheader("Add materials")

    uploader_key = f"uploader_{st.session_state.get('uploader_round', 0)}"
    uploads = st.file_uploader(
        "PDFs, lecture slides and Word documents",
        type=["pdf", "pptx", "docx"],
        accept_multiple_files=True,
        key=uploader_key,
        help=f"Up to {config.MAX_FILE_MB} MB per file.",
    )

    options = _subject_options(sources)
    subject = st.selectbox(
        "Subject",
        options,
        index=options.index("General") if "General" in options else 0,
        help="Tagged onto every chunk from these files, so you can scope questions later.",
    )
    custom_subject = st.text_input(
        "...or a new subject", placeholder="e.g. Company Law"
    ).strip()
    effective_subject = custom_subject or subject

    if not st.button(
        "Index files", type="primary", use_container_width=True, disabled=not uploads
    ):
        return

    valid, problems = [], []
    for upload in uploads or []:
        error = _validate_upload(upload)
        (problems if error else valid).append(error or upload)

    for problem in problems:
        st.warning(problem, icon="⚠️")
    if not valid:
        return

    indexed, failed = [], []
    with st.status(f"Indexing {len(valid)} file(s)...", expanded=True) as status:
        status.write("Loading the embedding model...")
        try:
            load_embedding_model()
        except Exception as exc:
            status.update(label="Could not load the embedding model", state="error")
            st.error(f"{exc}", icon="🚨")
            return

        bar = st.progress(0.0)
        for position, upload in enumerate(valid, start=1):
            try:
                written = _index_one(
                    upload, effective_subject, store, lambda m: status.write(m)
                )
                indexed.append((upload.name, written))
                status.write(f"✅ {upload.name} - {written} chunks indexed")
            except (DocumentParseError, UnsupportedFileTypeError) as exc:
                failed.append((upload.name, str(exc)))
                status.write(f"❌ {upload.name} - {exc}")
            except (VectorStoreError, DimensionMismatchError) as exc:
                failed.append((upload.name, str(exc)))
                status.write(f"❌ {upload.name} - {exc}")
            except Exception as exc:  # last-resort guard: never crash the app
                failed.append((upload.name, f"Unexpected error: {exc}"))
                status.write(f"❌ {upload.name} - unexpected error: {exc}")
            bar.progress(position / len(valid))

        state = "complete" if indexed and not failed else (
            "error" if not indexed else "complete"
        )
        status.update(
            label=f"Indexed {len(indexed)} of {len(valid)} file(s)", state=state
        )

    if indexed:
        invalidate_kb_cache()
        total = sum(count for _, count in indexed)
        st.toast(f"Indexed {total} chunks from {len(indexed)} file(s).", icon="✅")
        st.session_state["uploader_round"] = (
            st.session_state.get("uploader_round", 0) + 1
        )
        st.rerun()

    for name, message in failed:
        st.error(f"**{name}** - {message}", icon="🚨")


def render_knowledge_base(store: VectorStore, sources: Sequence[SourceSummary]) -> None:
    st.subheader("Knowledge base")

    if not sources:
        st.info(
            "No documents indexed yet. Upload a PDF, a deck or a Word file above.",
            icon="📭",
        )
        return

    total_chunks = sum(s.chunk_count for s in sources)
    st.caption(f"**{len(sources)}** document(s) · **{total_chunks}** chunks")

    for summary in sources:
        icon = {"pdf": "📄", "pptx": "📊", "docx": "📝"}.get(summary.doc_type, "📄")
        with st.expander(f"{icon} {summary.source}", expanded=False):
            st.caption(
                f"Subject: **{summary.subject}** · {summary.chunk_count} chunks"
            )
            if st.button(
                "Remove from knowledge base",
                key=f"delete_{summary.source}",
                use_container_width=True,
            ):
                try:
                    store.delete_source(summary.source)
                    invalidate_kb_cache()
                    st.toast(f"Removed {summary.source}.", icon="🗑️")
                    st.rerun()
                except VectorStoreError as exc:
                    st.error(str(exc), icon="🚨")


# --- Citations --------------------------------------------------------------


def render_citations(hits: Sequence[SearchHit]) -> None:
    if not hits:
        return
    with st.expander(f"Sources ({len(hits)} excerpt(s))", expanded=False):
        for position, hit in enumerate(hits, start=1):
            st.markdown(f"**[{position}] {hit.citation}** · similarity {hit.score:.2f}")
            st.caption(hit.text)
            if position < len(hits):
                st.divider()


# --- Ask tab ----------------------------------------------------------------


def render_chat(store: VectorStore, subject: str | None, sources: list[str]) -> None:
    history = st.session_state.setdefault("messages", [])

    # Reserve the conversation area *before* declaring the input, then write
    # every message into it. st.chat_input only pins itself to the bottom of
    # the viewport when it sits in the main page body; nested in a tab it
    # renders inline, so without this the newest answer appears below the box
    # the student is about to type in again.
    conversation = st.container()
    question = st.chat_input("Ask a question about your materials...")

    with conversation:
        for message in history:
            with st.chat_message(message["role"]):
                if message["content"]:
                    st.markdown(message["content"])
                else:
                    st.caption("_Interrupted - ask again to get an answer._")
                if message["role"] == "assistant":
                    render_citations(message.get("hits", []))

        if not question:
            return

        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Reserve the answer slot before streaming. Submitting another question
        # mid-stream cancels this script run, and without the placeholder the
        # question would be stranded in the history with nothing beneath it.
        pending = {"role": "assistant", "content": "", "hits": []}
        history.append(pending)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your materials..."):
                    hits, stream = rag.ask_stream(
                        question, store=store, subject=subject, sources=sources or None
                    )
                answer = st.write_stream(stream)
            except rag.LLMError as exc:
                answer, hits = f"⚠️ {exc}", []
                st.error(str(exc), icon="🚨")
            except VectorStoreError as exc:
                answer, hits = f"⚠️ {exc}", []
                st.error(str(exc), icon="🚨")
            except Exception as exc:
                answer, hits = f"⚠️ Unexpected error: {exc}", []
                st.error(f"Unexpected error: {exc}", icon="🚨")
            else:
                render_citations(hits)

        pending["content"] = answer
        pending["hits"] = list(hits)


# --- Study tools tab (Phase 5) ----------------------------------------------


def _run_task(instruction: str, target: str, store, subject, sources) -> None:
    try:
        with st.spinner("Working through your materials..."):
            result = rag.run_study_task(
                instruction, target, store=store, subject=subject, sources=sources or None
            )
    except (rag.LLMError, VectorStoreError) as exc:
        st.error(str(exc), icon="🚨")
        return

    st.markdown(result.answer)
    render_citations(result.hits)


def render_study_tools(
    store: VectorStore, subject: str | None, sources: list[str]
) -> None:
    st.caption(
        "Every tool below works only from your indexed materials, and cites them."
    )
    summarise, practice, define = st.tabs(
        ["Summarise a topic", "Practice questions", "Define a term"]
    )

    with summarise:
        topic = st.text_input(
            "Topic, case or chapter", key="summarise_topic", placeholder="e.g. Consideration"
        )
        if st.button("Summarise", key="summarise_go", disabled=not topic.strip()):
            _run_task(prompts.SUMMARISE_INSTRUCTION, topic, store, subject, sources)

    with practice:
        area = st.text_input(
            "Area to be tested on", key="practice_topic", placeholder="e.g. Negligence"
        )
        if st.button("Generate questions", key="practice_go", disabled=not area.strip()):
            _run_task(
                prompts.PRACTICE_QUESTIONS_INSTRUCTION, area, store, subject, sources
            )

    with define:
        term = st.text_input(
            "Term", key="define_term", placeholder="e.g. Promissory estoppel"
        )
        if st.button("Define", key="define_go", disabled=not term.strip()):
            _run_task(prompts.DEFINE_INSTRUCTION, term, store, subject, sources)


# --- Main -------------------------------------------------------------------


def main() -> None:
    render_header()

    missing = config.missing_secrets()
    if missing:
        with st.sidebar:
            st.subheader("Setup")
            st.error("Missing: " + ", ".join(f"`{m}`" for m in missing), icon="🔑")
        render_setup_help(missing)
        return

    try:
        store = get_store()
    except DimensionMismatchError as exc:
        st.error(str(exc), icon="🚨")
        if st.button("Recreate the collection (deletes all indexed documents)"):
            VectorStore().recreate_collection()
            get_store.clear()
            invalidate_kb_cache()
            st.rerun()
        return
    except VectorStoreError as exc:
        st.error(str(exc), icon="🚨")
        return

    try:
        sources = list_sources_cached()
    except VectorStoreError as exc:
        st.error(f"Could not read the knowledge base: {exc}", icon="🚨")
        st.caption(
            "This is a connection problem, not data loss - indexed documents "
            "live in Qdrant, not in this app. Check `QDRANT_URL` and "
            "`QDRANT_API_KEY`, and that the cluster is running."
        )
        return

    with st.sidebar:
        render_upload_panel(store, sources)
        st.divider()
        render_knowledge_base(store, sources)
        st.divider()
        store_label = "Local (embedded)" if config.USE_LOCAL_STORE else "Qdrant Cloud"
        st.caption(
            f"**Embeddings:** `{config.EMBED_MODEL}`  \n"
            f"**LLM:** `{config.LLM_MODEL}`  \n"
            f"**Vector store:** {store_label}"
        )

    if not sources:
        st.info(
            "Upload your course PDFs, lecture slides or Word documents in the "
            "sidebar to get started.",
            icon="👈",
        )
        return

    subjects = sorted({s.subject for s in sources if s.subject})
    filter_col, source_col = st.columns([1, 2])
    with filter_col:
        chosen_subject = st.selectbox("Subject filter", ["All subjects"] + subjects)
    with source_col:
        chosen_sources = st.multiselect(
            "Limit to specific documents (optional)",
            [s.source for s in sources],
        )
    subject = None if chosen_subject == "All subjects" else chosen_subject

    ask_tab, tools_tab = st.tabs(["Ask", "Study tools"])
    with ask_tab:
        render_chat(store, subject, chosen_sources)
    with tools_tab:
        render_study_tools(store, subject, chosen_sources)


if __name__ == "__main__":
    main()
