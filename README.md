---
title: Legal Study Assistant
emoji: ⚖️
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.65.0
app_file: app.py
pinned: false
license: mit
short_description: RAG study assistant over your own law course materials
---

# ⚖️ Legal Study Assistant

A Retrieval-Augmented Generation study aid that answers questions **strictly from
your own course materials** — PDFs and lecture slides you upload — and shows the
file and page or slide behind every answer.

> **Study aid, not legal advice.** Answers are generated from your documents and
> may be incomplete or wrong. Verify every citation against the primary source.

---

## What it does

- **Upload** `.pdf`, `.pptx` and `.docx` files through the browser; they are
  parsed, chunked, embedded and indexed live.
- **Ask** natural-language questions and get answers grounded in those
  documents, with `[1]`-style markers tied to `file.pdf — p. 12`,
  `deck.pptx — slide 4` or `notes.docx — Promissory Estoppel`.
- **Scope** questions by subject (Contracts, Torts, …) or to specific documents.
- **Study tools**: summarise a topic, generate practice questions, define a term
  — all from your materials, all cited.
- **Persists** outside the app, so the knowledge base survives restarts.

### How it avoids inventing law

Fabricated case names and holdings are the main risk in a tool like this, so
grounding is enforced in three independent places:

1. **Retrieval fails closed.** If nothing clears the similarity threshold, the
   LLM is never called — the app returns *"I couldn't find this in your
   materials."* A model asked with no context answers from training data, which
   is exactly the failure to avoid.
2. **A calibrated threshold.** `SCORE_THRESHOLD = 0.45`, measured against
   bge-base on legal text (on-topic 0.63–0.80; off-topic non-legal 0.23–0.36).
3. **A strict system prompt.** No threshold separates "legal but not in your
   materials" from "on topic" — bge rates all legal prose as broadly similar —
   so the prompt must decline when the excerpts do not actually answer the
   question. See [src/prompts.py](src/prompts.py).

---

## Architecture

```
UPLOAD   file → parse (PyMuPDF / python-pptx / python-docx) → chunk + metadata
                → embed (bge-base, local CPU) → upsert to Qdrant

QUERY    question → embed → Qdrant search (top-k, subject/source filter)
                 → grounded prompt → Groq gpt-oss-120b → answer + citations
```

Raw uploads are transient. **The vector store is the source of truth**, and it
lives in Qdrant Cloud outside the Space, which is what makes the knowledge base
survive a Space sleeping or rebuilding.

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| UI / host | Streamlit on Hugging Face Spaces (`sdk: streamlit`, free CPU Basic) |
| PDF / PPTX / DOCX | `pymupdf` · `python-pptx` · `python-docx` |
| Chunking | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) |
| Embeddings | `sentence-transformers`, `BAAI/bge-base-en-v1.5` (local, CPU, 768-dim) |
| Vector store | Qdrant Cloud (free 1 GB) · embedded local mode for offline dev |
| LLM | Groq `openai/gpt-oss-120b` (free tier) |

---

## Project layout

```
app.py                  Streamlit UI: upload, knowledge base, chat, study tools
src/config.py           Env vars, model names, chunking and retrieval constants
src/ingestion.py        PDF/PPTX/DOCX → metadata-rich chunks
src/embeddings.py       bge-base loader, document/query embedding
src/vector_store.py     Qdrant wrapper: collection, upsert, search, listing
src/prompts.py          Grounding system prompt + study-task instructions
src/rag.py              Retrieval → prompt → Groq → cited answer
tests/                  99 tests; the slow ones load the real embedding model
```

---

## Running it locally

**Requires Python 3.11 or 3.12.** Avoid 3.14 — the ML stack is still catching up.

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt
cp .env.example .env                              # then fill it in
streamlit run app.py
```

Using [uv](https://docs.astral.sh/uv/) instead, which can fetch Python for you:

```bash
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` (git-ignored) and fill in:

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | yes | Answer generation. Free, no card: [console.groq.com](https://console.groq.com) |
| `QDRANT_URL` | yes¹ | Qdrant Cloud endpoint: [cloud.qdrant.io](https://cloud.qdrant.io) |
| `QDRANT_API_KEY` | yes¹ | Qdrant Cloud auth |
| `QDRANT_COLLECTION` | no | Collection name. Default `legal_study_rag` |
| `LLM_MODEL` | no | Default `openai/gpt-oss-120b` |
| `EMBED_MODEL` | no | Default `BAAI/bge-base-en-v1.5` |
| `USE_LOCAL_STORE` | no | `true` for an embedded on-disk store (offline dev) |
| `LOCAL_STORE_PATH` | no | Where that store lives. Default `.local_store` |

¹ Not needed when `USE_LOCAL_STORE=true`.

To develop with no Qdrant account at all:

```bash
USE_LOCAL_STORE=true GROQ_API_KEY=... streamlit run app.py
```

### Tests

```bash
.venv/Scripts/python.exe -m pytest
```

`pytest -m "not slow"` skips the tests that load the real embedding model
(~440 MB on first run) and finishes in seconds.

---

## Deploying to Hugging Face Spaces

1. **New Space** → SDK **Streamlit** → hardware **CPU Basic (free)**.
   Use the native Streamlit SDK, not Docker — Docker Spaces no longer get free
   hosting for new accounts.
2. **Settings → Repository secrets**: add `GROQ_API_KEY`, `QDRANT_URL`,
   `QDRANT_API_KEY`. Never commit these.
3. Push:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space>
   git push space main
   ```
4. Watch the build log. **First boot is slow** — the Space downloads the
   embedding model (~440 MB). Later cold starts take 30–90 s.
5. Verify persistence: index a document, let the Space sleep, wake it, and
   confirm the document is still listed and queryable.

---

## Limitations

- **Scanned PDFs are not supported.** Text is extracted, not OCR'd; an
  image-only PDF is rejected with a message saying so.
- **Word files are cited by heading, not page.** Word stores no page numbers —
  pagination is produced by whatever renders the document — so a `.docx` chunk
  cites the heading it sits under, falling back to a section number when the
  document has no headings. Well-structured notes therefore cite far more
  precisely than one unbroken wall of text.
- **Legacy `.doc` and `.ppt` are not readable.** They are a different binary
  format; the app says so and asks for a `.docx`/`.pptx` copy instead.
- **Groq free tier** is roughly 30 requests/minute and ~1,000/day. Fine for one
  student, not for a public audience.
- **Cold starts.** Free Spaces sleep when idle; the first question after a wake
  waits for the model to load.
- **Changing `EMBED_MODEL` invalidates the collection.** Vector size must match
  (bge-base 768, MiniLM 384). The app detects the mismatch and offers to
  recreate the collection — which deletes indexed documents, so they must be
  re-uploaded.
- **Low-RAM hosts.** If 16 GB is unavailable (e.g. Streamlit Community Cloud at
  ~1 GB), switch to `EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2` and
  recreate the collection.
- **Retrieval quality bounds the answer.** The app will not invent law, but it
  can miss material that is present and phrased very differently.

---

## Note for Windows developers

If `pip`/`uv` installs fail with `os error 396` ("cloud operation cannot be
performed on a file with incompatible hardlinks") or hang for many minutes, a
filesystem filter driver is rejecting hardlinks. Install with copies instead:

```bash
UV_LINK_MODE=copy uv pip install -r requirements.txt
```

A package left half-installed by such a failure can fail later in confusing
ways. To check for it:

```bash
.venv/Scripts/python.exe -c "import dateutil.tz, pandas, torch; print('ok')"
```
