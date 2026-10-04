# Work Plan — Legal Study RAG Assistant

**Deliverable:** A Retrieval-Augmented Generation (RAG) web app that answers a law student's questions *from her own study materials* (PDFs and PowerPoint slides), with source citations. It has an upload UI that updates the knowledge base live, and it is hosted for free on Hugging Face Spaces.

**Audience for this document:** Claude Code (the builder). The human (Andrew) is the programmer driving the build and will review each phase.

**Build philosophy:** Build in the phases below *in order*. After each phase, stop, run the app/tests, and confirm the acceptance criteria before moving on. Do not scaffold everything at once.

---

## 1. Goal & success criteria

Build a tool that lets a law student:

1. Upload her course PDFs and lecture slides (`.pdf`, `.pptx`) through a web UI.
2. Have those files automatically parsed, chunked, embedded, and added to a persistent knowledge base.
3. Ask natural-language questions and get answers **grounded strictly in her materials**, with each answer showing *which file, page, or slide* it came from.
4. Scope questions by subject (e.g. Contracts, Torts, Constitutional Law).
5. Use it from any browser via a free public URL, with the knowledge base surviving app restarts.

**Definition of done:** A deployed, publicly reachable HF Space where a user can upload a PDF, ask a question about its contents, and receive a correct, source-cited answer — and where that uploaded document is still queryable after the Space sleeps and wakes.

---

## 2. Why this matters for a law student (design constraints)

These constraints are **not optional polish** — they shape the architecture:

- **Source attribution is mandatory.** A law student must verify and cite primary sources. Every answer must surface the originating file + page number (PDF) or slide number (PPTX). Store this as chunk metadata from day one.
- **No hallucinated law.** Fabricated case names, holdings, or citations are the single biggest risk. The generation prompt must enforce strict grounding: answer only from retrieved context, and explicitly say *"I couldn't find this in your materials"* when context is insufficient.
- **Not legal advice.** The UI must carry a persistent disclaimer: this is a study aid; all citations must be verified against primary sources.
- **Dense, long documents.** Legal texts are long and nuanced. Use a quality embedding model (not the smallest one) and generous chunk overlap so holdings/definitions aren't split mid-thought.

---

## 3. Locked technology stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | 3.12 OK; avoid 3.14 (dependency friction). |
| UI / host | **Streamlit**, deployed on **Hugging Face Spaces** (native Streamlit SDK, free CPU Basic: 2 vCPU / 16GB RAM) | Use `sdk: streamlit` in `README.md` front-matter — NOT Docker — to stay on free CPU. |
| PDF parsing | `PyMuPDF` (`fitz`) | Fast, gives per-page text + page numbers. |
| PPTX parsing | `python-pptx` | Iterate slides; capture slide numbers + notes. |
| Chunking/orchestration | `langchain` (+ `langchain-text-splitters`) | `RecursiveCharacterTextSplitter`. |
| Embeddings | `sentence-transformers`, model **`BAAI/bge-base-en-v1.5`** (local, CPU) | No API key, no rate limits. Fallback for low RAM: `all-MiniLM-L6-v2`. Add the `bge` query instruction prefix. |
| Vector store | **Qdrant Cloud** (forever-free 1GB cluster) | Lives outside the Space → persists across restarts for free. Local-dev fallback: Chroma with a persist dir. |
| LLM (generation) | **Groq** → `llama-3.3-70b-versatile` (free, no card, OpenAI-compatible, very fast) | Alt: Google Gemini free tier (1M-token context) for long context. Wire via env so it's swappable. |
| Secrets | HF Space **Repository Secrets** | `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`. Never commit keys. |

**Do not** introduce paid services, and do not use a hosted embedding API (local embeddings keep it free and rate-limit-free).

---

## 4. Architecture

Two loops sharing one vector store.

**Ingestion loop** (triggered by upload):
```
file upload → detect type → parse (PyMuPDF / python-pptx)
→ attach metadata {source, page/slide, subject} → chunk (overlap)
→ embed (bge-base, local) → upsert into Qdrant collection
```

**Query loop** (triggered by a question):
```
question (+ optional subject filter) → embed query
→ Qdrant similarity search (top-k, metadata filter) → build grounded prompt
→ Groq LLM → answer + rendered source citations
```

**Persistence model:** Raw files are transient; the *vector store is the source of truth* and lives in Qdrant Cloud outside the Space. Optionally mirror raw uploads to a private HF Dataset repo for re-indexing, but this is not required for v1.

---

## 5. Repository structure

```
legal-study-rag/
├── app.py                  # Streamlit entry point (UI + routing)
├── README.md               # HF Space config front-matter (sdk: streamlit) + docs
├── requirements.txt
├── .gitignore              # excludes .env, __pycache__, local chroma dir
├── .env.example            # documents required vars (no real values)
├── src/
│   ├── __init__.py
│   ├── config.py           # loads env vars, model names, constants
│   ├── ingestion.py        # parse PDF/PPTX → chunks with metadata
│   ├── embeddings.py       # loads bge-base once, cached
│   ├── vector_store.py     # Qdrant client: create collection, upsert, search
│   ├── rag.py              # retrieval + prompt assembly + Groq call
│   └── prompts.py          # grounded system prompt + disclaimers
└── tests/
    └── test_ingestion.py   # chunking + metadata unit tests
```

---

## 6. Phased work plan

Each phase lists **Objective → Tasks → Acceptance criteria**. Build, run, and verify before advancing.

### Phase 0 — Project setup
**Objective:** A runnable empty Streamlit app and clean project skeleton.
**Tasks:**
- [ ] Create the repo structure above; init git.
- [ ] `requirements.txt` with pinned versions: `streamlit`, `pymupdf`, `python-pptx`, `langchain`, `langchain-text-splitters`, `sentence-transformers`, `qdrant-client`, `groq`, `python-dotenv`.
- [ ] `config.py` reads env vars via `python-dotenv`; `.env.example` documents them.
- [ ] Minimal `app.py` that renders a title and the disclaimer banner.
**Acceptance:** `streamlit run app.py` serves a page locally with the title and disclaimer.

### Phase 1 — Document ingestion
**Objective:** Turn an uploaded file into clean, metadata-rich chunks.
**Tasks:**
- [ ] `ingestion.py`: `parse_pdf(path)` → list of `{text, page}` per page via PyMuPDF.
- [ ] `parse_pptx(path)` → list of `{text, slide}` per slide (include slide notes) via python-pptx.
- [ ] Chunk each page/slide with `RecursiveCharacterTextSplitter` (start: `chunk_size=1000`, `chunk_overlap=200`).
- [ ] Attach metadata to every chunk: `source` (filename), `doc_type`, `page` or `slide`, `subject`, `chunk_index`.
- [ ] Unit tests in `tests/test_ingestion.py` for chunk counts and metadata presence.
**Acceptance:** Feeding a sample PDF and a sample PPTX returns chunks each carrying correct source + page/slide metadata; tests pass.

### Phase 2 — Embeddings + vector store
**Objective:** Persist chunks as searchable vectors in Qdrant.
**Tasks:**
- [ ] `embeddings.py`: load `bge-base-en-v1.5` once (module-level / `@st.cache_resource`); helper to embed a list of texts and a single query (with bge query prefix).
- [ ] `vector_store.py`: connect to Qdrant Cloud; `ensure_collection()` with the correct vector size (768 for bge-base) and cosine distance; `upsert_chunks(chunks)`; `search(query_vector, top_k, subject_filter)`.
- [ ] Provide a local Chroma fallback path guarded by an env flag for offline dev.
**Acceptance:** Ingested chunks appear in the Qdrant collection; a manual vector search returns relevant chunks with metadata intact.

### Phase 3 — Retrieval + grounded generation
**Objective:** Answer questions strictly from retrieved context, with citations.
**Tasks:**
- [ ] `prompts.py`: a strict system prompt — answer ONLY from provided context; if insufficient, say so plainly; never invent case names/citations; always reference the source. Include the "study aid, not legal advice" framing.
- [ ] `rag.py`: embed question → `search()` (honor subject filter) → assemble context block that includes each chunk's source/page/slide → call Groq `llama-3.3-70b-versatile` → return answer + the list of source chunks used.
- [ ] Handle the empty-retrieval case gracefully (return the "not in your materials" message, don't call the LLM blind).
**Acceptance:** Asking a question answerable from an ingested doc returns a correct answer plus accurate source references; asking something absent returns the honest "not found" response.

### Phase 4 — Streamlit UI
**Objective:** The full user experience.
**Tasks:**
- [ ] Sidebar: `st.file_uploader` (accept `.pdf`, `.pptx`, multiple); a subject selector/text input applied as metadata on upload; an "Index files" button that runs ingestion → embedding → upsert with a progress indicator and success/error toasts.
- [ ] Knowledge-base panel: list indexed sources (query Qdrant for distinct `source` values); show counts.
- [ ] Main area: chat interface (`st.chat_input` / `st.chat_message`), optional subject filter for queries, streamed answer.
- [ ] Render citations under each answer as an expander: file name + page/slide.
- [ ] Persistent disclaimer banner at top.
**Acceptance:** A user can upload → index → see the file listed → ask a question → read a cited answer, entirely through the UI.

### Phase 5 — Law-student study features (build if core is solid)
**Objective:** Make it a study tool, not just Q&A.
**Tasks:**
- [ ] "Summarize this case/topic" action over a selected source.
- [ ] "Generate practice questions" from a subject.
- [ ] "Define this term" using retrieved context.
**Acceptance:** Each feature produces grounded, source-cited output. Skip to Phase 6 if time-constrained — these are enhancements.

### Phase 6 — Deploy to Hugging Face Spaces
**Objective:** Live, free, public, persistent.
**Tasks:**
- [ ] `README.md` front-matter: `title`, `emoji`, `sdk: streamlit`, `app_file: app.py`, `pinned: false`.
- [ ] Create the Space (New Space → **Streamlit** SDK → **CPU Basic (free)**). If free CPU isn't offered, see §8 fallbacks.
- [ ] Add `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` as **Repository Secrets** (Space Settings).
- [ ] Push via git; watch build logs; confirm the Space boots (first boot downloads the embedding model — expect a slow cold start).
**Acceptance:** Public `*.hf.space` URL works; upload → ask → cited answer works end-to-end; after the Space sleeps and is re-woken, previously indexed documents are still queryable (proves external persistence).

### Phase 7 — Harden & document
**Objective:** Robust and handoff-ready.
**Tasks:**
- [ ] Input validation: file-type/size guards, friendly errors for parse failures.
- [ ] Deduplicate re-uploads (skip or replace by `source` name).
- [ ] Final `README.md`: what it is, setup, env vars, deploy steps, limitations.
- [ ] Confirm no secrets in git history.
**Acceptance:** App handles bad input without crashing; README lets a new developer run and deploy it unaided.

---

## 7. Environment variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq LLM access (get free at console.groq.com — no card). |
| `QDRANT_URL` | Qdrant Cloud cluster endpoint. |
| `QDRANT_API_KEY` | Qdrant Cloud auth. |
| `LLM_MODEL` | Default `llama-3.3-70b-versatile`; swappable. |
| `EMBED_MODEL` | Default `BAAI/bge-base-en-v1.5`. |
| `USE_LOCAL_CHROMA` | `true` for offline dev with Chroma instead of Qdrant. |

Document all of these in `.env.example`. Set real values locally in `.env` (git-ignored) and as Repository Secrets on HF.

---

## 8. Known constraints & fallbacks

- **Free-tier flux (2026):** HF restricted free hosting for new Gradio/Docker Spaces; native **Streamlit** Spaces on free CPU still appear available. Using `sdk: streamlit` (not Docker) is the safest route. If free CPU is unavailable at creation, fall back to **Streamlit Community Cloud** (works, but ~1GB RAM — may force the lighter `all-MiniLM-L6-v2` embedding model) or **Render** free web service.
- **Cold starts:** Free Spaces sleep after idle; first wake re-downloads the embedding model (~30–90s). Acceptable for a study tool.
- **LLM rate limits:** Groq free tier is roughly 30 requests/minute and ~1,000/day — fine for one student; not for a public audience.
- **Persistence:** Do NOT rely on the Space's local disk for the knowledge base — it's wiped on rebuild. Qdrant Cloud (external) is the free persistence mechanism.
- **Embedding/vector-size match:** If the embedding model is swapped, the Qdrant collection's vector size must match (bge-base = 768, MiniLM = 384). Recreate the collection on model change.

---

## 9. First command for the builder

Start with **Phase 0 only**: scaffold the repo, pin `requirements.txt`, and get an empty Streamlit app with the disclaimer running locally. Report back before starting Phase 1.
