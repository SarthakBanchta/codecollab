# 🛡️ TrustRAG

**A Retrieval-Augmented Generation pipeline with built-in AI output validation and guardrails.**

TrustRAG doesn't just answer questions from your documents — it grades its own answers before showing them to you, catching hallucinations, unsupported claims, and low-confidence retrievals. If it can't ground an answer in the retrieved context, it refuses to answer rather than guess.

---

## Why This Exists

Standard RAG systems have a critical blind spot: **they can hallucinate confidently even when retrieval works correctly.** The LLM might:

- Invent facts that aren't in the retrieved chunks
- Mix its training data with retrieved context seamlessly
- Fabricate citations that look real but reference nothing
- Answer questions the retrieved context doesn't actually cover

TrustRAG solves this with a **validation layer** — a second LLM call that acts as a "judge," evaluating every claim in the generated answer against the actual source chunks before anything is shown to the user.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────┐
│   RETRIEVER      │  ← ChromaDB top-k similarity search
│   + Confidence   │  ← Flags queries below similarity threshold
│     Gate         │
└────────┬─────────┘
         │ chunks + scores
         ▼
┌──────────────────┐
│   GENERATOR      │  ← LLM with grounded RAG prompt
│   + Citation     │  ← Forces [Source N] citation format
│     Prompting    │
└────────┬─────────┘
         │ cited answer
         ▼
┌──────────────────┐
│   VALIDATOR      │  ← Second LLM call (the "judge")
│                  │
│ Stage 1: Ground- │  ← Breaks answer into claims,
│   edness Check   │    checks each against sources
│                  │
│ Stage 2: Citation│  ← Verifies [Source N] references
│   Cross-Check    │    actually exist
│                  │
│ Stage 3: Decision│  ← score ≥ 80 → Show
│   Policy         │    score 50-79 → Warn
│                  │    score < 50 → Refuse
└────────┬─────────┘
         │ verdict
         ▼
┌──────────────────┐
│   LOGGER         │  ← SQLite: every query logged with
│   (Observability)│    full telemetry + latency breakdown
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   STREAMLIT UI   │  ← Chat interface + confidence badges
│   + Dashboard    │    + validation details + monitoring
└──────────────────┘
```

---

## Quick Start

### 1. Clone and install

```bash
cd trustrag
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure API key

```bash
copy .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-your-key-here
```

### 3. Ingest documents

Place your `.txt`, `.md`, or `.pdf` files in the `data/` directory (sample files are included), then run:

```bash
python ingest.py
# Use --reset to clear and re-ingest:
# python ingest.py --reset
```

### 4. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Project Structure

```
trustrag/
├── data/                     # Source documents go here
│   ├── sample_ai_safety.txt  # Sample doc (included)
│   └── sample_rag_explained.md
├── config.py                 # Centralized config from .env
├── ingest.py                 # Chunking + embedding + ChromaDB
├── retriever.py              # Top-k retrieval + confidence gate
├── generator.py              # RAG prompt + LLM generation
├── validator.py              # 3-stage groundedness validation
├── logger.py                 # SQLite observability logging
├── app.py                    # Streamlit UI + dashboard
├── requirements.txt
├── .env.example
└── README.md
```

---

## Configuration Reference

All settings are in `.env` (see `.env.example` for the full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `"openai"` or `"anthropic"` |
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key |
| `GENERATION_MODEL` | `gpt-4o-mini` | Model for answer generation |
| `JUDGE_MODEL` | `gpt-4o-mini` | Model for validation (keep cheap) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `TOP_K` | `4` | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.5` | Confidence gate threshold (0-1) |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `SCORE_SHOW_THRESHOLD` | `80` | Min score to show answer |
| `SCORE_WARN_THRESHOLD` | `50` | Min score before refusing |

---

## Swapping LLM Providers

### Switch to Anthropic Claude

Edit `.env`:
```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
GENERATION_MODEL=claude-3-5-haiku-20241022
JUDGE_MODEL=claude-3-5-haiku-20241022
```

> **Note:** Embeddings still use OpenAI (`text-embedding-3-small`) regardless of the LLM provider, since ChromaDB stores are tied to the embedding model used during ingestion. To switch embedding models, you must re-ingest with `python ingest.py --reset`.

---

## How the Validation Layer Works

The validator (`validator.py`) runs three stages on every generated answer:

1. **Groundedness Check**: A second LLM call acts as a "judge." It receives the generated answer and the source chunks, breaks the answer into individual claims, and classifies each as `supported`, `partial`, or `unsupported`. Returns a 0-100 groundedness score.

2. **Citation Cross-Check**: Regex extracts all `[Source N]` citations from the answer and verifies each one references a real chunk from retrieval. Catches fabricated citations.

3. **Decision Policy**: Combines the groundedness score, citation integrity, and retrieval confidence into a final decision:
   - **Show** (score ≥ 80): Answer displayed normally with green badge
   - **Warn** (score 50-79): Answer shown with yellow "unverified claims" banner
   - **Refuse** (score < 50): Answer suppressed, user sees refusal message

---

## Observability

Every query is logged to SQLite (`trustrag_logs.db`) with:
- Timestamp, query text, generated answer
- Retrieved chunk IDs and similarity scores
- Groundedness score and per-claim verdicts
- Fabricated citations (if any)
- Latency breakdown: retrieval / generation / validation (separately)
- Final decision: show / warn / refuse

The Streamlit sidebar has a **Dashboard** tab showing:
- Total queries, average groundedness, average latency
- Decision breakdown (shown / warned / refused)
- Table of flagged and refused queries

---

## License

MIT
