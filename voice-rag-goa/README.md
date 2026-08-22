# 🌴 Voice RAG — Goa Hacker House

**Speak a question. Get a grounded, cited answer in well under a second — with every stage of the pipeline timed live in front of you.**

An end-to-end, ultra-low-latency **voice → retrieval-augmented-generation** system built on the
[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) dataset. Real-time
speech-to-text feeds a hybrid (dense + BM25) retriever over three chunking strategies; a
Pydantic-strict LLM harness streams a grounded answer; and a full guardrail stack refuses anything
the corpus can't support. The whole thing runs **with zero API keys** (type instead of speak, and a
deterministic extractive answerer stands in for the LLM) — so every latency number you see is real
on any machine.

<p align="center"><em>Deep palm green · sunny Goa gold · hot pink neon — the Hacker House on the beach.</em></p>

---

## Table of contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Repository layout](#repository-layout)
4. [Quickstart](#quickstart)
5. [Zero-key offline mode](#zero-key-offline-mode)
6. [Configuration](#configuration)
7. [API reference](#api-reference)
8. [Benchmarking & measured results](#benchmarking--measured-results)
9. [How the pipeline works](#how-the-pipeline-works)
10. [Guardrails](#guardrails)
11. [Model & version notes (read this)](#model--version-notes-read-this)
12. [Troubleshooting](#troubleshooting)
13. [Tech stack](#tech-stack)

---

## What it does

| # | Capability | Where |
|---|------------|-------|
| 1 | **Real-time voice transcription** — Sarvam AI (Indic-first), ElevenLabs Scribe, or Groq Whisper, tried in a configurable fallback order. Mic audio is captured as PCM16 @ 16 kHz and streamed over a WebSocket. | `app/stt.py`, `src/lib/audio.ts` |
| 2 | **Multi-strategy chunking + hybrid vector search** — Semantic (cosine-breakpoint), Hierarchical (small children / parent context), and Sliding-window chunking, each retrievable via dense cosine, BM25, or an RRF-fused hybrid. | `app/ingest.py`, `app/retrieval.py` |
| 3 | **Sub-200 ms retrieval-to-generation with streaming** — query → first token is the number the whole system is tuned around; the answer streams token-by-token over the socket. | `app/main.py` (`run_turn`) |
| 4 | **P50 / P70 / P100 latency analytics & benchmarking** — a real benchmark harness scores latency *and* retrieval quality (Recall@k, MRR, nDCG) against the dataset's gold labels. | `app/benchmark.py`, `app/metrics.py` |
| 5 | **Structured Pydantic LLM orchestration** — strict JSON I/O, retries with backoff, a circuit breaker, and provider failover (Groq → Cerebras → offline extractive). | `app/harness.py`, `app/schemas.py` |
| 6 | **Guardrails** — off-topic filtering, prompt-injection blocking, context-sufficiency gating, and post-generation grounding verification with a fixed refusal string. | `app/guardrails.py` |
| 7 | **Full-stack Goa-aesthetic UI** — live mic visualiser, per-stage pipeline timers, P50/P70/P100 gauges, a strategy switcher, a citation inspector, and a Grounded/Blocked badge. | `frontend/src/**` |

---

## Architecture

```
                        ┌──────────────────────────── Browser (Next.js) ───────────────────────────┐
   🎙  mic  ──PCM16──▶   │  MicRecorder → WebSocket        VoiceRecorder · GroundingBadge            │
                        │       │                          ChunkVisualizer · LatencyDashboard        │
                        └───────┼────────────────────────────────────────────────────────▲─────────┘
                                │  ws://…/ws/audio  (binary PCM up · JSON events down)      │ live events
                                ▼                                                           │
   ┌──────────────────────────────────────── FastAPI backend ───────────────────────────────────────┐
   │                                                                                                  │
   │   STT ──▶ embed query ──▶ hybrid retrieve ──▶ guardrail(context) ──▶ LLM harness ──▶ grounding   │
   │  stt.py    embeddings.py     retrieval.py       guardrails.py         harness.py     guardrails  │
   │              (fp32 ONNX)   dense ⊕ BM25 (RRF)                       Groq→Cerebras→offline         │
   │                                   │                                                              │
   │                       ┌───────────┴───────────┐                                                  │
   │                    memory (NumPy+rank_bm25)  lancedb (ANN + FTS)   ◀── built by ingest.py         │
   └──────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    ▲
                                         ai4bharat/MSMARCO-XI (parquet over HfFileSystem)
```

Every arrow in the top row is a **timed stage**. The backend emits a `stage` WebSocket event at the
start and end of each, so the Live Pipeline panel is a faithful trace of one real request — not a
mock-up.

---

## Repository layout

```
voice-rag-goa/
├── README.md                     ← you are here
├── backend/
│   ├── requirements.txt          ← pinned, all cp314 wheels, no compiler needed
│   ├── .env.example              ← every key optional; copy to .env
│   ├── data/                     ← index artifacts (auto-built on first run)
│   └── app/
│       ├── main.py               ← FastAPI app: REST + /ws/audio + run_turn() pipeline
│       ├── config.py             ← all settings, env parsing, language shards
│       ├── ingest.py             ← dataset → 3 chunking strategies → memory + lancedb
│       ├── embeddings.py         ← fp32 ONNX embedder (BAAI/bge-small-en-v1.5)
│       ├── retrieval.py          ← dense / sparse / hybrid (RRF) over both backends
│       ├── stt.py                ← Sarvam / ElevenLabs / Groq Whisper with fallback
│       ├── harness.py            ← Pydantic-strict LLM orchestration (retry/breaker/failover)
│       ├── guardrails.py         ← input / context / grounding guardrails
│       ├── benchmark.py          ← latency + retrieval-quality benchmark harness
│       ├── metrics.py            ← percentile/summary math
│       ├── schemas.py            ← Pydantic models = the wire contract
│       └── textutil.py           ← tokenisation / sentence splitting helpers
└── frontend/
    ├── package.json              ← Next 16 · React 19 · Tailwind v4
    ├── .env.local.example        ← NEXT_PUBLIC_API_BASE (default http://localhost:8000)
    └── src/
        ├── app/                  ← layout.tsx · page.tsx · globals.css (Goa @theme)
        ├── components/           ← VoiceRecorder · GroundingBadge · ChunkVisualizer · LatencyDashboard
        └── lib/                  ← types.ts (mirrors schemas.py) · api.ts · audio.ts · useVoicePipeline.ts · wav.ts
```

---

## Quickstart

> **Prerequisites:** Python **3.11+** (built and measured on 3.14) and Node **20.9+**.
> No GPU, no compiler, no Docker required.

### 1 · Backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# Optional — copy the env template and add any keys you have. It runs fine without.
cp .env.example .env          # Windows: copy .env.example .env

# Start the API. On first run it builds the index from the dataset
# (~2 min for the default 1200 rows) unless data/ is already populated.
python -m app.main
# equivalently: uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The server is up at **http://localhost:8000** — open **http://localhost:8000/docs** for interactive
API docs. Health lives at `GET /api/health` and reports `initialising` while the index builds,
then `ok` once it is ready — **including key-less offline mode, which is fully functional and still
reports `ok`**. It only returns `degraded` if the index failed to build.

**Build the index explicitly** (optional — startup does it for you):

```bash
python -m app.ingest                        # build all three strategies (default: Hindi shard)
python -m app.ingest --languages hin,tam    # pick Indic language shards
```

> Valid shard keys: `asm ben guj hin kan mal mar nep ori pan san tam tel urd`. English passages ride
> along *inside* every Indic shard, so there is no separate `eng` shard — asking for one raises
> `ValueError: Unknown language 'eng'`.

### 2 · Frontend

In a **second terminal**:

```bash
cd frontend
npm install

# Optional — only needed if the backend is not on http://localhost:8000
cp .env.local.example .env.local           # Windows: copy .env.local.example .env.local

npm run dev
```

Open **http://localhost:3000**. Tap the mic and ask a question — or type one and hit Enter. Try the
example chips (one is a prompt-injection, one is off-topic) to watch the guardrails fire.

> **Production build:** `npm run build && npm start`. The build is verified green in this repo
> (TypeScript strict, 3 static routes).

---

## Zero-key offline mode

**The system is fully functional with no API keys whatsoever.** This is deliberate — a demo shouldn't
depend on someone's quota.

| With no keys… | You get | You lose |
|---|---|---|
| **Retrieval** | 100% local: fp32 ONNX embeddings + NumPy dense search + `rank_bm25`. Every retrieval/latency number is real. | Nothing. |
| **Generation** | A deterministic **extractive answerer** that stitches together the top grounded sentences. | Fluent LLM phrasing. |
| **Voice input** | Type your question in the UI instead. | Spoken input (STT needs a key). |

Add a single **free Groq key** (`GROQ_API_KEY`) and you unlock both streaming LLM generation *and*
Whisper STT at once. See [Configuration](#configuration).

---

## Configuration

All configuration is environment variables, documented inline in
[`backend/.env.example`](backend/.env.example). **Every key is optional.** The headline knobs:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(empty)* | Unlocks LLM generation **and** Whisper STT. [Free key](https://console.groq.com/keys). |
| `GROQ_MODEL` / `GROQ_FALLBACK_MODEL` | `openai/gpt-oss-20b` / `…-120b` | Generation models (see [version notes](#model--version-notes-read-this)). |
| `SARVAM_API_KEY` | *(empty)* | Indic-first STT; best for Hindi and other Indian languages. |
| `CEREBRAS_API_KEY` | *(empty)* | Optional second LLM provider; harness fails over to it. |
| `INGEST_LANGUAGES` | `hin` | Comma-separated shards: `asm ben brx guj hin kan mal mar ori pan tam tel urd eng`. |
| `INGEST_ROW_LIMIT` | `1200` | Source rows to ingest (~10k passages, ~15–30k chunks/strategy). Controls build time. |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | 384-dim embedder. Switch to `intfloat/multilingual-e5-small` for Indic text. |
| `RETRIEVAL_BACKEND` | `both` | `memory` (lowest tail), `lancedb` (scales past RAM), or `both` (head-to-head). |
| `DEFAULT_STRATEGY` | `hierarchical` | Default chunking strategy. |
| `FUSION_METHOD` | `rrf` | `rrf` (rank-based) or `weighted` (score-based, uses `HYBRID_ALPHA`). |
| `GROUNDING_THRESHOLD` | `0.65` | Below this mean sentence-grounding, the answer is refused. |
| `CONTEXT_SUFFICIENCY_THRESHOLD` | `0.32` | Below this best-retrieval score, the query is deemed out-of-corpus. |
| `CORS_ORIGINS` | `localhost:3000,…` | Allowed browser origins. |

Frontend: [`frontend/.env.local.example`](frontend/.env.local.example) has a single variable,
`NEXT_PUBLIC_API_BASE` (default `http://localhost:8000`). The WebSocket URL is derived from it, so
`http`→`ws` and `https`→`wss` stay in sync automatically.

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/health` | Readiness, indexed strategies, models, offline-mode flag, warnings. |
| `POST` | `/api/query` | One-shot text query → grounded answer + chunks + timings (non-streaming). |
| `POST` | `/api/query/stream` | Same, as Server-Sent Events (token streaming). |
| `POST` | `/api/transcribe` | `multipart/form-data` audio → transcript (one-shot STT). |
| `GET`  | `/api/analytics` | Live rolling percentiles + last benchmark report + chunk stats. |
| `POST` | `/api/benchmark` | Kick off a benchmark run in the background (409 if already running). |
| `WS`   | `/ws/audio` | The live path: stream PCM up, receive `ready`/`stage`/`transcript`/`chunks`/`guardrail`/`token`/`done`/`error` events down. |

**WebSocket protocol.** The client sends binary PCM16 frames followed by a JSON control frame
(`{type:"end", strategy, mode, top_k, sample_rate, …}`), or a `{type:"text", text, …}` frame to skip
STT. The server replies with a stream of typed JSON events; the discriminated union is mirrored
exactly in [`frontend/src/lib/types.ts`](frontend/src/lib/types.ts). If the socket is unavailable the
UI transparently falls back to the REST endpoints (no streaming, but everything else works).

---

## Benchmarking & measured results

```bash
cd backend
python -m app.benchmark --queries 60                 # latency + quality grid
python -m app.benchmark --queries 120 --no-llm       # retrieval-only, larger sample
python -m app.benchmark --strategies hierarchical --modes dense,hybrid
```

The harness pulls gold-labelled queries from the dataset (`passages.is_selected` → relevant
passages), runs the full strategy × mode grid, and writes `data/benchmark_report.json`, which the UI
reads via `/api/analytics`. You can also trigger a run from the **Run benchmark** button in the
Latency panel.

### Results shipped in this repo

*60 gold-labelled Hindi-shard queries · `memory` backend · offline mode (no LLM key) · concurrency 4,
measured on an AMD Zen CPU. Reproduce with `python -m app.benchmark --queries 60`.*

**Latency (ms), overall:**

| Stage | P50 | P70 | P95 | P100 | Budget |
|---|---:|---:|---:|---:|---:|
| Embed query | 11.4 | 13.2 | 17.3 | 26.2 | 12 |
| Hybrid retrieval | 5.9 | 11.2 | 23.5 | 34.4 | 10 |
| Guardrails | 0.3 | 94.0 | 154.2 | 209.5 | 8 |
| **Query → first token** | **29.0** | **100.5** | **162.5** | **212.8** | **200** |

> **Reading these honestly:** this report was generated **offline (no LLM key)**, so the extractive
> answerer returns instantly and *query → first token = end-to-end*. P50 of **29 ms** is ~7× under the
> 200 ms budget. The P70+ tail is driven entirely by **sparse (BM25) mode**: with no dense score to
> reuse, the context guardrail re-embeds the top chunks to judge sufficiency (~150 ms). That cost is
> real and reported rather than hidden — and sparse is not the serving default. In dense/hybrid mode
> the guardrail is sub-millisecond. Adding a Groq key moves first-token latency into the generation
> stage, which is measured separately and folded into `total_e2e_ms`.

**Retrieval quality (higher is better):**

| Strategy | Mode | Recall@1 | Recall@5 | MRR@10 | nDCG@5 |
|---|---|---:|---:|---:|---:|
| Hierarchical | dense | 0.417 | 0.883 | 0.630 | 0.683 |
| Sliding | dense | 0.392 | **0.892** | 0.611 | 0.669 |
| Semantic | dense | 0.417 | 0.867 | 0.621 | 0.671 |
| Hierarchical | hybrid | 0.342 | 0.800 | 0.554 | 0.603 |
| Hierarchical | sparse | 0.217 | 0.717 | 0.418 | 0.480 |

Dense retrieval wins on this dataset because the MSMARCO-XI queries are natural-language questions
(embeddings capture paraphrase; BM25 rewards literal term overlap). Hybrid RRF trades a little
top-1 precision for robustness on rare terms. Sparse is the honest floor.

---

## How the pipeline works

**Chunking strategies** (`app/ingest.py`) — all three are pre-built and independently retrievable:

- **Semantic** — splits at cosine-distance breakpoints between adjacent sentences, so each chunk is a
  coherent topic.
- **Hierarchical** — indexes small 128-token *children* for precise matching but returns the 512-token
  *parent* as context, so the model reads a full passage around the hit.
- **Sliding** — fixed 256-token windows with 25% overlap; the window's positional metadata is folded
  into the embedded text so retrieval is boundary-aware.

**Retrieval modes** (`app/retrieval.py`):

- **dense** — exact cosine over fp32 embeddings (NumPy matmul on the `memory` backend, ANN on
  `lancedb`).
- **sparse** — BM25 (`rank_bm25` in-memory, or LanceDB native FTS).
- **hybrid** — the two fused via **Reciprocal Rank Fusion** (`k=60`), or a min-max weighted blend
  (`HYBRID_ALPHA`).

**Latency accounting.** `query_to_first_token_ms` measures *transcript-in-hand → first generated
token* — this is the sub-200 ms target. STT is measured separately (`stt_ms`) and folded into
`total_e2e_ms`, because network STT latency is provider-bound and shouldn't mask the retrieval-to-
generation performance the system actually controls.

---

## Guardrails

Three checkpoints (`app/guardrails.py`), each surfaced live in the UI's Grounded/Blocked badge:

1. **Input** — rejects empty/oversized queries, **prompt-injection** attempts, unsafe content, and
   off-topic questions *before* any work is done.
2. **Context** — after retrieval, if the best score is below `CONTEXT_SUFFICIENCY_THRESHOLD` the query
   is judged out-of-corpus and refused without calling the LLM.
3. **Grounding** — after generation, each answer sentence is checked against the retrieved context; if
   mean grounding falls below `GROUNDING_THRESHOLD`, the answer is thrown out.

When any check refuses, the system returns the **exact** string:

> `I cannot answer this based on the verified dataset.`

This is a hard contract — the extractive fallback and every LLM path use the identical refusal text, so
"the model couldn't ground it" and "the model made something up we caught" are indistinguishable to a
downstream consumer, which is the safe default.

---

## Model & version notes (read this)

A few things diverge from what older tutorials / specs will tell you — each was hit and fixed during
the build:

- **Groq model IDs.** `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` were **decommissioned on
  2026-08-16** and now 404. The working defaults are **`openai/gpt-oss-20b`** (primary) and
  **`openai/gpt-oss-120b`** (fallback).
- **Sarvam model.** The current speech model is **`saaras:v3`**. `saaras:v1` and `sarvam-2.5` from older
  docs are retired and return 404.
- **Embeddings run on fp32 ONNX, not fastembed.** `fastembed` ships an int8 graph that measured **~13×
  slower** than the official fp32 export on this CPU (AMD Zen). We drive `onnxruntime` directly — see the
  measurement note in `app/embeddings.py`. Result: embed P50 of **~11 ms**.
- **`datasets` server is bypassed.** The hub's datasets-server viewer API returns HTTP 500 on this repo
  (a nested-list Arrow conversion bug), so `ingest.py` reads the parquet shards directly over
  `HfFileSystem`. The `datasets` package is listed as an *optional* convenience dependency but the
  ingest path never imports it — you can drop it from `requirements.txt` if you don't need it for
  ad-hoc dataset exploration.
- **ESLint / `@types/react-dom` pins.** `next build` on Next 16 no longer runs a lint step (the `eslint`
  config key was removed); ESLint is a dev convenience only.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Health stuck on `initialising` | First-run index build (~2 min for 1200 rows). Watch the server log; `data/` populates as it goes. |
| Health is `degraded` | You're key-less → offline mode. Expected. Add `GROQ_API_KEY` for LLM + voice. |
| Mic button does nothing | Browsers require **HTTPS or localhost** for `getUserMedia`. On localhost it just works; over a LAN IP, use the text box or serve over HTTPS. |
| "Reconnecting to server…" in the UI | Backend not up yet, or `NEXT_PUBLIC_API_BASE` points elsewhere. The UI auto-reconnects and falls back to REST. |
| `404 model_not_found` from Groq | You overrode `GROQ_MODEL` with a decommissioned ID — see the note above. |
| Want it faster to build | Lower `INGEST_ROW_LIMIT`, or set `RETRIEVAL_BACKEND=memory` to skip building the LanceDB index. |

---

## Tech stack

**Backend** — Python 3.14 · FastAPI 0.135 · Uvicorn · Pydantic v2 · onnxruntime 1.29 (fp32
BGE-small) · LanceDB 0.37 + Tantivy FTS · rank-bm25 · NumPy 2.4 · httpx · orjson.

**Frontend** — Next.js 16 (App Router, Turbopack) · React 19 · TypeScript (strict) · Tailwind CSS v4
(CSS-first `@theme`) · lucide-react · Web Audio (AudioWorklet + ScriptProcessor fallback) · native
WebSocket.

**Data** — [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI), read as
parquet over `HfFileSystem`; gold relevance from `passages.is_selected`.

---

<p align="center">Built for the Goa Hacker House. 🌊🥥</p>
