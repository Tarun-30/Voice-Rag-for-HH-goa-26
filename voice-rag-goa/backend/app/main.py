"""
FastAPI application: the pipeline orchestrator and all HTTP/WebSocket surfaces.

Endpoints
---------
    GET  /api/health          liveness + which capabilities are configured
    POST /api/query           run one turn, non-streaming (used by the benchmark)
    POST /api/query/stream     same, as an SSE token stream
    POST /api/transcribe      audio file -> transcript (STT only)
    GET  /api/analytics       offline benchmark report + live rolling percentiles
    POST /api/benchmark       kick off a benchmark run in the background
    WS   /ws/audio            the real-time voice path: audio/text in, events out

The single source of truth for a turn is `run_turn()`. Both the REST and the
WebSocket paths call it; the only difference is the `emit` callback, which the
WebSocket uses to forward staged events (stage timers, chunks, tokens) and REST
ignores. That guarantees the streamed demo and the benchmarked numbers exercise
identical code.

Latency accounting: `run_turn` starts its clock with the *transcript in hand*,
so `query_to_first_token_ms` - the number the 200 ms target refers to - never
includes STT (a third-party network call we don't control). STT is measured and
reported separately, and folded into `total_e2e_ms`.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

from fastapi import Body, FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__, config
from . import guardrails, ingest, stt
from .harness import GenerationResult, get_harness
from .metrics import summarize
from .retrieval import get_retriever
from .schemas import (
    AnalyticsResponse,
    BenchmarkRequest,
    GuardrailVerdict,
    HealthResponse,
    PipelineTimings,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    TranscriptionResponse,
    WSChunks,
    WSClientConfig,
    WSDone,
    WSError,
    WSGuardrail,
    WSReady,
    WSStage,
    WSToken,
    WSTranscript,
)

logger = logging.getLogger(__name__)

EmitFn = Callable[[object], Awaitable[None]]


# --------------------------------------------------------------------------- #
# Runtime state
# --------------------------------------------------------------------------- #


class AppState:
    def __init__(self) -> None:
        self.ready = False
        self.indexing = False
        self.error: str | None = None
        self.warnings: list[str] = []
        self.live_timings: deque[PipelineTimings] = deque(maxlen=500)
        self.benchmark_running = False

    def record(self, timings: PipelineTimings) -> None:
        self.live_timings.append(timings)


STATE = AppState()


async def _startup_index() -> None:
    """Ensure indexes exist, building missing ones off the event loop."""
    STATE.indexing = True
    loop = asyncio.get_running_loop()
    try:
        missing = [s for s in config.ALL_STRATEGIES if s not in ingest.indexed_strategies()]
        if missing:
            if not config.AUTO_INDEX_ON_STARTUP:
                STATE.warnings.append(
                    f"Strategies not indexed: {missing}. Run `python -m app.ingest`."
                )
            else:
                logger.info("auto-indexing missing strategies: %s", missing)
                await loop.run_in_executor(None, ingest.build_all)
        # Warm the retriever + embedder so the first real request is not cold.
        retriever = await loop.run_in_executor(None, lambda: get_retriever(reload=True))
        await loop.run_in_executor(None, retriever.embedder.warmup)
        if not retriever.ready:
            STATE.error = "No indexes available and indexing did not produce any."
        else:
            STATE.ready = True
    except Exception as exc:  # pragma: no cover - surfaced via /api/health
        logger.exception("startup indexing failed")
        STATE.error = f"{type(exc).__name__}: {exc}"
    finally:
        STATE.indexing = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config.ensure_dirs()
    if not config.llm_configured():
        STATE.warnings.append("No LLM configured - using offline extractive answerer.")
    if not config.stt_configured():
        STATE.warnings.append("No STT configured - type queries instead of speaking.")
    # Index in the background so uvicorn binds the port immediately.
    task = asyncio.create_task(_startup_index())
    try:
        yield
    finally:
        task.cancel()
        await stt.aclose()
        await get_harness().aclose()


app = FastAPI(title="Voice RAG - Goa Hacker House", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #


async def run_turn(
    query: str,
    *,
    strategy: str | None = None,
    mode: str = "hybrid",
    top_k: int | None = None,
    include_thought_process: bool = False,
    request_id: str = "",
    transcript_prefix_ms: float = 0.0,
    emit: EmitFn | None = None,
) -> QueryResponse:
    """Run one full turn. Emits staged events if `emit` is provided.

    `transcript_prefix_ms` is the STT time already spent before this call; it is
    added to total_e2e but never to the query-to-first-token figure.
    """
    strategy = strategy or config.DEFAULT_STRATEGY
    top_k = top_k or config.RETRIEVE_TOP_K
    retriever = get_retriever()
    harness = get_harness()
    request_id = request_id or uuid.uuid4().hex[:12]
    timings = PipelineTimings(provider=harness.primary_provider, model=harness.primary_model)

    turn_start = time.perf_counter()

    async def stage(name, status, ms=0.0, budget=0.0, detail=""):
        if emit:
            await emit(WSStage(request_id=request_id, t_ms=_since(turn_start),
                               stage=name, status=status, ms=round(ms, 2),
                               budget_ms=budget, detail=detail))

    # --- 1. Input guardrail ------------------------------------------------ #
    g0 = time.perf_counter()
    await stage("guardrail", "start", budget=config.BUDGET.guardrail_ms)
    input_verdict = guardrails.check_input(query)
    if emit:
        await emit(WSGuardrail(request_id=request_id, t_ms=_since(turn_start),
                               verdict=input_verdict, phase="input"))
    if not input_verdict.allowed:
        timings.guardrail_ms = _ms(g0)
        return await _finish_refused(
            query, input_verdict, timings, strategy, mode, [], request_id,
            turn_start, transcript_prefix_ms, emit, stage,
        )
    clean_query = input_verdict.sanitized_query or query

    # --- 2. Embed ---------------------------------------------------------- #
    await stage("embed", "start", budget=config.BUDGET.embed_ms)
    e0 = time.perf_counter()
    query_vector = retriever.embed_query(clean_query) if mode != "sparse" else None
    timings.embed_ms = _ms(e0)
    await stage("embed", "done", ms=timings.embed_ms, budget=config.BUDGET.embed_ms)

    # --- 3. Retrieve ------------------------------------------------------- #
    await stage("retrieval", "start", budget=config.BUDGET.retrieval_ms)
    r0 = time.perf_counter()
    chunks = retriever.search(
        clean_query, query_vector=query_vector, strategy=strategy, mode=mode, top_k=top_k
    )
    timings.retrieval_ms = _ms(r0)
    await stage("retrieval", "done", ms=timings.retrieval_ms,
                budget=config.BUDGET.retrieval_ms, detail=f"{len(chunks)} chunks")
    if emit:
        await emit(WSChunks(request_id=request_id, t_ms=_since(turn_start),
                            strategy=strategy, mode=mode, chunks=chunks))

    # --- 4. Context guardrail --------------------------------------------- #
    ctx_verdict = guardrails.check_context(clean_query, chunks, query_vector=query_vector)
    timings.guardrail_ms = _ms(g0) - timings.embed_ms - timings.retrieval_ms
    await stage("guardrail", "done", ms=timings.guardrail_ms, budget=config.BUDGET.guardrail_ms,
                detail=f"context={ctx_verdict.context_sufficiency}")
    if emit:
        await emit(WSGuardrail(request_id=request_id, t_ms=_since(turn_start),
                               verdict=ctx_verdict, phase="context"))
    if not ctx_verdict.allowed:
        await stage("generation", "skipped", detail="insufficient context")
        return await _finish_refused(
            query, ctx_verdict, timings, strategy, mode, chunks, request_id,
            turn_start, transcript_prefix_ms, emit, stage,
        )

    # --- 5. Generation (streamed) ----------------------------------------- #
    await stage("generation", "start", budget=config.BUDGET.ttft_ms)
    first_token_at: dict[str, float] = {}

    async def on_token(delta: str, is_first: bool) -> None:
        if is_first and "t" not in first_token_at:
            first_token_at["t"] = time.perf_counter()
        if emit:
            await emit(WSToken(request_id=request_id, t_ms=_since(turn_start),
                               text=delta, is_first=is_first))

    gen: GenerationResult = await harness.generate(
        clean_query, chunks, include_thought_process=include_thought_process, on_token=on_token
    )
    timings.ttft_ms = gen.ttft_ms
    timings.generation_ms = gen.generation_ms
    timings.tokens_out = gen.tokens_out
    timings.tokens_per_second = gen.tokens_per_second
    timings.provider = gen.provider
    timings.model = gen.model
    timings.attempts = gen.attempts
    if "t" in first_token_at:
        timings.query_to_first_token_ms = round((first_token_at["t"] - turn_start) * 1000.0, 2)
    await stage("generation", "done", ms=timings.generation_ms, budget=config.BUDGET.ttft_ms,
                detail=f"{gen.tokens_out} tok @ {gen.tokens_per_second} tok/s ({gen.provider})")

    # --- 6. Grounding guardrail ------------------------------------------- #
    await stage("grounding", "start")
    gr0 = time.perf_counter()
    grounding = guardrails.check_grounding(
        gen.response.answer, chunks, llm_flagged_ungrounded=not gen.response.is_grounded
    )
    timings.grounding_ms = _ms(gr0)
    # Merge context sufficiency into the final verdict for the UI.
    grounding.context_sufficiency = ctx_verdict.context_sufficiency
    await stage("grounding", "done", ms=timings.grounding_ms,
                detail=f"score={grounding.grounding_score}")
    if emit:
        await emit(WSGuardrail(request_id=request_id, t_ms=_since(turn_start),
                               verdict=grounding, phase="output"))

    refused = not grounding.allowed
    answer = config.REFUSAL_MESSAGE if refused else gen.response.answer

    timings.query_to_first_token_ms = timings.query_to_first_token_ms or (
        timings.embed_ms + timings.retrieval_ms + timings.guardrail_ms + timings.ttft_ms
    )
    timings.retrieval_to_generation_ms = round(timings.retrieval_ms + timings.ttft_ms, 2)
    timings.stt_ms = transcript_prefix_ms
    timings.total_e2e_ms = round(_since(turn_start) + transcript_prefix_ms, 2)
    STATE.record(timings)

    response = QueryResponse(
        query=query, transcript=query, answer=answer, refused=refused,
        confidence=0.0 if refused else gen.response.confidence,
        thought_process=gen.response.thought_process if include_thought_process else "",
        strategy=strategy, mode=mode, chunks=chunks,
        cited_chunk_ids=[] if refused else gen.response.cited_chunk_ids,
        guardrail=grounding, timings=timings, request_id=request_id,
    )
    if emit:
        await emit(_done_event(response, turn_start))
    return response


async def _finish_refused(
    query, verdict, timings, strategy, mode, chunks, request_id,
    turn_start, transcript_prefix_ms, emit, stage,
) -> QueryResponse:
    """Common exit for a blocked/insufficient turn: emit refusal, record, return."""
    timings.stt_ms = transcript_prefix_ms
    timings.query_to_first_token_ms = round(
        timings.embed_ms + timings.retrieval_ms + timings.guardrail_ms, 2
    )
    timings.total_e2e_ms = round(_since(turn_start) + transcript_prefix_ms, 2)
    STATE.record(timings)
    response = QueryResponse(
        query=query, transcript=query, answer=config.REFUSAL_MESSAGE, refused=True,
        confidence=0.0, strategy=strategy, mode=mode, chunks=chunks,
        guardrail=verdict, timings=timings, request_id=request_id,
    )
    if emit:
        await emit(_done_event(response, turn_start))
    return response


def _done_event(response: QueryResponse, turn_start: float) -> WSDone:
    return WSDone(
        request_id=response.request_id, t_ms=_since(turn_start),
        answer=response.answer, refused=response.refused, confidence=response.confidence,
        thought_process=response.thought_process, cited_chunk_ids=response.cited_chunk_ids,
        guardrail=response.guardrail, timings=response.timings,
    )


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


def _since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


# --------------------------------------------------------------------------- #
# REST endpoints
# --------------------------------------------------------------------------- #


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    retriever = get_retriever() if STATE.ready else None
    harness = get_harness()
    status = "ok" if STATE.ready else ("initialising" if STATE.indexing else "degraded")
    return HealthResponse(
        status=status,  # type: ignore[arg-type]
        version=__version__,
        dataset=config.HF_DATASET_REPO,
        indexed_strategies=ingest.indexed_strategies() if STATE.ready else [],  # type: ignore[arg-type]
        default_strategy=config.DEFAULT_STRATEGY,  # type: ignore[arg-type]
        total_chunks=retriever.total_chunks() if retriever else 0,
        languages=[s.key for s in config.resolve_languages()],
        embed_model=config.EMBED_MODEL,
        embed_dim=config.EMBED_DIM,
        llm_provider=harness.primary_provider,
        llm_model=harness.primary_model,
        stt_provider=(config.stt_chain() or ["offline"])[0],
        stt_model=config.SARVAM_MODEL if config.SARVAM_API_KEY else config.GROQ_STT_MODEL,
        offline_mode=not config.llm_configured(),
        warnings=STATE.warnings + ([STATE.error] if STATE.error else []),
    )


def _guard_ready() -> JSONResponse | None:
    if STATE.ready:
        return None
    detail = STATE.error or ("indexing in progress" if STATE.indexing else "not ready")
    return JSONResponse(status_code=503, content={"detail": detail, "indexing": STATE.indexing})


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    not_ready = _guard_ready()
    if not_ready:
        return not_ready
    return await run_turn(
        request.query, strategy=request.strategy, mode=request.mode, top_k=request.top_k,
        include_thought_process=request.include_thought_process,
    )


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """Server-Sent Events variant: one `data:` line of JSON per pipeline event."""
    not_ready = _guard_ready()
    if not_ready:
        return not_ready

    queue: asyncio.Queue = asyncio.Queue()

    async def emit(event) -> None:
        await queue.put(event)

    async def producer() -> None:
        try:
            await run_turn(
                request.query, strategy=request.strategy, mode=request.mode,
                top_k=request.top_k, include_thought_process=request.include_thought_process,
                emit=emit,
            )
        except Exception as exc:  # pragma: no cover
            await queue.put(WSError(message=str(exc), fatal=True))
        finally:
            await queue.put(None)

    async def event_stream():
        task = asyncio.create_task(producer())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {event.model_dump_json()}\n\n"
        finally:
            task.cancel()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
):
    audio = await file.read()
    try:
        result = await stt.transcribe(
            audio, is_pcm=False,
            content_type=file.content_type or "audio/wav",
            filename=file.filename or "audio.wav",
            language=language,
        )
    except stt.STTUnavailable as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    except stt.STTError as exc:
        return JSONResponse(status_code=502, content={"detail": str(exc)})
    return TranscriptionResponse(
        transcript=result.transcript, language_code=result.language_code,
        language_probability=result.language_probability, provider=result.provider,
        model=result.model, stt_ms=result.stt_ms, audio_seconds=result.audio_seconds,
        request_id=uuid.uuid4().hex[:12],
    )


@app.get("/api/analytics", response_model=AnalyticsResponse)
async def analytics_endpoint() -> AnalyticsResponse:
    from .benchmark import load_report

    report = load_report()
    live = _live_summaries()
    chunk_stats = []
    if STATE.ready:
        for strategy in ingest.indexed_strategies():
            st = ingest.load_stats(strategy)
            if st:
                chunk_stats.append(st)
    return AnalyticsResponse(
        available=report is not None or bool(live),
        report=report,
        chunk_stats=chunk_stats,
        live_requests=len(STATE.live_timings),
        live=live,
        budget=_budget_dict(),
        generated_at=report.generated_at if report else "",
        benchmark_running=STATE.benchmark_running,
    )


@app.post("/api/benchmark")
async def benchmark_endpoint(request: BenchmarkRequest = Body(default=BenchmarkRequest())):
    not_ready = _guard_ready()
    if not_ready:
        return not_ready
    if STATE.benchmark_running:
        return JSONResponse(status_code=409, content={"detail": "benchmark already running"})

    from .benchmark import run_benchmark

    async def runner():
        STATE.benchmark_running = True
        try:
            await run_benchmark(
                queries=request.queries, concurrency=request.concurrency,
                strategies=request.strategies, modes=request.modes,
                include_llm=request.include_llm,
            )
        except Exception:
            logger.exception("benchmark failed")
        finally:
            STATE.benchmark_running = False

    asyncio.create_task(runner())
    return {"status": "started", "queries": request.queries}


def _live_summaries() -> dict:
    rows = list(STATE.live_timings)
    if not rows:
        return {}
    budget = config.BUDGET
    fields = {
        "embed_ms": budget.embed_ms, "retrieval_ms": budget.retrieval_ms,
        "guardrail_ms": budget.guardrail_ms, "ttft_ms": budget.ttft_ms,
        "query_to_first_token_ms": budget.total_ms, "total_e2e_ms": 0.0,
        "generation_ms": 0.0, "grounding_ms": 0.0,
    }
    out = {}
    for name, bud in fields.items():
        values = [getattr(t, name) for t in rows if getattr(t, name) > 0]
        if values:
            out[name] = summarize(name, values, budget_ms=bud)
    return out


def _budget_dict() -> dict[str, float]:
    b = config.BUDGET
    return {"embed_ms": b.embed_ms, "retrieval_ms": b.retrieval_ms,
            "guardrail_ms": b.guardrail_ms, "ttft_ms": b.ttft_ms, "total_ms": b.total_ms}


# --------------------------------------------------------------------------- #
# WebSocket - the real-time voice path
# --------------------------------------------------------------------------- #


@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket) -> None:
    await ws.accept()
    lock = asyncio.Lock()

    async def emit(event) -> None:
        # Serialise sends so streamed tokens never interleave with stage frames.
        async with lock:
            await ws.send_text(event.model_dump_json())

    # Per-connection config, overridable by a `config` frame.
    settings = {
        "strategy": config.DEFAULT_STRATEGY, "mode": "hybrid",
        "top_k": config.RETRIEVE_TOP_K, "language": None,
        "include_thought_process": False, "sample_rate": config.AUDIO_SAMPLE_RATE,
    }
    audio_chunks: list[bytes] = []

    await emit(WSReady(strategy=settings["strategy"], offline_mode=not config.llm_configured(),
                       sample_rate=settings["sample_rate"]))

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if message.get("bytes") is not None:
                # A binary frame is a chunk of raw PCM16 audio.
                audio_chunks.append(message["bytes"])
                continue

            text = message.get("text")
            if not text:
                continue
            try:
                frame = WSClientConfig.model_validate_json(text)
            except Exception as exc:
                await emit(WSError(message=f"bad control frame: {exc}", code="bad_frame"))
                continue

            if frame.type == "config":
                _apply_config(settings, frame)
                continue
            if frame.type == "cancel":
                audio_chunks.clear()
                continue
            if frame.type == "text" and frame.text:
                _apply_config(settings, frame)
                if not _ws_ready_guard(emit):
                    await _emit_not_ready(emit)
                    continue
                await run_turn(
                    frame.text, strategy=settings["strategy"], mode=settings["mode"],
                    top_k=settings["top_k"], include_thought_process=settings["include_thought_process"],
                    emit=emit,
                )
                continue
            if frame.type == "end":
                _apply_config(settings, frame)
                pcm = b"".join(audio_chunks)
                audio_chunks.clear()
                await _handle_audio_turn(pcm, settings, emit)
                continue
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover
        logger.exception("ws error")
        try:
            await emit(WSError(message=str(exc), fatal=True))
        except Exception:
            pass


async def _handle_audio_turn(pcm: bytes, settings: dict, emit: EmitFn) -> None:
    request_id = uuid.uuid4().hex[:12]
    if not pcm:
        await emit(WSError(request_id=request_id, message="no audio received", code="empty_audio"))
        return
    # STT stage.
    await emit(WSStage(request_id=request_id, stage="stt", status="start"))
    try:
        result = await stt.transcribe(
            pcm, is_pcm=True, sample_rate=settings["sample_rate"], language=settings["language"]
        )
    except stt.STTUnavailable as exc:
        await emit(WSStage(request_id=request_id, stage="stt", status="error", detail=str(exc)))
        await emit(WSError(request_id=request_id, message=str(exc), code="stt_unavailable"))
        return
    except stt.STTError as exc:
        await emit(WSStage(request_id=request_id, stage="stt", status="error", detail=str(exc)))
        await emit(WSError(request_id=request_id, message=str(exc), code="stt_failed"))
        return

    await emit(WSStage(request_id=request_id, stage="stt", status="done", ms=result.stt_ms))
    await emit(WSTranscript(request_id=request_id, text=result.transcript, is_final=True,
                            language_code=result.language_code, provider=result.provider,
                            audio_seconds=result.audio_seconds))
    if not _ws_ready_guard(emit):
        await _emit_not_ready(emit, request_id)
        return
    await run_turn(
        result.transcript, strategy=settings["strategy"], mode=settings["mode"],
        top_k=settings["top_k"], include_thought_process=settings["include_thought_process"],
        request_id=request_id, transcript_prefix_ms=result.stt_ms, emit=emit,
    )


def _apply_config(settings: dict, frame: WSClientConfig) -> None:
    if frame.strategy:
        settings["strategy"] = frame.strategy
    if frame.mode:
        settings["mode"] = frame.mode
    if frame.top_k:
        settings["top_k"] = frame.top_k
    if frame.language is not None:
        settings["language"] = frame.language
    if frame.include_thought_process is not None:
        settings["include_thought_process"] = frame.include_thought_process
    if frame.sample_rate:
        settings["sample_rate"] = frame.sample_rate


def _ws_ready_guard(emit: EmitFn) -> bool:
    return STATE.ready


async def _emit_not_ready(emit: EmitFn, request_id: str = "") -> None:
    detail = STATE.error or ("indexing in progress - try again shortly" if STATE.indexing else "not ready")
    await emit(WSError(request_id=request_id, message=detail, code="not_ready"))


@app.get("/")
async def root():
    return {"service": "voice-rag-goa", "version": __version__, "docs": "/docs",
            "health": "/api/health", "ws": "/ws/audio"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, reload=False)
