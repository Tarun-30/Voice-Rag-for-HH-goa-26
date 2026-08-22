"""
LLM orchestration harness.

Responsibilities, in one place so the pipeline stays honest about failure:

* **Structured output** - the model must return JSON that validates against
  `schemas.AnswerResponse` (thought_process / answer / confidence /
  cited_chunk_ids / is_grounded). Anything else is a *retryable* error.
* **Streaming** - we stream the completion and pull the `answer` field's value
  out of the growing JSON incrementally, so the UI sees real tokens (real TTFT)
  even though the wire format is a JSON object.
* **Retries** - up to `LLM_MAX_RETRIES` with exponential backoff on timeouts,
  5xx, malformed JSON, and schema-validation failures.
* **Failover** - Groq primary model -> Groq fallback model -> Cerebras. Each
  target has a circuit breaker so a provider having a bad day is skipped fast
  instead of eating the timeout on every request.
* **Offline mode** - if no provider is configured (or all fail), a deterministic
  extractive answerer composes an answer from the retrieved context. The system
  still demonstrably works end to end with zero API keys.

Groq and Cerebras are both OpenAI-compatible, so one httpx SSE reader drives
both. We use `response_format={"type":"json_object"}`, which the gpt-oss models
honour, plus a schema-describing system prompt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Sequence

import httpx

from . import config
from . import textutil
from .schemas import AnswerResponse, RetrievedChunk

logger = logging.getLogger(__name__)

TokenCallback = Callable[[str, bool], Awaitable[None]] | Callable[[str, bool], None]


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class GenerationResult:
    response: AnswerResponse
    provider: str
    model: str
    attempts: int
    ttft_ms: float
    generation_ms: float
    tokens_out: int
    raw: str
    used_offline: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        seconds = self.generation_ms / 1000.0
        return round(self.tokens_out / seconds, 1) if seconds > 0 else 0.0


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #


class CircuitBreaker:
    """Open after N consecutive failures; auto-close after a cooldown."""

    def __init__(self, threshold: int, cooldown_s: float):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._failures = 0
        self._opened_at = 0.0

    def is_open(self, now: float) -> bool:
        if self._failures < self.threshold:
            return False
        if now - self._opened_at >= self.cooldown_s:
            # Cooldown elapsed: half-open, allow one probe.
            self._failures = self.threshold - 1
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = 0.0

    def record_failure(self, now: float) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_at = now


# --------------------------------------------------------------------------- #
# Provider targets
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Target:
    provider: str
    base_url: str
    path: str
    model: str
    api_key: str


def _build_targets() -> list[Target]:
    """Ordered, key-filtered list of (provider, model) endpoints to try."""
    targets: list[Target] = []
    if config.GROQ_API_KEY:
        targets.append(
            Target("groq", config.GROQ_BASE_URL, "/openai/v1/chat/completions",
                   config.GROQ_MODEL, config.GROQ_API_KEY)
        )
        if config.GROQ_FALLBACK_MODEL and config.GROQ_FALLBACK_MODEL != config.GROQ_MODEL:
            targets.append(
                Target("groq", config.GROQ_BASE_URL, "/openai/v1/chat/completions",
                       config.GROQ_FALLBACK_MODEL, config.GROQ_API_KEY)
            )
    if config.CEREBRAS_API_KEY:
        targets.append(
            Target("cerebras", config.CEREBRAS_BASE_URL, "/chat/completions",
                   config.CEREBRAS_MODEL, config.CEREBRAS_API_KEY)
        )
    return targets


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """You are a retrieval-grounded answering engine. You answer \
ONLY from the numbered context passages provided. You never use outside \
knowledge.

Rules:
- If the context contains the answer, answer concisely (1-4 sentences).
- If the context does NOT contain the answer, set "is_grounded" to false and \
set "answer" to exactly: "{refusal}"
- Cite the id of every passage you used in "cited_chunk_ids" (e.g. ["{example}"]).
- "confidence" is your calibrated confidence in [0,1] that the answer is correct \
and supported by the context.

Respond with a single JSON object and NOTHING else, exactly this shape:
{{"thought_process": string, "answer": string, "confidence": number, \
"cited_chunk_ids": array of strings, "is_grounded": boolean}}"""


def _build_messages(
    query: str, chunks: Sequence[RetrievedChunk], include_thought_process: bool
) -> list[dict[str, str]]:
    context = "\n\n".join(chunk.as_context_block() for chunk in chunks) or "(no context retrieved)"
    example_id = chunks[0].chunk_id if chunks else "0:0#hie0"
    system = _SYSTEM_PROMPT.format(refusal=config.REFUSAL_MESSAGE, example=example_id)
    thought_hint = (
        "Include a brief 'thought_process'."
        if include_thought_process
        else "Keep 'thought_process' to one short sentence."
    )
    user = (
        f"CONTEXT PASSAGES:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"{thought_hint} Answer strictly from the context above as JSON."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# Incremental JSON "answer" extraction
# --------------------------------------------------------------------------- #


def _extract_partial_answer(raw: str) -> str:
    """Best-effort decode of the (possibly incomplete) `answer` string so far.

    The model streams a JSON object; we want to show the human-readable answer as
    it arrives rather than raw JSON. Find the "answer" key, then walk its string
    value honouring escapes, stopping at the closing quote or the end of what has
    streamed so far.
    """
    key = raw.find('"answer"')
    if key == -1:
        return ""
    colon = raw.find(":", key + 8)
    if colon == -1:
        return ""
    quote = raw.find('"', colon + 1)
    if quote == -1:
        return ""
    out: list[str] = []
    i = quote + 1
    while i < len(raw):
        char = raw[i]
        if char == "\\" and i + 1 < len(raw):
            nxt = raw[i + 1]
            out.append({"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}.get(nxt, nxt))
            i += 2
            continue
        if char == '"':
            break
        out.append(char)
        i += 1
    return "".join(out)


def _salvage_json(raw: str) -> dict | None:
    """Extract the first balanced JSON object from `raw`, tolerating chatter."""
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        char = raw[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


class Harness:
    def __init__(self) -> None:
        self._targets = _build_targets()
        self._breakers: dict[str, CircuitBreaker] = {
            f"{t.provider}:{t.model}": CircuitBreaker(config.BREAKER_THRESHOLD, config.BREAKER_COOLDOWN_S)
            for t in self._targets
        }
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._targets)

    @property
    def primary_provider(self) -> str:
        return self._targets[0].provider if self._targets else "offline"

    @property
    def primary_model(self) -> str:
        return self._targets[0].model if self._targets else "extractive-fallback"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=config.LLM_TIMEOUT_S)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        *,
        include_thought_process: bool = False,
        on_token: TokenCallback | None = None,
    ) -> GenerationResult:
        """Generate a grounded, validated answer, streaming tokens via on_token."""
        if not self._targets:
            return await self._offline(query, chunks, on_token)

        messages = _build_messages(query, chunks, include_thought_process)
        errors: list[str] = []
        attempts = 0

        for target in self._targets:
            breaker = self._breakers[f"{target.provider}:{target.model}"]
            for attempt in range(config.LLM_MAX_RETRIES + 1):
                now = time.perf_counter()
                if breaker.is_open(now):
                    errors.append(f"{target.provider}:{target.model} breaker open")
                    break
                attempts += 1
                try:
                    result = await self._stream_once(target, messages, on_token, attempts, errors)
                    breaker.record_success()
                    return result
                except _NonRetryable as exc:
                    breaker.record_failure(time.perf_counter())
                    errors.append(f"{target.provider}:{target.model} fatal: {exc}")
                    break  # do not retry this target (auth/4xx)
                except Exception as exc:
                    breaker.record_failure(time.perf_counter())
                    errors.append(f"{target.provider}:{target.model} attempt {attempt+1}: {exc}")
                    logger.warning("llm %s attempt %d failed: %s", target.model, attempt + 1, exc)
                    if attempt < config.LLM_MAX_RETRIES:
                        await asyncio.sleep(
                            min(config.LLM_BACKOFF_MAX_S, config.LLM_BACKOFF_BASE_S * (2 ** attempt))
                        )

        logger.error("all llm targets failed; using offline fallback: %s", errors)
        return await self._offline(query, chunks, on_token, errors=errors, attempts=attempts)

    async def _stream_once(
        self,
        target: Target,
        messages: list[dict[str, str]],
        on_token: TokenCallback | None,
        attempts: int,
        errors: list[str],
    ) -> GenerationResult:
        client = await self._get_client()
        body = {
            "model": target.model,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
            "stream": True,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {target.api_key}", "Content-Type": "application/json"}

        raw_parts: list[str] = []
        emitted = 0
        tokens_out = 0
        started = time.perf_counter()
        ttft_ms = 0.0

        async with client.stream(
            "POST", target.base_url.rstrip("/") + target.path, json=body, headers=headers
        ) as response:
            if response.status_code >= 400:
                text = (await response.aread()).decode("utf-8", "replace")[:300]
                if response.status_code in (400, 401, 403, 404, 422):
                    raise _NonRetryable(f"HTTP {response.status_code}: {text}")
                raise RuntimeError(f"HTTP {response.status_code}: {text}")

            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (event.get("choices") or [{}])[0].get("delta", {})
                piece = delta.get("content") or ""
                if not piece:
                    continue
                if ttft_ms == 0.0:
                    ttft_ms = (time.perf_counter() - started) * 1000.0
                tokens_out += 1
                raw_parts.append(piece)
                # Stream the human-facing answer text incrementally.
                if on_token is not None:
                    current = _extract_partial_answer("".join(raw_parts))
                    if len(current) > emitted:
                        await _maybe_await(on_token(current[emitted:], emitted == 0))
                        emitted = len(current)

        raw = "".join(raw_parts)
        generation_ms = (time.perf_counter() - started) * 1000.0
        parsed = _salvage_json(raw)
        if parsed is None:
            raise RuntimeError(f"no valid JSON in completion: {raw[:120]!r}")
        response_obj = AnswerResponse.model_validate(parsed)  # raises on schema mismatch -> retry

        # If we somehow never streamed (answer arrived un-extractably), flush it.
        if on_token is not None and emitted == 0 and response_obj.answer:
            await _maybe_await(on_token(response_obj.answer, True))

        return GenerationResult(
            response=response_obj,
            provider=target.provider,
            model=target.model,
            attempts=attempts,
            ttft_ms=round(ttft_ms, 2),
            generation_ms=round(generation_ms, 2),
            tokens_out=tokens_out,
            raw=raw,
            errors=errors,
        )

    # -- offline fallback --------------------------------------------------- #

    async def _offline(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        on_token: TokenCallback | None,
        *,
        errors: list[str] | None = None,
        attempts: int = 0,
    ) -> GenerationResult:
        """Deterministic extractive answerer - no API keys required.

        Picks the sentences from the top chunks that best lexically overlap the
        query, stitches them into a short answer, and streams it word by word so
        the UI behaves identically to the LLM path.
        """
        started = time.perf_counter()
        answer, cited, confidence, grounded = _extractive_answer(query, chunks)

        ttft_ms = 0.0
        tokens = answer.split(" ")
        for i, word in enumerate(tokens):
            if on_token is not None:
                if ttft_ms == 0.0:
                    ttft_ms = (time.perf_counter() - started) * 1000.0
                await _maybe_await(on_token(word + (" " if i < len(tokens) - 1 else ""), i == 0))
                await asyncio.sleep(0.006)  # gentle typing cadence
        generation_ms = (time.perf_counter() - started) * 1000.0

        response = AnswerResponse(
            thought_process="Offline extractive mode: answer composed from the "
            "highest-scoring retrieved sentences (no generative LLM configured).",
            answer=answer,
            confidence=confidence,
            cited_chunk_ids=cited,
            is_grounded=grounded,
        )
        return GenerationResult(
            response=response,
            provider="offline",
            model="extractive-fallback",
            attempts=attempts,
            ttft_ms=round(ttft_ms or 1.0, 2),
            generation_ms=round(generation_ms, 2),
            tokens_out=len(tokens),
            raw=answer,
            used_offline=True,
            errors=errors or [],
        )


def _extractive_answer(
    query: str, chunks: Sequence[RetrievedChunk]
) -> tuple[str, list[str], float, bool]:
    if not chunks:
        return config.REFUSAL_MESSAGE, [], 0.0, False

    query_terms = set(textutil.tokenize_for_bm25(query))
    best_sentences: list[tuple[float, str, str]] = []
    for chunk in chunks[:3]:
        for sentence in textutil.split_sentences(chunk.context_text):
            terms = set(textutil.tokenize_for_bm25(sentence))
            if not terms:
                continue
            overlap = len(query_terms & terms) / max(1, len(query_terms))
            best_sentences.append((overlap, sentence, chunk.chunk_id))
    best_sentences.sort(key=lambda item: item[0], reverse=True)

    top = [s for s in best_sentences if s[0] > 0][:2]
    if not top:
        # Nothing overlapped - fall back to the single highest-ranked chunk lead.
        lead = textutil.split_sentences(chunks[0].context_text)[:1]
        answer = " ".join(lead) if lead else chunks[0].context_text[:280]
        return textutil.truncate(answer, 400), [chunks[0].chunk_id], 0.3, True

    answer = textutil.truncate(" ".join(s[1] for s in top), 400)
    cited = textutil.dedupe_preserving_order([s[2] for s in top])
    confidence = min(0.85, 0.4 + top[0][0] * 0.5)
    return answer, cited, round(confidence, 2), True


class _NonRetryable(Exception):
    """A provider error there is no point retrying (auth, bad request)."""


async def _maybe_await(value) -> None:
    if asyncio.iscoroutine(value):
        await value


# --------------------------------------------------------------------------- #
# Singleton
# --------------------------------------------------------------------------- #

_harness: Harness | None = None


def get_harness() -> Harness:
    global _harness
    if _harness is None:
        _harness = Harness()
    return _harness
