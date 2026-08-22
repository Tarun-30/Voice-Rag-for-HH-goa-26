"""
Guardrails: three checkpoints around generation.

    1. INPUT    (before retrieval)   check_input()
       Sanitise the query, reject empties/oversize/gibberish, and flag prompt
       injection and unsafe requests.

    2. CONTEXT  (after retrieval, before generation)   check_context()
       Is the answer even in the corpus? If the best passage's semantic
       similarity to the query is below CONTEXT_SUFFICIENCY_THRESHOLD, we refuse
       *before* spending an LLM call - the honest "not in the dataset" path.

    3. GROUNDING (after generation)   check_grounding()
       Does the produced answer actually follow from the retrieved context?
       Score each answer sentence against the context (semantic cosine OR lexical
       coverage - either is enough), average it, and if the mean is below
       GROUNDING_THRESHOLD, replace the answer with the exact refusal string.

The refusal string is fixed by spec and returned verbatim:
    "I cannot answer this based on the verified dataset."

Grounding uses the embedder we already host, so no extra model is loaded. The
grounding pass runs *after* first-token latency is measured, so its cost never
counts against the sub-200 ms query-to-first-token target.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

import numpy as np

from . import config
from . import textutil
from .embeddings import cosine_matrix, get_embedder
from .schemas import AnswerResponse, GuardrailCheck, GuardrailVerdict, RetrievedChunk

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pattern banks
# --------------------------------------------------------------------------- #

# Prompt-injection / jailbreak attempts. Matched against the NFKC-folded,
# lower-cased query, so fullwidth or zero-width evasion is already neutralised by
# sanitize_text upstream.
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\b"), "ignore-previous"),
    (re.compile(r"\bdisregard\s+(all\s+|the\s+|your\s+)?(previous|prior|above|instructions?|rules?)\b"), "disregard"),
    (re.compile(r"\b(system|developer)\s+(prompt|message|instructions?)\b"), "system-prompt-probe"),
    (re.compile(r"\b(reveal|show|print|repeat|output|leak)\b.{0,30}\b(prompt|instructions?|system)\b"), "leak-prompt"),
    (re.compile(r"\byou\s+are\s+now\b|\bact\s+as\b.{0,30}\b(dan|jailbreak|unrestricted|no\s+rules)\b"), "role-override"),
    (re.compile(r"\bjailbreak\b|\bdeveloper\s+mode\b|\bdo\s+anything\s+now\b"), "jailbreak"),
    (re.compile(r"\bforget\s+(everything|all|your)\b"), "forget"),
    (re.compile(r"\bnew\s+instructions?\s*:|\boverride\b.{0,20}\b(instructions?|rules?|safety)\b"), "override"),
    (re.compile(r"\b(pretend|imagine)\b.{0,40}\bno\s+(restrictions?|rules?|guidelines?)\b"), "pretend-unrestricted"),
]

# Unsafe-content requests. Deliberately narrow: this is a factual QA system over
# a web corpus, and over-blocking legitimate questions ("what is a corporation")
# is worse than the rare miss. We match requests to *produce* harm, not mentions.
_UNSAFE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bhow\s+(to|do\s+i|can\s+i)\b.{0,40}\b(make|build|synthesi[sz]e|manufacture)\b.{0,40}\b(bomb|explosive|nerve\s+agent|bioweapon|meth(amphetamine)?|nuclear\s+weapon)\b"), "weapon-synthesis"),
    (re.compile(r"\b(synthesi[sz]e|produce|manufacture)\b.{0,30}\b(sarin|vx|ricin|anthrax)\b"), "chemical-weapon"),
    (re.compile(r"\bhow\s+(to|do\s+i)\b.{0,30}\b(kill|murder|poison)\b\s+(someone|a\s+person|my|him|her|them)\b"), "violence"),
    (re.compile(r"\b(how\s+to|ways?\s+to|best\s+way\s+to)\b.{0,20}\b(kill\s+myself|commit\s+suicide|end\s+my\s+life)\b"), "self-harm"),
    (re.compile(r"\b(write|generate|create)\b.{0,30}\b(malware|ransomware|keylogger|computer\s+virus)\b"), "malware"),
]


# --------------------------------------------------------------------------- #
# 1. Input guardrail
# --------------------------------------------------------------------------- #


def check_input(query: str) -> GuardrailVerdict:
    """Sanitise and screen a raw query before it enters the pipeline."""
    checks: list[GuardrailCheck] = []
    sanitized = textutil.sanitize_text(query, max_chars=config.MAX_QUERY_CHARS + 64)

    # Length.
    length_ok = len(sanitized) >= config.MIN_QUERY_CHARS
    checks.append(GuardrailCheck(
        name="length", passed=length_ok, score=float(len(sanitized)),
        threshold=float(config.MIN_QUERY_CHARS),
        detail="" if length_ok else "query too short",
    ))
    if not sanitized:
        return _verdict("block", "empty_query", "Empty query after sanitisation.", 1.0, checks, sanitized)
    if not length_ok:
        return _verdict("block", "empty_query", "Query is too short to answer.", 0.9, checks, sanitized)
    if len(query) > config.MAX_QUERY_CHARS:
        checks.append(GuardrailCheck(name="max_length", passed=False, score=float(len(query)),
                                     threshold=float(config.MAX_QUERY_CHARS), detail="over max length"))
        return _verdict("block", "too_long",
                        f"Query exceeds {config.MAX_QUERY_CHARS} characters.", 0.7, checks, sanitized)

    lowered = sanitized.lower()

    # Prompt injection.
    injection_hits = [tag for pattern, tag in _INJECTION_PATTERNS if pattern.search(lowered)]
    checks.append(GuardrailCheck(
        name="prompt_injection", passed=not injection_hits,
        score=float(len(injection_hits)), threshold=1.0,
        detail=", ".join(injection_hits),
    ))
    if injection_hits:
        return _verdict("block", "prompt_injection",
                        f"Possible prompt-injection attempt ({injection_hits[0]}).",
                        1.0, checks, sanitized)

    # Unsafe content.
    unsafe_hits = [tag for pattern, tag in _UNSAFE_PATTERNS if pattern.search(lowered)]
    checks.append(GuardrailCheck(
        name="unsafe_content", passed=not unsafe_hits,
        score=float(len(unsafe_hits)), threshold=1.0, detail=", ".join(unsafe_hits),
    ))
    if unsafe_hits:
        return _verdict("block", "unsafe_content",
                        "This request asks for potentially harmful content and cannot be processed.",
                        1.0, checks, sanitized)

    # Gibberish / ASR-hallucination-on-silence.
    is_junk, reason = textutil.looks_like_gibberish(sanitized)
    checks.append(GuardrailCheck(name="coherence", passed=not is_junk, detail=reason))
    if is_junk:
        return _verdict("block", "gibberish",
                        "Query does not appear to be a coherent question.", 0.8, checks, sanitized)

    return _verdict("allow", "ok", "", 0.0, checks, sanitized)


# --------------------------------------------------------------------------- #
# 2. Context sufficiency
# --------------------------------------------------------------------------- #


def check_context(
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    query_vector: np.ndarray | None = None,
) -> GuardrailVerdict:
    """Decide whether retrieval found anything good enough to answer from.

    Uses the semantic similarity between the query and the best retrieved chunk.
    Prefers the dense score already computed during retrieval; if the search was
    sparse-only (no dense score), it embeds the top chunks here to get a real
    cosine. Below CONTEXT_SUFFICIENCY_THRESHOLD -> refuse without calling the LLM.
    """
    checks: list[GuardrailCheck] = []
    if not chunks:
        verdict = _verdict("refuse", "insufficient_context",
                           "No passages were retrieved for this query.", 1.0, checks, query)
        verdict.context_sufficiency = 0.0
        return verdict

    sufficiency = max((c.dense_score for c in chunks), default=0.0)
    if sufficiency <= 0.0:
        # Sparse-only path: compute a genuine similarity now.
        embedder = get_embedder()
        qv = query_vector if query_vector is not None else embedder.embed_query(query)
        top_texts = [c.context_text for c in chunks[:3]]
        matrix = embedder.embed_passages(top_texts, sort_by_length=False)
        sims = cosine_matrix(qv, matrix)[0]
        sufficiency = float(sims.max()) if sims.size else 0.0

    threshold = config.CONTEXT_SUFFICIENCY_THRESHOLD
    passed = sufficiency >= threshold
    checks.append(GuardrailCheck(
        name="context_sufficiency", passed=passed, score=round(sufficiency, 4),
        threshold=threshold,
        detail="best passage similarity to the query",
    ))
    decision = "allow" if passed else "refuse"
    category = "ok" if passed else "insufficient_context"
    reason = "" if passed else (
        f"Best passage similarity {sufficiency:.2f} < {threshold:.2f}; "
        "the answer is not in the verified dataset."
    )
    verdict = _verdict(decision, category, reason, round(1.0 - sufficiency, 3), checks, query)
    verdict.context_sufficiency = round(sufficiency, 4)
    return verdict


# --------------------------------------------------------------------------- #
# 3. Grounding
# --------------------------------------------------------------------------- #


def check_grounding(
    answer: str,
    chunks: Sequence[RetrievedChunk],
    *,
    llm_flagged_ungrounded: bool = False,
) -> GuardrailVerdict:
    """Verify the answer follows from the context. Sentence by sentence.

    A sentence is considered grounded if EITHER its embedding cosine to some
    context chunk clears GROUNDING_THRESHOLD, OR its content-word coverage by the
    context is high (lexical backstop for short factual statements that embed
    poorly). The verdict's grounding_score is the mean per-sentence best score.
    """
    checks: list[GuardrailCheck] = []
    answer = (answer or "").strip()

    # A model that already refused is trivially "grounded" in the refusal.
    if not answer or answer == config.REFUSAL_MESSAGE:
        verdict = _verdict("refuse", "insufficient_context", "Model refused to answer.", 1.0, checks, "")
        verdict.grounding_score = 0.0
        return verdict

    sentences = textutil.split_sentences(answer)
    if not sentences:
        sentences = [answer]
    context_texts = [c.context_text for c in chunks] or [""]

    embedder = get_embedder()
    sentence_vecs = embedder.embed_passages(sentences, sort_by_length=False)
    context_vecs = embedder.embed_passages(context_texts, sort_by_length=False)
    sims = cosine_matrix(sentence_vecs, context_vecs)  # (S, C)

    per_sentence: list[float] = []
    unsupported: list[str] = []
    context_joined = " ".join(context_texts)
    for i, sentence in enumerate(sentences):
        semantic = float(sims[i].max()) if sims.shape[1] else 0.0
        lexical = textutil.coverage(sentence, context_joined)
        score = max(semantic, lexical)
        per_sentence.append(score)
        if score < config.GROUNDING_THRESHOLD:
            unsupported.append(textutil.truncate(sentence, 120))

    grounding_score = float(np.mean(per_sentence)) if per_sentence else 0.0
    grounded_sentences = sum(1 for s in per_sentence if s >= config.GROUNDING_THRESHOLD)

    checks.append(GuardrailCheck(
        name="grounding", passed=grounding_score >= config.GROUNDING_THRESHOLD,
        score=round(grounding_score, 4), threshold=config.GROUNDING_THRESHOLD,
        detail=f"{grounded_sentences}/{len(sentences)} sentences grounded",
    ))

    passed = grounding_score >= config.GROUNDING_THRESHOLD and not llm_flagged_ungrounded
    decision = "allow" if passed else "refuse"
    category = "ok" if passed else "ungrounded_answer"
    reason = "" if passed else (
        f"Answer grounding {grounding_score:.2f} < {config.GROUNDING_THRESHOLD:.2f}"
        + ("; model self-flagged ungrounded" if llm_flagged_ungrounded else "")
    )
    verdict = _verdict(decision, category, reason, round(1.0 - grounding_score, 3), checks, "")
    verdict.grounding_score = round(grounding_score, 4)
    verdict.grounded_sentences = grounded_sentences
    verdict.total_sentences = len(sentences)
    verdict.unsupported_claims = unsupported
    return verdict


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _verdict(
    decision: str, category: str, reason: str, risk: float,
    checks: list[GuardrailCheck], sanitized: str,
) -> GuardrailVerdict:
    return GuardrailVerdict(
        decision=decision,  # type: ignore[arg-type]
        category=category,  # type: ignore[arg-type]
        reason=reason,
        risk_score=round(risk, 3),
        checks=checks,
        sanitized_query=sanitized,
    )


def refusal_answer() -> AnswerResponse:
    """The canonical refusal, as a validated AnswerResponse."""
    return AnswerResponse(
        thought_process="Blocked or unsupported by the verified dataset.",
        answer=config.REFUSAL_MESSAGE,
        confidence=0.0,
        cited_chunk_ids=[],
        is_grounded=False,
    )
