"""
Speech-to-text with provider failover.

Three providers, tried in the order `config.stt_chain()` returns (which is the
configured-and-keyed subset of STT_FALLBACK_ORDER):

* **sarvam**     - best for the 14 Indian languages in this dataset. In
                   `translate` mode the `saaras` model returns English text
                   directly, which is what an English-indexed corpus wants.
* **groq**       - Whisper-large-v3-turbo over the OpenAI-compatible endpoint.
                   Fast and cheap; good multilingual fallback.
* **elevenlabs** - Scribe. Included for completeness / redundancy.

The browser streams raw PCM16 @ 16 kHz (Web Audio gives us Float32 which the
client downsamples and converts). Both vendors want a real container, not loose
samples, so `pcm16_to_wav` wraps the buffer in a WAV header in-process - no
ffmpeg, no temp files. The REST upload path passes already-containered audio
(webm/ogg/wav) straight through.

Everything here is async httpx; a single shared client is reused across turns so
we pay TLS setup once.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import wave
from dataclasses import dataclass

import httpx

from . import config

logger = logging.getLogger(__name__)


class STTError(RuntimeError):
    """Raised when transcription fails on every configured provider."""


class STTUnavailable(STTError):
    """Raised when no STT provider is configured at all (offline mode)."""


@dataclass(slots=True)
class STTResult:
    transcript: str
    language_code: str = ""
    language_probability: float = 0.0
    provider: str = ""
    model: str = ""
    stt_ms: float = 0.0
    audio_seconds: float = 0.0


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #


def pcm16_to_wav(
    pcm: bytes,
    *,
    sample_rate: int = config.AUDIO_SAMPLE_RATE,
    channels: int = config.AUDIO_CHANNELS,
    width: int = config.AUDIO_SAMPLE_WIDTH,
) -> bytes:
    """Wrap little-endian PCM16 samples in a minimal WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def pcm_duration_seconds(
    pcm: bytes,
    *,
    sample_rate: int = config.AUDIO_SAMPLE_RATE,
    channels: int = config.AUDIO_CHANNELS,
    width: int = config.AUDIO_SAMPLE_WIDTH,
) -> float:
    frames = len(pcm) / max(1, channels * width)
    return frames / max(1, sample_rate)


# --------------------------------------------------------------------------- #
# Shared async client
# --------------------------------------------------------------------------- #

_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:
                _client = httpx.AsyncClient(timeout=config.STT_TIMEOUT_S)
    return _client


async def aclose() -> None:
    """Close the shared client on server shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# --------------------------------------------------------------------------- #
# Provider implementations
# --------------------------------------------------------------------------- #


async def _transcribe_sarvam(audio: bytes, filename: str, mime: str, language: str | None) -> STTResult:
    translate = config.SARVAM_MODE.lower() == "translate" or config.SARVAM_MODEL.startswith("saaras")
    url = config.SARVAM_STT_URL
    if translate and not url.endswith("-translate"):
        url = url + "-translate"

    data: dict[str, str] = {"model": config.SARVAM_MODEL}
    if not translate:
        # STT (non-translate) needs a language code; "unknown" asks Sarvam to detect.
        data["language_code"] = language or config.SARVAM_LANGUAGE

    client = await _get_client()
    response = await client.post(
        url,
        headers={"api-subscription-key": config.SARVAM_API_KEY or ""},
        data=data,
        files={"file": (filename, audio, mime)},
    )
    response.raise_for_status()
    payload = response.json()
    transcript = payload.get("transcript") or payload.get("text") or ""
    return STTResult(
        transcript=transcript.strip(),
        language_code=payload.get("language_code", "") or "",
        provider="sarvam",
        model=config.SARVAM_MODEL,
    )


async def _transcribe_groq(audio: bytes, filename: str, mime: str, language: str | None) -> STTResult:
    data: dict[str, str] = {"model": config.GROQ_STT_MODEL, "response_format": "json"}
    if language and language not in ("unknown", "auto"):
        data["language"] = language
    client = await _get_client()
    response = await client.post(
        config.GROQ_STT_URL,
        headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
        data=data,
        files={"file": (filename, audio, mime)},
    )
    response.raise_for_status()
    payload = response.json()
    return STTResult(
        transcript=(payload.get("text") or "").strip(),
        language_code=payload.get("language", "") or "",
        provider="groq",
        model=config.GROQ_STT_MODEL,
    )


async def _transcribe_elevenlabs(audio: bytes, filename: str, mime: str, language: str | None) -> STTResult:
    data = {"model_id": config.ELEVENLABS_MODEL}
    client = await _get_client()
    response = await client.post(
        config.ELEVENLABS_STT_URL,
        headers={"xi-api-key": config.ELEVENLABS_API_KEY or ""},
        data=data,
        files={"file": (filename, audio, mime)},
    )
    response.raise_for_status()
    payload = response.json()
    return STTResult(
        transcript=(payload.get("text") or "").strip(),
        language_code=payload.get("language_code", "") or "",
        language_probability=float(payload.get("language_probability", 0.0) or 0.0),
        provider="elevenlabs",
        model=config.ELEVENLABS_MODEL,
    )


_PROVIDERS = {
    "sarvam": _transcribe_sarvam,
    "groq": _transcribe_groq,
    "elevenlabs": _transcribe_elevenlabs,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


async def transcribe(
    audio: bytes,
    *,
    is_pcm: bool = True,
    sample_rate: int = config.AUDIO_SAMPLE_RATE,
    content_type: str = "audio/wav",
    filename: str = "audio.wav",
    language: str | None = None,
) -> STTResult:
    """Transcribe audio, trying each configured provider until one succeeds.

    `is_pcm=True` means `audio` is raw PCM16 and gets WAV-wrapped here (the
    WebSocket path). `is_pcm=False` passes the bytes through unchanged (the REST
    upload path, where the browser already produced a container).
    """
    chain = config.stt_chain()
    if not chain:
        raise STTUnavailable(
            "No STT provider configured. Set SARVAM_API_KEY, GROQ_API_KEY, or "
            "ELEVENLABS_API_KEY in .env - or type your question instead of speaking."
        )

    if is_pcm:
        audio_seconds = pcm_duration_seconds(audio, sample_rate=sample_rate)
        audio = pcm16_to_wav(audio, sample_rate=sample_rate)
        content_type, filename = "audio/wav", "audio.wav"
    else:
        audio_seconds = 0.0

    if len(audio) > config.MAX_AUDIO_BYTES + 1024:
        raise STTError(f"Audio too large: {len(audio)} bytes > {config.MAX_AUDIO_BYTES}.")

    started = time.perf_counter()
    errors: list[str] = []
    for provider in chain:
        func = _PROVIDERS.get(provider)
        if func is None:
            continue
        try:
            result = await func(audio, filename, content_type, language)
            if not result.transcript:
                errors.append(f"{provider}: empty transcript")
                continue
            result.stt_ms = (time.perf_counter() - started) * 1000.0
            result.audio_seconds = round(audio_seconds, 3)
            logger.info(
                "stt %s ok in %.0fms: %r", provider, result.stt_ms, result.transcript[:60]
            )
            return result
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:200] if exc.response is not None else ""
            errors.append(f"{provider}: HTTP {exc.response.status_code} {body}")
            logger.warning("stt %s failed: %s", provider, errors[-1])
        except Exception as exc:  # network, timeout, JSON - try the next provider
            errors.append(f"{provider}: {type(exc).__name__}: {exc}")
            logger.warning("stt %s failed: %s", provider, errors[-1])

    raise STTError("All STT providers failed. " + " | ".join(errors))
