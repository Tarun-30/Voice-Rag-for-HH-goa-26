"use client";

/**
 * useVoicePipeline - owns the /ws/audio WebSocket and assembles one "turn" of
 * pipeline events (stage timers, transcript, retrieved chunks, streamed answer
 * tokens, guardrail verdicts, final timings) into a single reactive object the
 * UI renders.
 *
 * Resilience: if the socket is not open, text queries fall back to POST
 * /api/query and audio falls back to POST /api/transcribe + POST /api/query, so
 * the app still works (just without token streaming) when the WS path is
 * blocked by a proxy.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { postQuery, transcribeBlob, wsUrl } from "./api";
import { pcm16ToWavBlob } from "./wav";
import type {
  GuardrailPhase,
  GuardrailVerdict,
  PipelineStage,
  PipelineTimings,
  RetrievalMode,
  RetrievedChunk,
  ServerEvent,
  Strategy,
  WSClientConfig,
} from "./types";

export interface StageState {
  status: "pending" | "start" | "done" | "error" | "skipped";
  ms: number;
  budget_ms: number;
  detail: string;
}

export type TurnStatus =
  | "idle"
  | "transcribing"
  | "running"
  | "streaming"
  | "done"
  | "error";

export interface TurnState {
  requestId: string;
  status: TurnStatus;
  transcript: string;
  transcriptProvider: string;
  transcriptLanguage: string;
  audioSeconds: number;
  stages: Record<PipelineStage, StageState>;
  chunks: RetrievedChunk[];
  strategy: Strategy;
  mode: RetrievalMode;
  answer: string;
  streaming: boolean;
  guardrails: Partial<Record<GuardrailPhase, GuardrailVerdict>>;
  refused: boolean;
  confidence: number;
  thoughtProcess: string;
  citedChunkIds: string[];
  timings: PipelineTimings | null;
  error: string | null;
  startedAt: number;
}

export interface PipelineSettings {
  strategy: Strategy;
  mode: RetrievalMode;
  topK: number;
  includeThoughtProcess: boolean;
  language: string | null;
}

export const STAGE_ORDER: PipelineStage[] = [
  "stt",
  "embed",
  "retrieval",
  "guardrail",
  "generation",
  "grounding",
];

function emptyStages(): Record<PipelineStage, StageState> {
  const stages = {} as Record<PipelineStage, StageState>;
  for (const stage of STAGE_ORDER) {
    stages[stage] = { status: "pending", ms: 0, budget_ms: 0, detail: "" };
  }
  return stages;
}

function newTurn(strategy: Strategy, mode: RetrievalMode, withStt: boolean): TurnState {
  const stages = emptyStages();
  if (!withStt) stages.stt.status = "skipped";
  return {
    requestId: "",
    status: withStt ? "transcribing" : "running",
    transcript: "",
    transcriptProvider: "",
    transcriptLanguage: "",
    audioSeconds: 0,
    stages,
    chunks: [],
    strategy,
    mode,
    answer: "",
    streaming: false,
    guardrails: {},
    refused: false,
    confidence: 0,
    thoughtProcess: "",
    citedChunkIds: [],
    timings: null,
    error: null,
    startedAt: typeof performance !== "undefined" ? performance.now() : 0,
  };
}

export interface UseVoicePipeline {
  connected: boolean;
  offlineMode: boolean;
  sampleRate: number;
  turn: TurnState | null;
  busy: boolean;
  sendText: (text: string, settings: PipelineSettings) => Promise<void>;
  sendAudio: (pcm: Int16Array, settings: PipelineSettings) => Promise<void>;
  reset: () => void;
}

export function useVoicePipeline(): UseVoicePipeline {
  const [connected, setConnected] = useState(false);
  const [offlineMode, setOfflineMode] = useState(false);
  const [sampleRate, setSampleRate] = useState(16000);
  const [turn, setTurn] = useState<TurnState | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const turnRef = useRef<TurnState | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closedByUs = useRef(false);

  const update = useCallback((mutate: (draft: TurnState) => void) => {
    setTurn((prev) => {
      if (!prev) return prev;
      const next = { ...prev, stages: { ...prev.stages } };
      mutate(next);
      turnRef.current = next;
      return next;
    });
  }, []);

  const handleEvent = useCallback(
    (event: ServerEvent) => {
      switch (event.type) {
        case "ready":
          setSampleRate(event.sample_rate);
          setOfflineMode(event.offline_mode);
          break;
        case "stage":
          update((draft) => {
            const stage = draft.stages[event.stage];
            stage.status = event.status;
            if (event.ms) stage.ms = event.ms;
            if (event.budget_ms) stage.budget_ms = event.budget_ms;
            if (event.detail) stage.detail = event.detail;
            if (event.stage === "generation" && event.status === "start") {
              draft.status = "streaming";
              draft.streaming = true;
            }
          });
          break;
        case "transcript":
          update((draft) => {
            draft.transcript = event.text;
            draft.transcriptProvider = event.provider;
            draft.transcriptLanguage = event.language_code;
            draft.audioSeconds = event.audio_seconds;
            draft.status = "running";
          });
          break;
        case "chunks":
          update((draft) => {
            draft.chunks = event.chunks;
            draft.strategy = event.strategy;
            draft.mode = event.mode;
          });
          break;
        case "guardrail":
          update((draft) => {
            draft.guardrails[event.phase] = event.verdict;
          });
          break;
        case "token":
          update((draft) => {
            if (event.is_first) draft.answer = "";
            draft.answer += event.text;
            draft.streaming = true;
          });
          break;
        case "done":
          update((draft) => {
            draft.answer = event.answer;
            draft.refused = event.refused;
            draft.confidence = event.confidence;
            draft.thoughtProcess = event.thought_process;
            draft.citedChunkIds = event.cited_chunk_ids;
            draft.guardrails.output = event.guardrail;
            draft.timings = event.timings;
            draft.streaming = false;
            draft.status = "done";
          });
          break;
        case "error":
          update((draft) => {
            draft.error = event.message;
            draft.status = "error";
            draft.streaming = false;
          });
          break;
      }
    },
    [update],
  );

  // --- socket lifecycle --------------------------------------------------- //
  useEffect(() => {
    closedByUs.current = false;

    const connect = () => {
      let socket: WebSocket;
      try {
        socket = new WebSocket(wsUrl("/ws/audio"));
      } catch {
        scheduleReconnect();
        return;
      }
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;

      socket.onopen = () => setConnected(true);
      socket.onclose = () => {
        setConnected(false);
        if (!closedByUs.current) scheduleReconnect();
      };
      socket.onerror = () => socket.close();
      socket.onmessage = (message) => {
        if (typeof message.data !== "string") return;
        try {
          handleEvent(JSON.parse(message.data) as ServerEvent);
        } catch {
          /* ignore malformed frame */
        }
      };
    };

    const scheduleReconnect = () => {
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      reconnectRef.current = setTimeout(connect, 1500);
    };

    connect();
    return () => {
      closedByUs.current = true;
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [handleEvent]);

  const wsSend = useCallback((frame: WSClientConfig) => {
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(frame));
      return true;
    }
    return false;
  }, []);

  const configFrame = (
    type: WSClientConfig["type"],
    settings: PipelineSettings,
    extra: Partial<WSClientConfig> = {},
  ): WSClientConfig => ({
    type,
    strategy: settings.strategy,
    mode: settings.mode,
    top_k: settings.topK,
    language: settings.language,
    include_thought_process: settings.includeThoughtProcess,
    ...extra,
  });

  // --- REST fallbacks ----------------------------------------------------- //
  const runRestQuery = useCallback(
    async (text: string, settings: PipelineSettings) => {
      try {
        const response = await postQuery({
          query: text,
          strategy: settings.strategy,
          mode: settings.mode,
          top_k: settings.topK,
          include_thought_process: settings.includeThoughtProcess,
        });
        update((draft) => {
          draft.transcript = response.transcript || text;
          draft.chunks = response.chunks;
          draft.answer = response.answer;
          draft.refused = response.refused;
          draft.confidence = response.confidence;
          draft.thoughtProcess = response.thought_process;
          draft.citedChunkIds = response.cited_chunk_ids;
          draft.guardrails.output = response.guardrail;
          draft.timings = response.timings;
          draft.strategy = response.strategy;
          draft.mode = response.mode;
          for (const stage of STAGE_ORDER) {
            if (draft.stages[stage].status === "pending") draft.stages[stage].status = "done";
          }
          draft.stages.embed.ms = response.timings.embed_ms;
          draft.stages.retrieval.ms = response.timings.retrieval_ms;
          draft.stages.guardrail.ms = response.timings.guardrail_ms;
          draft.stages.generation.ms = response.timings.ttft_ms;
          draft.status = "done";
        });
      } catch (error) {
        update((draft) => {
          draft.error = error instanceof Error ? error.message : String(error);
          draft.status = "error";
        });
      }
    },
    [update],
  );

  // --- public API --------------------------------------------------------- //
  const sendText = useCallback(
    async (text: string, settings: PipelineSettings) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      const fresh = newTurn(settings.strategy, settings.mode, false);
      fresh.transcript = trimmed;
      turnRef.current = fresh;
      setTurn(fresh);

      if (!wsSend(configFrame("text", settings, { text: trimmed }))) {
        await runRestQuery(trimmed, settings);
      }
    },
    [runRestQuery, wsSend],
  );

  const sendAudio = useCallback(
    async (pcm: Int16Array, settings: PipelineSettings) => {
      if (!pcm.length) return;
      const fresh = newTurn(settings.strategy, settings.mode, true);
      turnRef.current = fresh;
      setTurn(fresh);

      const socket = socketRef.current;
      if (socket && socket.readyState === WebSocket.OPEN) {
        // Stream raw PCM frames, then close the turn with an `end` control frame.
        const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
        const frameBytes = 32768;
        for (let offset = 0; offset < bytes.length; offset += frameBytes) {
          socket.send(bytes.slice(offset, offset + frameBytes));
        }
        wsSend(configFrame("end", settings, { sample_rate: sampleRate }));
        return;
      }

      // Fallback: WAV-wrap and use the REST transcribe + query endpoints.
      try {
        const wav = pcm16ToWavBlob(pcm, sampleRate);
        const transcription = await transcribeBlob(wav, "audio.wav");
        update((draft) => {
          draft.transcript = transcription.transcript;
          draft.transcriptProvider = transcription.provider;
          draft.transcriptLanguage = transcription.language_code;
          draft.audioSeconds = transcription.audio_seconds;
          draft.stages.stt.status = "done";
          draft.stages.stt.ms = transcription.stt_ms;
          draft.status = "running";
        });
        await runRestQuery(transcription.transcript, settings);
      } catch (error) {
        update((draft) => {
          draft.error = error instanceof Error ? error.message : String(error);
          draft.status = "error";
        });
      }
    },
    [runRestQuery, sampleRate, update, wsSend],
  );

  const reset = useCallback(() => {
    turnRef.current = null;
    setTurn(null);
  }, []);

  const busy = Boolean(
    turn && ["transcribing", "running", "streaming"].includes(turn.status),
  );

  return {
    connected,
    offlineMode,
    sampleRate,
    turn,
    busy,
    sendText,
    sendAudio,
    reset,
  };
}
