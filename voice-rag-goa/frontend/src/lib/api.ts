/**
 * Backend client: REST helpers + the audio WebSocket URL.
 *
 * The base origin comes from NEXT_PUBLIC_API_BASE (default localhost:8000).
 * The WebSocket URL is derived from it so http->ws and https->wss stay in sync
 * with whatever the browser is actually talking to.
 */

import type {
  AnalyticsResponse,
  HealthResponse,
  QueryRequest,
  QueryResponse,
  TranscriptionResponse,
} from "./types";

export const API_BASE: string = (
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"
).replace(/\/+$/, "");

export function wsUrl(path: string): string {
  const base = API_BASE.replace(/^http/, "ws");
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* non-JSON error body; keep the status line */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export function getAnalytics(): Promise<AnalyticsResponse> {
  return request<AnalyticsResponse>("/api/analytics");
}

export function postQuery(body: QueryRequest): Promise<QueryResponse> {
  return request<QueryResponse>("/api/query", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function triggerBenchmark(body?: {
  queries?: number;
  concurrency?: number;
  include_llm?: boolean;
}): Promise<{ status: string; queries?: number; detail?: string }> {
  return request("/api/benchmark", {
    method: "POST",
    body: JSON.stringify(body ?? {}),
  });
}

/** Upload an audio blob (webm/ogg/wav) for one-shot transcription. */
export async function transcribeBlob(
  blob: Blob,
  filename = "audio.webm",
): Promise<TranscriptionResponse> {
  const form = new FormData();
  form.append("file", blob, filename);
  const response = await fetch(`${API_BASE}/api/transcribe`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      detail = (await response.json())?.detail ?? detail;
    } catch {
      /* keep status */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as TranscriptionResponse;
}
