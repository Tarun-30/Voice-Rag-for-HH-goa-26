"use client";

/**
 * GroundingBadge - the guardrail verdict at a glance.
 *
 * Collapses the three guardrail phases (input / context / output) into a single
 * status: a blocked input or refused context wins over an output verdict, so the
 * badge always shows the *reason a turn was stopped* when one was. When the
 * answer is allowed it surfaces the grounding score (share of answer sentences
 * traceable to retrieved context).
 */

import {
  Shield,
  ShieldAlert,
  ShieldCheck,
  ShieldX,
  Loader2,
} from "lucide-react";

import type { GuardrailPhase, GuardrailVerdict } from "@/lib/types";
import type { TurnStatus } from "@/lib/useVoicePipeline";

interface GroundingBadgeProps {
  guardrails: Partial<Record<GuardrailPhase, GuardrailVerdict>>;
  refused: boolean;
  status: TurnStatus;
  confidence: number;
}

const CATEGORY_LABEL: Record<string, string> = {
  ok: "OK",
  empty_query: "Empty query",
  too_long: "Query too long",
  prompt_injection: "Prompt injection blocked",
  unsafe_content: "Unsafe content",
  off_topic: "Off-topic",
  insufficient_context: "Insufficient context",
  ungrounded_answer: "Ungrounded answer",
  gibberish: "Unintelligible query",
};

type Tone = "neutral" | "good" | "warn" | "bad" | "busy";

const TONE_STYLES: Record<Tone, string> = {
  neutral: "border-cream/20 bg-cream/5 text-cream/70",
  good: "border-emerald-300/40 bg-emerald-400/10 text-emerald-200",
  warn: "border-gold/45 bg-gold/10 text-gold",
  bad: "border-pink/50 bg-pink/12 text-pink",
  busy: "border-cream/25 bg-cream/8 text-cream/80",
};

function pct(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${Math.round(value * 100)}%`;
}

export default function GroundingBadge({
  guardrails,
  refused,
  status,
  confidence,
}: GroundingBadgeProps) {
  const input = guardrails.input;
  const context = guardrails.context;
  const output = guardrails.output;

  // Find the phase that stopped the turn, if any.
  const blocker =
    input && input.decision !== "allow"
      ? { phase: "input" as const, verdict: input }
      : context && context.decision !== "allow"
        ? { phase: "context" as const, verdict: context }
        : output && output.decision !== "allow"
          ? { phase: "output" as const, verdict: output }
          : null;

  let tone: Tone = "neutral";
  let Icon = Shield;
  let title = "Awaiting query";
  let subtitle = "Guardrails standing by";
  let scoreBar: { label: string; value: number; tone: Tone } | null = null;

  if (status === "idle" && !output && !blocker) {
    // defaults above
  } else if (status === "transcribing" || status === "running" || status === "streaming") {
    if (blocker) {
      tone = blocker.verdict.decision === "block" ? "bad" : "warn";
      Icon = blocker.verdict.decision === "block" ? ShieldX : ShieldAlert;
      title = blocker.verdict.decision === "block" ? "Blocked" : "Refused";
      subtitle = CATEGORY_LABEL[blocker.verdict.category] ?? blocker.verdict.reason;
    } else {
      tone = "busy";
      Icon = Loader2;
      title = "Verifying";
      subtitle = "Running guardrail checks";
    }
  } else if (blocker || refused) {
    const verdict = blocker?.verdict ?? output;
    const decision = verdict?.decision ?? "refuse";
    tone = decision === "block" ? "bad" : "warn";
    Icon = decision === "block" ? ShieldX : ShieldAlert;
    title = decision === "block" ? "Blocked" : "Refused";
    subtitle = verdict
      ? (CATEGORY_LABEL[verdict.category] ?? verdict.reason)
      : "Not grounded in dataset";
    if (verdict?.grounding_score !== null && verdict?.grounding_score !== undefined) {
      scoreBar = { label: "Grounding", value: verdict.grounding_score, tone };
    } else if (verdict?.context_sufficiency !== null && verdict?.context_sufficiency !== undefined) {
      scoreBar = { label: "Context", value: verdict.context_sufficiency, tone };
    }
  } else if (output && output.decision === "allow") {
    tone = "good";
    Icon = ShieldCheck;
    title = "Grounded";
    subtitle =
      output.total_sentences > 0
        ? `${output.grounded_sentences}/${output.total_sentences} sentences supported`
        : "Answer verified against context";
    if (output.grounding_score !== null && output.grounding_score !== undefined) {
      scoreBar = { label: "Grounding", value: output.grounding_score, tone: "good" };
    }
  }

  const spinning = Icon === Loader2;

  return (
    <div
      className={`glass-tight flex flex-col gap-3 p-4 transition-colors ${TONE_STYLES[tone]}`}
      role="status"
      aria-live="polite"
    >
      <div className="flex items-center gap-3">
        <span className="grid h-10 w-10 place-items-center rounded-full border border-current/30 bg-current/10">
          <Icon className={`h-5 w-5 ${spinning ? "animate-spin" : ""}`} aria-hidden />
        </span>
        <div className="min-w-0">
          <p className="text-sm font-bold uppercase tracking-wide">{title}</p>
          <p className="truncate text-xs opacity-80" title={subtitle}>
            {subtitle}
          </p>
        </div>
      </div>

      {scoreBar && (
        <div className="space-y-1">
          <div className="flex justify-between text-[11px] font-semibold uppercase tracking-wide opacity-80">
            <span>{scoreBar.label}</span>
            <span>{pct(scoreBar.value)}</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-cream/10">
            <div
              className="h-full rounded-full bg-current transition-[width] duration-500"
              style={{ width: `${Math.round(Math.max(0, Math.min(1, scoreBar.value)) * 100)}%` }}
            />
          </div>
        </div>
      )}

      {status === "done" && !refused && !blocker && (
        <div className="flex items-center justify-between text-[11px] font-semibold uppercase tracking-wide opacity-80">
          <span>Confidence</span>
          <span>{pct(confidence)}</span>
        </div>
      )}
    </div>
  );
}
