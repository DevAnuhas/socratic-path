"use client";

import { Sparkles, RotateCcw, LoaderCircle, Route, Tag } from "lucide-react";
import { useAppStore } from "@/lib/store";
import {
  ALL_QUESTION_TYPES,
  QUESTION_TYPE_CONFIG,
  type QuestionType,
  type InputType,
} from "@/lib/types";
import { cn } from "@/lib/utils";

const INPUT_TYPE_CONFIG: Record<
  InputType,
  { label: string; color: string; bg: string; border: string }
> = {
  argumentative: {
    label: "Argumentative",
    color: "text-orange-700",
    bg: "bg-orange-50",
    border: "border-orange-200",
  },
  factual: {
    label: "Factual",
    color: "text-sky-700",
    bg: "bg-sky-50",
    border: "border-sky-200",
  },
  opinion: {
    label: "Opinion",
    color: "text-purple-700",
    bg: "bg-purple-50",
    border: "border-purple-200",
  },
  vague: {
    label: "Vague",
    color: "text-stone-600",
    bg: "bg-stone-50",
    border: "border-stone-200",
  },
};

const PIPELINE_CONFIG: Record<string, { label: string; icon: string }> = {
  wikipedia: { label: "Wikipedia", icon: "📚" },
  gemini: { label: "AI-Generated", icon: "✨" },
  fallback: { label: "Fallback", icon: "🔄" },
};

export function InputPanel() {
  const topic = useAppStore((s) => s.topic);
  const setTopic = useAppStore((s) => s.setTopic);
  const selectedTypes = useAppStore((s) => s.selectedTypes);
  const toggleType = useAppStore((s) => s.toggleType);
  const submitInitialInput = useAppStore((s) => s.submitInitialInput);
  const reset = useAppStore((s) => s.reset);
  const generationStage = useAppStore((s) => s.generationStage);
  const rootId = useAppStore((s) => s.rootId);
  const lastClassification = useAppStore((s) => s.lastClassification);
  const nodes = useAppStore((s) => s.nodes);

  const isLoading = generationStage !== "idle";
  const hasGenerated = rootId !== null;

  // Get pipeline path from root node
  const pipelinePath = rootId
    ? nodes[rootId]?.metadata.pipelinePath
    : undefined;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    submitInitialInput();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      submitInitialInput();
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Topic input */}
      <div className="relative">
        <textarea
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter a topic, opinion, or argument to explore with Socratic questioning..."
          rows={3}
          disabled={isLoading}
          className={cn(
            "w-full resize-none rounded-lg border bg-white px-4 py-3",
            "text-[15px] leading-relaxed placeholder:text-muted-foreground/50",
            "transition-colors duration-150",
            "focus:border-ring focus:outline-none focus:ring-2 focus:ring-ring/20",
            "disabled:opacity-60",
          )}
        />
        <span className="pointer-events-none absolute right-3 bottom-2.5 font-mono text-[11px] text-muted-foreground/40">
          {topic.length > 0
            ? `${topic.length} chars`
            : "Cmd+Enter to submit"}
        </span>
      </div>

      {/* Classification indicator (shown after generation) */}
      {hasGenerated && lastClassification && (
        <div className="flex flex-wrap items-center gap-2">
          {/* Input type badge */}
          {(() => {
            const cfg =
              INPUT_TYPE_CONFIG[lastClassification.input_type];
            return (
              <span
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5",
                  "text-[11px] font-semibold",
                  cfg.bg,
                  cfg.border,
                  cfg.color,
                )}
              >
                <Tag className="h-3 w-3" />
                {cfg.label} input
              </span>
            );
          })()}

          {/* Pipeline path */}
          {pipelinePath && PIPELINE_CONFIG[pipelinePath] && (
            <span className="inline-flex items-center gap-1 rounded-full border border-border bg-white px-2.5 py-0.5 text-[11px] font-medium text-muted-foreground">
              <Route className="h-3 w-3" />
              Context: {PIPELINE_CONFIG[pipelinePath].label}
            </span>
          )}

          {/* Confidence */}
          <span className="font-mono text-[10px] text-muted-foreground/50">
            {(lastClassification.confidence * 100).toFixed(0)}% confidence
          </span>
        </div>
      )}

      {/* Question type chips */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="mr-1 text-xs font-medium tracking-wide text-muted-foreground uppercase">
          Types
        </span>
        {ALL_QUESTION_TYPES.map((type) => {
          const config = QUESTION_TYPE_CONFIG[type];
          const isSelected = selectedTypes.includes(type);
          return (
            <button
              key={type}
              type="button"
              onClick={() => toggleType(type)}
              className={cn(
                "rounded-full border px-3 py-1 text-xs font-medium cursor-pointer",
                "transition-all duration-150",
                isSelected
                  ? `${config.bg} ${config.border} ${config.color}`
                  : "border-border bg-transparent text-muted-foreground/50 hover:border-muted-foreground/30",
              )}
            >
              {config.label}
            </button>
          );
        })}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={isLoading || !topic.trim()}
          className={cn(
            "inline-flex items-center gap-2 rounded-lg px-5 py-2.5",
            "bg-foreground text-background text-sm font-medium",
            "transition-all duration-150",
            "hover:bg-foreground/90 cursor-pointer",
            "disabled:cursor-not-allowed disabled:opacity-40",
          )}
        >
          {isLoading ? (
            <>
              <LoaderCircle className="h-4 w-4 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4" />
              {hasGenerated ? "New Exploration" : "Begin Exploration"}
            </>
          )}
        </button>

        {hasGenerated && (
          <button
            type="button"
            onClick={reset}
            className="inline-flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Reset
          </button>
        )}
      </div>
    </form>
  );
}
