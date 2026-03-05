"use client";

import { useState } from "react";
import { Send, X } from "lucide-react";
import { useAppStore } from "@/lib/store";
import { cn } from "@/lib/utils";

interface ReflectionPanelProps {
  questionId: string;
  questionText: string;
}

const MAX_CHARS = 2000;

export function ReflectionPanel({
  questionId,
  questionText,
}: ReflectionPanelProps) {
  const [text, setText] = useState("");
  const submitReflection = useAppStore((s) => s.submitReflection);
  const setActiveReflection = useAppStore((s) => s.setActiveReflection);
  const generationStage = useAppStore((s) => s.generationStage);

  const isSubmitting = generationStage !== "idle";
  const canSubmit = text.trim().length > 0 && !isSubmitting;

  const handleSubmit = () => {
    if (!canSubmit) return;
    submitReflection(questionId, text.trim());
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && canSubmit) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="mt-3 rounded-lg border border-dashed border-foreground/15 bg-secondary/30 p-3">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-xs font-medium text-muted-foreground">
          How would you respond to this question?
        </p>
        <button
          onClick={() => setActiveReflection(null)}
          className="rounded p-0.5 text-muted-foreground/50 transition-colors hover:bg-secondary hover:text-foreground cursor-pointer"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
        onKeyDown={handleKeyDown}
        placeholder="Share your thoughts, reasoning, or counter-argument..."
        rows={3}
        disabled={isSubmitting}
        className={cn(
          "w-full resize-none rounded-md border bg-white px-3 py-2",
          "text-sm leading-relaxed placeholder:text-muted-foreground/40",
          "transition-colors duration-150",
          "focus:border-ring focus:outline-none focus:ring-2 focus:ring-ring/20",
          "disabled:opacity-50",
        )}
      />

      <div className="mt-2 flex items-center justify-between">
        <span className="font-mono text-[10px] text-muted-foreground/40">
          {text.length}/{MAX_CHARS} &middot; Cmd+Enter to submit
        </span>

        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5",
            "bg-foreground text-background text-xs font-medium",
            "transition-all duration-150",
            "hover:bg-foreground/90 cursor-pointer",
            "disabled:cursor-not-allowed disabled:opacity-40",
          )}
        >
          <Send className="h-3 w-3" />
          Generate Follow-ups
        </button>
      </div>
    </div>
  );
}
