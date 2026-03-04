"use client";

import { MessageCircleQuestion } from "lucide-react";
import { useAppStore } from "@/lib/store";
import { QUESTION_TYPE_CONFIG } from "@/lib/types";
import { cn } from "@/lib/utils";

export function QuestionCards() {
  const { questions, selectedKeyphrase, processingTimeMs } = useAppStore();

  const filtered = selectedKeyphrase
    ? questions.filter((q) =>
        q.related_keyphrases.some(
          (kp) => kp.toLowerCase() === selectedKeyphrase.toLowerCase()
        )
      )
    : questions;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-wide text-foreground/80 uppercase">
          Socratic Questions
        </h2>
        {processingTimeMs !== null && (
          <span className="font-mono text-[11px] text-muted-foreground">
            {(processingTimeMs / 1000).toFixed(1)}s
          </span>
        )}
      </div>

      {selectedKeyphrase && (
        <p className="text-xs text-muted-foreground">
          Showing questions related to{" "}
          <span className="font-medium text-foreground">
            &ldquo;{selectedKeyphrase}&rdquo;
          </span>
          {" "}
          &middot;{" "}
          <button
            onClick={() => useAppStore.getState().selectKeyphrase(null)}
            className="underline transition-colors hover:text-foreground"
          >
            show all
          </button>
        </p>
      )}

      <div className="space-y-2.5">
        {filtered.map((question, i) => {
          const config = QUESTION_TYPE_CONFIG[question.type];
          return (
            <div
              key={question.id}
              className={cn(
                "group rounded-lg border bg-white p-4",
                "transition-all duration-200",
                "hover:shadow-sm",
                `stagger-${i + 1}`
              )}
              style={{ animationFillMode: "backwards" }}
            >
              <div className="mb-2 flex items-start gap-2.5">
                <span
                  className={cn(
                    "inline-flex shrink-0 items-center rounded-full border px-2 py-0.5",
                    "text-[11px] font-semibold tracking-wide uppercase",
                    config.bg,
                    config.border,
                    config.color
                  )}
                >
                  {config.label}
                </span>
              </div>

              <p className="leading-relaxed text-[15px] text-foreground/90">
                <MessageCircleQuestion className="mr-1.5 inline-block h-4 w-4 -translate-y-px text-muted-foreground/40" />
                {question.text}
              </p>
            </div>
          );
        })}
      </div>

      {filtered.length === 0 && questions.length > 0 && (
        <p className="py-6 text-center text-sm text-muted-foreground">
          No questions match the selected keyphrase.
        </p>
      )}
    </div>
  );
}
