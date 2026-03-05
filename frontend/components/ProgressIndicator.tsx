"use client";

import { Scan, Database, BrainCircuit, Check } from "lucide-react";
import { useAppStore } from "@/lib/store";
import type { GenerationStage } from "@/lib/types";
import { cn } from "@/lib/utils";

const STAGES: {
  key: GenerationStage;
  label: string;
  icon: typeof Scan;
}[] = [
  { key: "classifying", label: "Classifying input", icon: Scan },
  { key: "gathering-context", label: "Gathering context", icon: Database },
  {
    key: "generating-questions",
    label: "Generating questions",
    icon: BrainCircuit,
  },
];

function stageIndex(stage: GenerationStage): number {
  const idx = STAGES.findIndex((s) => s.key === stage);
  return idx === -1 ? -1 : idx;
}

export function ProgressIndicator() {
  const generationStage = useAppStore((s) => s.generationStage);

  if (generationStage === "idle") return null;

  const activeIdx = stageIndex(generationStage);

  return (
    <div className="flex items-center justify-center gap-1 py-8">
      {STAGES.map((stage, i) => {
        const isActive = i === activeIdx;
        const isComplete = i < activeIdx;
        const Icon = isComplete ? Check : stage.icon;

        return (
          <div key={stage.key} className="flex items-center gap-1">
            <div
              className={cn(
                "flex items-center gap-2 rounded-full border px-3.5 py-1.5",
                "text-xs font-medium transition-all duration-300",
                isActive &&
                  "border-foreground/20 bg-foreground text-background",
                isComplete &&
                  "border-emerald-200 bg-emerald-50 text-emerald-700",
                !isActive &&
                  !isComplete &&
                  "border-border bg-white text-muted-foreground/40",
              )}
            >
              <Icon
                className={cn(
                  "h-3.5 w-3.5",
                  isActive && "animate-pulse",
                )}
              />
              {stage.label}
            </div>

            {i < STAGES.length - 1 && (
              <div
                className={cn(
                  "h-px w-6 transition-colors duration-300",
                  i < activeIdx ? "bg-emerald-300" : "bg-border",
                )}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
