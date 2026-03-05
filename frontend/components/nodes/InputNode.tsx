"use client";

import { Handle, Position, type NodeProps } from "@xyflow/react";
import { BrainCircuit, Tag, Route } from "lucide-react";
import type { InputType, PipelinePath } from "@/lib/types";
import { cn } from "@/lib/utils";

const INPUT_TYPE_COLORS: Record<InputType, string> = {
  argumentative: "bg-orange-100 text-orange-700 border-orange-200",
  factual: "bg-sky-100 text-sky-700 border-sky-200",
  opinion: "bg-purple-100 text-purple-700 border-purple-200",
  vague: "bg-stone-100 text-stone-600 border-stone-200",
};

const PIPELINE_LABELS: Record<PipelinePath, string> = {
  wikipedia: "Wikipedia",
  gemini: "AI Context",
  fallback: "Fallback",
};

export interface InputNodeData {
  label: string;
  inputType?: InputType;
  pipelinePath?: PipelinePath;
  isRoot?: boolean;
  [key: string]: unknown;
}

export function InputNode({ data }: NodeProps) {
  const { label, inputType, pipelinePath, isRoot } =
    data as unknown as InputNodeData;

  return (
    <div
      className={cn(
        "w-[280px] rounded-xl border-2 bg-white px-4 py-3 shadow-md",
        "transition-shadow duration-200 hover:shadow-lg",
        isRoot ? "border-foreground/25" : "border-foreground/15",
      )}
    >
      {/* Header */}
      <div className="mb-2 flex items-center gap-2">
        <BrainCircuit className="h-4 w-4 shrink-0 text-foreground/60" />
        <span className="text-[10px] font-semibold tracking-wider text-muted-foreground/60 uppercase">
          {isRoot ? "Original Input" : "Reflection"}
        </span>
      </div>

      {/* Text */}
      <p className="mb-2.5 line-clamp-3 text-sm leading-relaxed text-foreground/90">
        {label}
      </p>

      {/* Badges */}
      <div className="flex flex-wrap items-center gap-1.5">
        {inputType && (
          <span
            className={cn(
              "inline-flex items-center gap-1 rounded-full border px-2 py-0.5",
              "text-[10px] font-semibold capitalize",
              INPUT_TYPE_COLORS[inputType],
            )}
          >
            <Tag className="h-2.5 w-2.5" />
            {inputType}
          </span>
        )}
        {pipelinePath && (
          <span className="inline-flex items-center gap-1 rounded-full border border-border bg-secondary/50 px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
            <Route className="h-2.5 w-2.5" />
            {PIPELINE_LABELS[pipelinePath]}
          </span>
        )}
      </div>

      {/* Output handle (bottom) */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="h-2.5! w-2.5! border-2! border-white! bg-foreground/30!"
      />

      {/* Input handle (top — for non-root reflection-turned-input nodes) */}
      {!isRoot && (
        <Handle
          type="target"
          position={Position.Top}
          className="h-2.5! w-2.5! border-2! border-white! bg-foreground/30!"
        />
      )}
    </div>
  );
}
