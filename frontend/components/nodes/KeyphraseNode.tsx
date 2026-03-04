"use client";

import { Handle, Position, type NodeProps } from "@xyflow/react";
import { Search, Compass } from "lucide-react";
import { cn } from "@/lib/utils";

export interface KeyphraseNodeData {
  label: string;
  score: number;
  isSelected: boolean;
  hasContext: boolean;
  onSelect: (keyphrase: string) => void;
  onExplore: (keyphrase: string) => void;
  [key: string]: unknown;
}

export function KeyphraseNode({ data }: NodeProps) {
  const {
    label,
    score,
    isSelected,
    hasContext,
    onSelect,
    onExplore,
  } = data as unknown as KeyphraseNodeData;

  return (
    <div
      className={cn(
        "group relative rounded-lg border bg-white px-3.5 py-2 shadow-sm",
        "transition-all duration-150 cursor-pointer",
        isSelected
          ? "border-foreground/40 shadow-md ring-2 ring-foreground/10"
          : "border-border hover:border-foreground/20 hover:shadow-md"
      )}
      onClick={() => onSelect(label)}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="!h-2 !w-2 !border-none !bg-foreground/20"
      />
      <Handle
        type="target"
        position={Position.Top}
        id="top"
        className="!h-2 !w-2 !border-none !bg-foreground/20"
      />
      <Handle
        type="target"
        position={Position.Right}
        id="right"
        className="!h-2 !w-2 !border-none !bg-foreground/20"
      />
      <Handle
        type="target"
        position={Position.Bottom}
        id="bottom"
        className="!h-2 !w-2 !border-none !bg-foreground/20"
      />

      <div className="flex items-center gap-2">
        {hasContext ? (
          <Search className="h-3.5 w-3.5 shrink-0 text-blue-500" />
        ) : (
          <div className="h-2 w-2 shrink-0 rounded-full bg-muted-foreground/30" />
        )}
        <span className="text-xs font-medium text-foreground/90">
          {label}
        </span>
      </div>

      {/* Score bar */}
      <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className="h-full rounded-full bg-foreground/20 transition-all"
          style={{ width: `${Math.max(score * 100, 10)}%` }}
        />
      </div>

      {/* Explore button — appears on hover */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onExplore(label);
        }}
        className={cn(
          "absolute -right-1 -top-1 flex h-5 w-5 items-center justify-center",
          "rounded-full border bg-white text-muted-foreground shadow-sm",
          "opacity-0 transition-all group-hover:opacity-100",
          "hover:bg-foreground hover:text-background"
        )}
        title={`Explore "${label}"`}
      >
        <Compass className="h-3 w-3" />
      </button>
    </div>
  );
}
