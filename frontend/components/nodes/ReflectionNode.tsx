"use client";

import { Handle, Position, type NodeProps } from "@xyflow/react";
import { MessageSquareText, GitBranch } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ReflectionNodeData {
  label: string;
  childCount: number;
  [key: string]: unknown;
}

export function ReflectionNode({ data }: NodeProps) {
  const { label, childCount } = data as unknown as ReflectionNodeData;

  return (
    <div
      className={cn(
        "w-[240px] rounded-lg border border-dashed border-foreground/15",
        "bg-secondary/40 px-3.5 py-2.5 shadow-sm",
        "transition-all duration-200 hover:shadow-md",
      )}
    >
      {/* Input handle (top) */}
      <Handle
        type="target"
        position={Position.Top}
        className="!h-2 !w-2 !border-2 !border-white !bg-foreground/25"
      />

      {/* Header */}
      <div className="mb-1.5 flex items-center gap-1.5">
        <MessageSquareText className="h-3 w-3 shrink-0 text-foreground/40" />
        <span className="text-[10px] font-semibold tracking-wider text-muted-foreground/60 uppercase">
          Reflection
        </span>
      </div>

      {/* Reflection text */}
      <p className="mb-2 line-clamp-3 text-xs leading-relaxed text-foreground/75 italic">
        &ldquo;{label}&rdquo;
      </p>

      {/* Branch count */}
      {childCount > 0 && (
        <div className="flex items-center gap-1 text-[10px] font-medium text-muted-foreground/60">
          <GitBranch className="h-2.5 w-2.5" />
          Led to {childCount} question{childCount !== 1 ? "s" : ""}
        </div>
      )}

      {/* Output handle (bottom) */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-2 !w-2 !border-2 !border-white !bg-foreground/25"
      />
    </div>
  );
}
