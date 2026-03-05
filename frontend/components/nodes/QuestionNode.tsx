"use client";

import { Handle, Position, type NodeProps } from "@xyflow/react";
import { MessageCircleQuestion, MessageSquareText } from "lucide-react";
import { QUESTION_TYPE_CONFIG } from "@/lib/types";
import type { QuestionType } from "@/lib/types";
import { cn } from "@/lib/utils";

export interface QuestionNodeData {
  label: string;
  questionType: QuestionType;
  isExplored: boolean;
  nodeId: string;
  onExplore?: (nodeId: string) => void;
  [key: string]: unknown;
}

export function QuestionNode({ data }: NodeProps) {
  const { label, questionType, isExplored, nodeId, onExplore } =
    data as unknown as QuestionNodeData;

  const config = QUESTION_TYPE_CONFIG[questionType];

  return (
    <div
      className={cn(
        "w-[260px] rounded-lg border bg-white px-3.5 py-2.5 shadow-sm",
        "transition-all duration-200 hover:shadow-md",
        isExplored
          ? `${config.border} ${config.bg}`
          : "border-border hover:border-foreground/20",
      )}
    >
      {/* Input handle (top) */}
      <Handle
        type="target"
        position={Position.Top}
        className="!h-2 !w-2 !border-2 !border-white !bg-foreground/25"
      />

      {/* Type badge */}
      <div className="mb-1.5 flex items-center justify-between">
        <span
          className={cn(
            "inline-flex items-center rounded-full border px-2 py-0.5",
            "text-[10px] font-semibold tracking-wide uppercase",
            config.bg,
            config.border,
            config.color,
          )}
        >
          {config.label}
        </span>
        {isExplored && (
          <span className="text-[9px] font-semibold tracking-wider text-emerald-600 uppercase">
            Explored
          </span>
        )}
      </div>

      {/* Question text */}
      <p className="mb-2 line-clamp-3 text-xs leading-relaxed text-foreground/85">
        <MessageCircleQuestion className="mr-1 inline-block h-3 w-3 -translate-y-px text-muted-foreground/40" />
        {label}
      </p>

      {/* Explore button (only if not yet explored) */}
      {!isExplored && onExplore && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onExplore(nodeId);
          }}
          className={cn(
            "inline-flex w-full items-center justify-center gap-1.5",
            "rounded-md border border-border px-2.5 py-1",
            "text-[11px] font-medium text-muted-foreground",
            "transition-all duration-150 cursor-pointer",
            "hover:border-foreground/20 hover:bg-secondary/50 hover:text-foreground",
          )}
        >
          <MessageSquareText className="h-3 w-3" />
          Reflect & Explore
        </button>
      )}

      {/* Output handle (bottom — for connecting to reflection) */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-2 !w-2 !border-2 !border-white !bg-foreground/25"
      />
    </div>
  );
}
