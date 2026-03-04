"use client";

import { Handle, Position, type NodeProps } from "@xyflow/react";
import { BrainCircuit } from "lucide-react";

export interface TopicNodeData {
  label: string;
  [key: string]: unknown;
}

export function TopicNode({ data }: NodeProps) {
  const label = data.label as string;
  return (
    <div className="flex items-center gap-2.5 rounded-xl border-2 border-foreground/20 bg-white px-5 py-3 shadow-md">
      <BrainCircuit className="h-5 w-5 shrink-0 text-foreground/60" />
      <span className="max-w-[200px] truncate text-sm font-semibold text-foreground">
        {label}
      </span>
      <Handle
        type="source"
        position={Position.Right}
        className="!h-2 !w-2 !border-none !bg-foreground/30"
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="bottom"
        className="!h-2 !w-2 !border-none !bg-foreground/30"
      />
      <Handle
        type="source"
        position={Position.Left}
        id="left"
        className="!h-2 !w-2 !border-none !bg-foreground/30"
      />
      <Handle
        type="source"
        position={Position.Top}
        id="top"
        className="!h-2 !w-2 !border-none !bg-foreground/30"
      />
    </div>
  );
}
