"use client";

import {
  MessageCircleQuestion,
  MessageSquareText,
} from "lucide-react";
import { useAppStore } from "@/lib/store";
import { QUESTION_TYPE_CONFIG } from "@/lib/types";
import type { ExplorationNode, QuestionType } from "@/lib/types";
import { cn } from "@/lib/utils";
import { ReflectionPanel } from "./ReflectionPanel";

function QuestionCard({ node }: { node: ExplorationNode }) {
  const activeReflectionId = useAppStore((s) => s.activeReflectionId);
  const setActiveReflection = useAppStore((s) => s.setActiveReflection);
  const nodes = useAppStore((s) => s.nodes);

  const qType = node.metadata.questionType as QuestionType;
  const config = QUESTION_TYPE_CONFIG[qType];
  const isReflecting = activeReflectionId === node.id;
  const hasChildren = node.children.length > 0;

  // Check if this question has been explored (has a reflection child)
  const reflectionChild = node.children
    .map((id) => nodes[id])
    .find((n) => n?.type === "reflection");

  return (
    <div
      id={`question-card-${node.id}`}
      className={cn(
        "rounded-lg border bg-white transition-all duration-200",
        isReflecting && "border-foreground/20 shadow-sm",
        !isReflecting && "hover:shadow-sm",
      )}
    >
      <div className="p-4">
        {/* Type badge */}
        <div className="mb-2 flex items-center justify-between">
          <span
            className={cn(
              "inline-flex shrink-0 items-center rounded-full border px-2 py-0.5",
              "text-[11px] font-semibold tracking-wide uppercase",
              config.bg,
              config.border,
              config.color,
            )}
          >
            {config.label}
          </span>

          {hasChildren && (
            <span className="text-[10px] font-medium text-emerald-600">
              Explored
            </span>
          )}
        </div>

        {/* Question text */}
        <p className="mb-3 text-[15px] leading-relaxed text-foreground/90">
          <MessageCircleQuestion className="mr-1.5 inline-block h-4 w-4 -translate-y-px text-muted-foreground/40" />
          {node.text}
        </p>

        {/* Reflect button (only show if not already explored) */}
        {!reflectionChild && (
          <button
            onClick={() =>
              setActiveReflection(isReflecting ? null : node.id)
            }
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5",
              "text-xs font-medium transition-all duration-150 cursor-pointer",
              isReflecting
                ? "border-foreground/20 bg-foreground/5 text-foreground"
                : "border-border text-muted-foreground hover:border-foreground/20 hover:text-foreground",
            )}
          >
            <MessageSquareText className="h-3.5 w-3.5" />
            Reflect
          </button>
        )}

        {/* Show existing reflection summary if explored */}
        {reflectionChild && (
          <div className="rounded-md border border-dashed border-foreground/10 bg-secondary/20 p-2.5">
            <p className="text-xs leading-relaxed text-muted-foreground">
              <span className="font-medium text-foreground/70">
                Your reflection:
              </span>{" "}
              {reflectionChild.text.length > 150
                ? reflectionChild.text.slice(0, 150) + "..."
                : reflectionChild.text}
            </p>
          </div>
        )}
      </div>

      {/* Reflection panel (inline expand) */}
      {isReflecting && (
        <div className="border-t border-dashed border-foreground/10 px-4 pb-4">
          <ReflectionPanel questionId={node.id} />
        </div>
      )}
    </div>
  );
}

function FollowUpSkeleton({ depth }: { depth: number }) {
  return (
    <div className={cn("ml-4 border-l-2 border-foreground/5 pl-4")}>
      <div className="mb-3 pt-3 flex items-center gap-2">
        <div className="h-px flex-1 bg-border" />
        <span className="text-[10px] font-semibold tracking-wider text-muted-foreground/60 uppercase">
          Generating follow-ups...
        </span>
        <div className="h-px flex-1 bg-border" />
      </div>
      <div className="space-y-2.5">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className={cn(
              "rounded-lg border border-border/60 bg-white p-4 space-y-2.5",
              `stagger-${i}`,
            )}
            style={{ animationFillMode: "backwards" }}
          >
            <div className="skeleton-shimmer h-5 w-20 rounded-full" />
            <div className="skeleton-shimmer h-4 w-full rounded" />
            <div className="skeleton-shimmer h-4 w-3/4 rounded" />
          </div>
        ))}
      </div>
    </div>
  );
}

function BranchSection({
  parentNode,
  questions,
  depth,
}: {
  parentNode: ExplorationNode;
  questions: ExplorationNode[];
  depth: number;
}) {
  const nodes = useAppStore((s) => s.nodes);
  const generationStage = useAppStore((s) => s.generationStage);
  const isLoading = generationStage !== "idle";

  return (
    <div className={cn(depth > 0 && "ml-4 border-l-2 border-foreground/5 pl-4")}>
      {/* Branch header for depth > 0 */}
      {depth > 0 && (
        <div className="mb-3 pt-3 flex items-center gap-2">
          <div className="h-px flex-1 bg-border" />
          <span className="text-[10px] font-semibold tracking-wider text-muted-foreground/60 uppercase">
            Depth {depth} — Follow-up Questions
          </span>
          <div className="h-px flex-1 bg-border" />
        </div>
      )}

      {/* Question cards at this level */}
      <div className="space-y-2.5">
        {questions.map((q) => (
          <div key={q.id}>
            <QuestionCard node={q} />

            {/* Recursively render child branches or loading skeleton */}
            {q.children.map((childId) => {
              const childNode = nodes[childId];
              if (!childNode || childNode.type !== "reflection") return null;

              // Get the questions under this reflection
              const childQuestions = childNode.children
                .map((id) => nodes[id])
                .filter((n) => n?.type === "question");

              // Reflection exists but questions haven't arrived yet → skeleton
              if (childQuestions.length === 0) {
                if (isLoading) {
                  return (
                    <FollowUpSkeleton key={childId} depth={depth + 1} />
                  );
                }
                return null;
              }

              return (
                <BranchSection
                  key={childId}
                  parentNode={childNode}
                  questions={childQuestions}
                  depth={depth + 1}
                />
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

export function QuestionCards() {
  const nodes = useAppStore((s) => s.nodes);
  const rootId = useAppStore((s) => s.rootId);

  if (!rootId) return null;

  const rootNode = nodes[rootId];
  if (!rootNode) return null;

  // Get top-level questions (children of root)
  const topQuestions = rootNode.children
    .map((id) => nodes[id])
    .filter((n) => n?.type === "question");

  // Calculate total processing time across all levels
  const totalTimeMs = Object.values(nodes)
    .filter((n) => n.metadata.processingTimeMs)
    .reduce((sum, n) => sum + (n.metadata.processingTimeMs ?? 0), 0);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-wide text-foreground/80 uppercase">
          Socratic Questions
        </h2>
        {totalTimeMs > 0 && (
          <span className="font-mono text-[11px] text-muted-foreground">
            {(totalTimeMs / 1000).toFixed(1)}s
          </span>
        )}
      </div>

      <BranchSection
        parentNode={rootNode}
        questions={topQuestions}
        depth={0}
      />
    </div>
  );
}
