"use client";

import { useState } from "react";
import { BrainCircuit, HelpCircle, Share2, RefreshCw, X } from "lucide-react";
import { useAppStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { InputPanel } from "@/components/InputPanel";
import { QuestionCards } from "@/components/QuestionCards";
import { SourcePanel } from "@/components/SourcePanel";
import { ProgressIndicator } from "@/components/ProgressIndicator";
import { ExplorationGraph } from "@/components/ExplorationGraph";
import { ExportDialog } from "@/components/ExportDialog";

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-secondary">
        <HelpCircle className="h-7 w-7 text-muted-foreground/60" />
      </div>
      <h2 className="mb-1.5 text-lg font-semibold text-foreground/80">
        Enter a topic to begin
      </h2>
      <p className="max-w-sm text-sm leading-relaxed text-muted-foreground">
        Type a topic, opinion, or argument above and SocraticPath will generate
        probing questions across five categories of Socratic inquiry.
      </p>
      <div className="mt-6 flex flex-wrap justify-center gap-2">
        {[
          "I think AI will replace most creative jobs",
          "Climate change effects on agriculture",
          "Social media does more harm than good",
        ].map((example) => (
          <button
            key={example}
            onClick={() => {
              useAppStore.getState().setTopic(example);
            }}
            className="rounded-full border border-border bg-white px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-foreground/20 hover:text-foreground"
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}

function ErrorBanner({
  message,
  canRetry,
}: {
  message: string;
  canRetry: boolean;
}) {
  const retryLast = useAppStore((s) => s.retryLast);
  const dismissError = useAppStore((s) => s.dismissError);

  return (
    <div className="flex items-start justify-between gap-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
      <div className="flex-1">
        <strong className="font-semibold">Error:</strong> {message}
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {canRetry && (
          <button
            onClick={retryLast}
            className="inline-flex items-center gap-1 rounded-md border border-red-200 bg-white px-2.5 py-1 text-xs font-medium text-red-700 transition-colors hover:bg-red-100 cursor-pointer"
          >
            <RefreshCw className="h-3 w-3" />
            Retry
          </button>
        )}
        <button
          onClick={dismissError}
          className="rounded p-0.5 text-red-400 transition-colors hover:text-red-700 cursor-pointer"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
}

function DepthNudge({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
      <strong className="font-semibold">Suggestion:</strong> {message}
    </div>
  );
}

export default function Home() {
  const generationStage = useAppStore((s) => s.generationStage);
  const rootId = useAppStore((s) => s.rootId);
  const error = useAppStore((s) => s.error);
  const nodes = useAppStore((s) => s.nodes);
  const lastFailedAction = useAppStore((s) => s.lastFailedAction);
  const [exportOpen, setExportOpen] = useState(false);

  const isLoading = generationStage !== "idle";
  const hasGenerated = rootId !== null;

  // Find the latest depth nudge from any node
  const latestNudge = Object.values(nodes)
    .filter((n) => n.metadata.depthNudge)
    .sort((a, b) => b.depth - a.depth)[0]?.metadata.depthNudge;

  return (
    <div className="mx-auto min-h-screen max-w-5xl px-6 py-8">
      {/* Header */}
      <header className="mb-8 flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2.5">
            <BrainCircuit className="h-6 w-6 text-foreground/80" />
            <h1 className="text-xl font-bold tracking-tight text-foreground">
              SocraticPath
            </h1>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            AI-powered Socratic exploration using a fine-tuned T5 model
          </p>
        </div>

        {hasGenerated && (
          <button
            onClick={() => setExportOpen(true)}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5",
              "text-xs font-medium text-muted-foreground",
              "transition-all cursor-pointer",
              "hover:border-foreground/20 hover:text-foreground",
            )}
          >
            <Share2 className="h-3.5 w-3.5" />
            Export
          </button>
        )}
      </header>

      {/* Input */}
      <section className="mb-8">
        <InputPanel />
      </section>

      {/* Error */}
      {error && (
        <section className="mb-6">
          <ErrorBanner message={error} canRetry={lastFailedAction !== null} />
        </section>
      )}

      {/* Progress indicator */}
      {isLoading && <ProgressIndicator />}

      {/* Depth nudge */}
      {!isLoading && latestNudge && (
        <section className="mb-6">
          <DepthNudge message={latestNudge} />
        </section>
      )}

      {/* Results */}
      <section>
        {hasGenerated && (
          <div className="space-y-6">
            <ExplorationGraph />
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_280px]">
              <QuestionCards />
              <SourcePanel />
            </div>
          </div>
        )}

        {!isLoading && !hasGenerated && !error && <EmptyState />}
      </section>

      {/* Export dialog */}
      <ExportDialog open={exportOpen} onClose={() => setExportOpen(false)} />
    </div>
  );
}
