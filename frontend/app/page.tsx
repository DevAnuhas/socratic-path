"use client";

import { BrainCircuit, HelpCircle } from "lucide-react";
import { useAppStore } from "@/lib/store";
import { InputPanel } from "@/components/InputPanel";
import { QuestionCards } from "@/components/QuestionCards";
import { SourcePanel } from "@/components/SourcePanel";
import { ProgressIndicator } from "@/components/ProgressIndicator";

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

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
      <strong className="font-semibold">Error:</strong> {message}
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

  const isLoading = generationStage !== "idle";
  const hasGenerated = rootId !== null;

  // Find the latest depth nudge from any node
  const latestNudge = Object.values(nodes)
    .filter((n) => n.metadata.depthNudge)
    .sort((a, b) => b.depth - a.depth)[0]?.metadata.depthNudge;

  return (
    <div className="mx-auto min-h-screen max-w-5xl px-6 py-8">
      {/* Header */}
      <header className="mb-8">
        <div className="flex items-center gap-2.5">
          <BrainCircuit className="h-6 w-6 text-foreground/80" />
          <h1 className="text-xl font-bold tracking-tight text-foreground">
            SocraticPath
          </h1>
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          AI-powered Socratic exploration using a fine-tuned T5 model
        </p>
      </header>

      {/* Input */}
      <section className="mb-8">
        <InputPanel />
      </section>

      {/* Error */}
      {error && (
        <section className="mb-6">
          <ErrorBanner message={error} />
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
        {!isLoading && hasGenerated && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_280px]">
              <QuestionCards />
              <SourcePanel />
            </div>
          </div>
        )}

        {!isLoading && !hasGenerated && !error && <EmptyState />}
      </section>
    </div>
  );
}
