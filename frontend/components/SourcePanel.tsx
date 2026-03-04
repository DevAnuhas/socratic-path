"use client";

import { ExternalLink, BookOpen, Tag } from "lucide-react";
import { useAppStore } from "@/lib/store";
import { cn } from "@/lib/utils";

export function SourcePanel() {
  const { sources, keyphrases, selectedKeyphrase, selectKeyphrase } =
    useAppStore();

  return (
    <div className="space-y-5">
      {/* Keyphrases */}
      {keyphrases.length > 0 && (
        <div className="space-y-2.5">
          <h3 className="flex items-center gap-1.5 text-xs font-semibold tracking-wide text-foreground/70 uppercase">
            <Tag className="h-3.5 w-3.5" />
            Key Concepts
          </h3>
          <div className="flex flex-wrap gap-1.5">
            {keyphrases.map((kp) => (
              <button
                key={kp.text}
                onClick={() => selectKeyphrase(kp.text)}
                className={cn(
                  "rounded-md border px-2.5 py-1 font-mono text-[11px]",
                  "transition-all duration-150",
                  selectedKeyphrase === kp.text
                    ? "border-foreground/30 bg-foreground text-background"
                    : "border-border bg-white text-muted-foreground hover:border-foreground/20 hover:text-foreground"
                )}
              >
                {kp.text}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Wikipedia sources */}
      {sources.length > 0 && (
        <div className="space-y-2.5">
          <h3 className="flex items-center gap-1.5 text-xs font-semibold tracking-wide text-foreground/70 uppercase">
            <BookOpen className="h-3.5 w-3.5" />
            Sources
          </h3>
          <div className="space-y-2">
            {sources.map((source) => (
              <div
                key={source.keyphrase}
                className="rounded-lg border bg-white p-3"
              >
                <div className="mb-1.5 flex items-center justify-between">
                  <span className="text-xs font-semibold text-foreground/80">
                    {source.keyphrase}
                  </span>
                  {source.url && (
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-[10px] font-medium text-blue-600 transition-colors hover:text-blue-800"
                    >
                      Wikipedia
                      <ExternalLink className="h-2.5 w-2.5" />
                    </a>
                  )}
                </div>
                <p className="line-clamp-3 text-[12px] leading-relaxed text-muted-foreground">
                  {source.summary}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
