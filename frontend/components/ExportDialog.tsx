"use client";

import { useState, useCallback } from "react";
import {
  X,
  Copy,
  Download,
  FileText,
  GitBranch,
  Image,
  Check,
} from "lucide-react";
import { toPng, toSvg } from "html-to-image";
import { useAppStore } from "@/lib/store";
import { exportMarkdown, exportMermaid } from "@/lib/export";
import { cn } from "@/lib/utils";

type ExportTab = "markdown" | "mermaid" | "image";

interface ExportDialogProps {
  open: boolean;
  onClose: () => void;
}

export function ExportDialog({ open, onClose }: ExportDialogProps) {
  const nodes = useAppStore((s) => s.nodes);
  const rootId = useAppStore((s) => s.rootId);
  const [activeTab, setActiveTab] = useState<ExportTab>("markdown");
  const [copied, setCopied] = useState(false);
  const [imageStatus, setImageStatus] = useState<
    "idle" | "generating" | "done" | "error"
  >("idle");

  const markdown = exportMarkdown(nodes, rootId);
  const mermaid = exportMermaid(nodes, rootId);

  const handleCopy = useCallback(
    async (text: string) => {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    },
    [],
  );

  const handleDownloadText = useCallback(
    (content: string, filename: string) => {
      const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    },
    [],
  );

  const handleDownloadImage = useCallback(
    async (format: "png" | "svg") => {
      const graphEl = document.querySelector(
        ".react-flow",
      ) as HTMLElement | null;
      if (!graphEl) return;

      setImageStatus("generating");
      try {
        const exportFn = format === "png" ? toPng : toSvg;
        const dataUrl = await exportFn(graphEl, {
          backgroundColor: "#fdfcfb",
          quality: 1,
          pixelRatio: 2,
        });

        const a = document.createElement("a");
        a.href = dataUrl;
        a.download = `socratic-path-exploration.${format}`;
        a.click();
        setImageStatus("done");
        setTimeout(() => setImageStatus("idle"), 2000);
      } catch {
        setImageStatus("error");
        setTimeout(() => setImageStatus("idle"), 3000);
      }
    },
    [],
  );

  if (!open) return null;

  const tabs: { key: ExportTab; label: string; icon: typeof FileText }[] = [
    { key: "markdown", label: "Markdown", icon: FileText },
    { key: "mermaid", label: "Mermaid", icon: GitBranch },
    { key: "image", label: "Image", icon: Image },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Dialog */}
      <div className="relative z-10 mx-4 flex max-h-[80vh] w-full max-w-2xl flex-col rounded-xl border bg-white shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b px-5 py-3.5">
          <h2 className="text-sm font-semibold text-foreground">
            Export Exploration
          </h2>
          <button
            onClick={onClose}
            className="rounded p-1 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground cursor-pointer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 border-b px-5 pt-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.key;
            return (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-t-md px-3 py-2",
                  "text-xs font-medium transition-colors cursor-pointer",
                  isActive
                    ? "border-b-2 border-foreground text-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Content */}
        <div className="min-h-0 flex-1 overflow-auto p-5">
          {activeTab === "markdown" && (
            <div className="space-y-3">
              <p className="text-xs text-muted-foreground">
                Full dialogue transcript with questions, reflections, and session
                statistics. Paste into your thesis or documentation.
              </p>
              <pre className="max-h-[40vh] overflow-auto rounded-lg border bg-secondary/30 p-4 font-mono text-xs leading-relaxed text-foreground/80">
                {markdown}
              </pre>
            </div>
          )}

          {activeTab === "mermaid" && (
            <div className="space-y-3">
              <p className="text-xs text-muted-foreground">
                Mermaid flowchart syntax. Renders as a diagram in GitHub,
                Notion, or any Mermaid-compatible viewer.
              </p>
              <pre className="max-h-[40vh] overflow-auto rounded-lg border bg-secondary/30 p-4 font-mono text-xs leading-relaxed text-foreground/80">
                {mermaid}
              </pre>
            </div>
          )}

          {activeTab === "image" && (
            <div className="space-y-4">
              <p className="text-xs text-muted-foreground">
                Export the exploration graph as an image. PNG for documents and
                presentations, SVG for scalable vector output.
              </p>
              <div className="flex gap-3">
                <button
                  onClick={() => handleDownloadImage("png")}
                  disabled={imageStatus === "generating"}
                  className={cn(
                    "inline-flex flex-1 items-center justify-center gap-2 rounded-lg border py-6",
                    "text-sm font-medium transition-all cursor-pointer",
                    "hover:border-foreground/20 hover:bg-secondary/30",
                    "disabled:opacity-50",
                  )}
                >
                  <Image className="h-5 w-5 text-muted-foreground" />
                  <div className="text-left">
                    <div>Download PNG</div>
                    <div className="text-[10px] text-muted-foreground">
                      High-res bitmap (2x)
                    </div>
                  </div>
                </button>
                <button
                  onClick={() => handleDownloadImage("svg")}
                  disabled={imageStatus === "generating"}
                  className={cn(
                    "inline-flex flex-1 items-center justify-center gap-2 rounded-lg border py-6",
                    "text-sm font-medium transition-all cursor-pointer",
                    "hover:border-foreground/20 hover:bg-secondary/30",
                    "disabled:opacity-50",
                  )}
                >
                  <GitBranch className="h-5 w-5 text-muted-foreground" />
                  <div className="text-left">
                    <div>Download SVG</div>
                    <div className="text-[10px] text-muted-foreground">
                      Scalable vector
                    </div>
                  </div>
                </button>
              </div>
              {imageStatus === "generating" && (
                <p className="text-center text-xs text-muted-foreground">
                  Generating image...
                </p>
              )}
              {imageStatus === "done" && (
                <p className="text-center text-xs text-emerald-600">
                  Image downloaded!
                </p>
              )}
              {imageStatus === "error" && (
                <p className="text-center text-xs text-red-600">
                  Failed to generate image. Try scrolling the graph into view first.
                </p>
              )}
            </div>
          )}
        </div>

        {/* Footer (for text tabs) */}
        {activeTab !== "image" && (
          <div className="flex items-center justify-end gap-2 border-t px-5 py-3">
            <button
              onClick={() =>
                handleCopy(activeTab === "markdown" ? markdown : mermaid)
              }
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5",
                "text-xs font-medium transition-all cursor-pointer",
                "hover:border-foreground/20 hover:bg-secondary/30",
              )}
            >
              {copied ? (
                <>
                  <Check className="h-3.5 w-3.5 text-emerald-600" />
                  Copied!
                </>
              ) : (
                <>
                  <Copy className="h-3.5 w-3.5" />
                  Copy
                </>
              )}
            </button>
            <button
              onClick={() =>
                handleDownloadText(
                  activeTab === "markdown" ? markdown : mermaid,
                  activeTab === "markdown"
                    ? "socratic-path-exploration.md"
                    : "socratic-path-exploration.mmd",
                )
              }
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5",
                "bg-foreground text-background text-xs font-medium",
                "transition-all cursor-pointer",
                "hover:bg-foreground/90",
              )}
            >
              <Download className="h-3.5 w-3.5" />
              Download
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
