import type { ExplorationNode, QuestionType } from "./types";
import { QUESTION_TYPE_CONFIG } from "./types";

// ── Markdown Export ─────────────────────────────────────────

export function exportMarkdown(
  nodes: Record<string, ExplorationNode>,
  rootId: string | null,
): string {
  if (!rootId || !nodes[rootId]) return "";

  const lines: string[] = [];
  const root = nodes[rootId];

  lines.push("# SocraticPath Exploration");
  lines.push("");
  lines.push(`**Topic:** ${root.text}`);

  if (root.metadata.inputType) {
    lines.push(`**Classification:** ${root.metadata.inputType}`);
  }
  if (root.metadata.pipelinePath) {
    lines.push(
      `**Context source:** ${root.metadata.pipelinePath === "wikipedia" ? "Wikipedia" : "AI-generated"}`,
    );
  }
  lines.push("");
  lines.push("---");
  lines.push("");

  // Walk children of root (questions)
  for (const childId of root.children) {
    walkMarkdown(nodes, childId, 0, lines);
  }

  // Metadata footer
  const totalTime = Object.values(nodes)
    .reduce((sum, n) => sum + (n.metadata.processingTimeMs ?? 0), 0);
  const totalQuestions = Object.values(nodes).filter(
    (n) => n.type === "question",
  ).length;
  const totalReflections = Object.values(nodes).filter(
    (n) => n.type === "reflection",
  ).length;
  const maxDepth = Math.max(...Object.values(nodes).map((n) => n.depth), 0);

  lines.push("");
  lines.push("---");
  lines.push("");
  lines.push("## Session Statistics");
  lines.push("");
  lines.push(`- **Questions generated:** ${totalQuestions}`);
  lines.push(`- **Reflections written:** ${totalReflections}`);
  lines.push(`- **Max exploration depth:** ${maxDepth}`);
  if (totalTime > 0) {
    lines.push(
      `- **Total processing time:** ${(totalTime / 1000).toFixed(1)}s`,
    );
  }
  lines.push("");
  lines.push(
    "*Exported from [SocraticPath](https://github.com/DevAnuhas/socratic-path)*",
  );

  return lines.join("\n");
}

function walkMarkdown(
  nodes: Record<string, ExplorationNode>,
  nodeId: string,
  indent: number,
  lines: string[],
): void {
  const node = nodes[nodeId];
  if (!node) return;

  const prefix = "  ".repeat(indent);

  if (node.type === "question") {
    const typeLabel = node.metadata.questionType
      ? QUESTION_TYPE_CONFIG[node.metadata.questionType as QuestionType]?.label ?? node.metadata.questionType
      : "Question";
    lines.push(`${prefix}- **[${typeLabel}]** ${node.text}`);
  } else if (node.type === "reflection") {
    lines.push(`${prefix}- **Reflection:** *"${node.text}"*`);
  }

  for (const childId of node.children) {
    walkMarkdown(nodes, childId, indent + 1, lines);
  }
}

// ── Mermaid Export ──────────────────────────────────────────

export function exportMermaid(
  nodes: Record<string, ExplorationNode>,
  rootId: string | null,
): string {
  if (!rootId || !nodes[rootId]) return "";

  const lines: string[] = [];
  lines.push("graph TD");

  // Define all nodes
  for (const node of Object.values(nodes)) {
    const safeText = escMermaid(truncate(node.text, 60));

    switch (node.type) {
      case "input":
        // Stadium shape for input
        lines.push(`  ${node.id}(["\u{1F4DD} ${safeText}"])`);
        break;
      case "question": {
        const typeLabel = node.metadata.questionType
          ? QUESTION_TYPE_CONFIG[node.metadata.questionType as QuestionType]?.label ?? ""
          : "";
        // Rounded rectangle for questions
        lines.push(
          `  ${node.id}("\u{2753} <b>${typeLabel}</b><br/>${safeText}")`,
        );
        break;
      }
      case "reflection":
        // Hexagon for reflections
        lines.push(`  ${node.id}{{"${"\u{1F4AC}"} ${safeText}"}}`);
        break;
    }
  }

  lines.push("");

  // Define edges
  for (const node of Object.values(nodes)) {
    if (node.parentId && nodes[node.parentId]) {
      lines.push(`  ${node.parentId} --> ${node.id}`);
    }
  }

  lines.push("");

  // Style classes
  lines.push("  classDef inputNode fill:#fef3c7,stroke:#f59e0b,color:#92400e");
  lines.push(
    "  classDef questionNode fill:#eff6ff,stroke:#3b82f6,color:#1e40af",
  );
  lines.push(
    "  classDef reflectionNode fill:#f3e8ff,stroke:#8b5cf6,color:#5b21b6",
  );

  // Apply styles
  const inputIds = Object.values(nodes)
    .filter((n) => n.type === "input")
    .map((n) => n.id);
  const questionIds = Object.values(nodes)
    .filter((n) => n.type === "question")
    .map((n) => n.id);
  const reflectionIds = Object.values(nodes)
    .filter((n) => n.type === "reflection")
    .map((n) => n.id);

  if (inputIds.length > 0) {
    lines.push(`  class ${inputIds.join(",")} inputNode`);
  }
  if (questionIds.length > 0) {
    lines.push(`  class ${questionIds.join(",")} questionNode`);
  }
  if (reflectionIds.length > 0) {
    lines.push(`  class ${reflectionIds.join(",")} reflectionNode`);
  }

  return lines.join("\n");
}

// ── Helpers ─────────────────────────────────────────────────

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 3) + "...";
}

function escMermaid(text: string): string {
  // Escape characters that break Mermaid syntax
  return text
    .replace(/"/g, "&quot;")
    .replace(/\n/g, " ");
}
