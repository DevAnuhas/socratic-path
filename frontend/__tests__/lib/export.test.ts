/**
 * Unit tests for lib/export.ts — exportMarkdown() and exportMermaid().
 *
 * Both functions are pure: they accept a nodes Record and a rootId string
 * and return a formatted string. No mocking is required.
 */

import { exportMarkdown, exportMermaid } from "../../lib/export";
import type { ExplorationNode } from "../../lib/types";

// ── Shared test fixture ───────────────────────────────────────

/**
 * Minimal two-level tree:
 *   input_1 (root)
 *   └── q_1  [clarity question]
 *       └── ref_1  [reflection]
 *           └── q_2  [reasons_evidence question]
 */
function buildTree(): Record<string, ExplorationNode> {
  return {
    input_1: {
      id: "input_1",
      type: "input",
      text: "Does social media harm teenagers?",
      parentId: null,
      depth: 0,
      metadata: {
        inputType: "argumentative",
        pipelinePath: "gemini",
        processingTimeMs: 3200,
      },
      children: ["q_1"],
      isCollapsed: false,
    },
    q_1: {
      id: "q_1",
      type: "question",
      text: "What do you mean by harm in this context?",
      parentId: "input_1",
      depth: 0,
      metadata: { questionType: "clarity" },
      children: ["ref_1"],
      isCollapsed: false,
    },
    ref_1: {
      id: "ref_1",
      type: "reflection",
      text: "I mean psychological harm like anxiety and depression.",
      parentId: "q_1",
      depth: 1,
      metadata: { processingTimeMs: 4100 },
      children: ["q_2"],
      isCollapsed: false,
    },
    q_2: {
      id: "q_2",
      type: "question",
      text: "What evidence links social media use to anxiety?",
      parentId: "ref_1",
      depth: 1,
      metadata: { questionType: "reasons_evidence" },
      children: [],
      isCollapsed: false,
    },
  };
}

// ── exportMarkdown ────────────────────────────────────────────

describe("exportMarkdown", () => {
  it("returns empty string when rootId is null", () => {
    expect(exportMarkdown({}, null)).toBe("");
  });

  it("returns empty string when root node is missing", () => {
    expect(exportMarkdown({}, "nonexistent_id")).toBe("");
  });

  it("includes the root topic text", () => {
    const result = exportMarkdown(buildTree(), "input_1");
    expect(result).toContain("Does social media harm teenagers?");
  });

  it("includes a question with its Socratic type label", () => {
    const result = exportMarkdown(buildTree(), "input_1");
    expect(result).toContain("[Clarity]");
    expect(result).toContain("What do you mean by harm in this context?");
  });

  it("includes a reflection with correct Markdown formatting", () => {
    const result = exportMarkdown(buildTree(), "input_1");
    expect(result).toContain("**Reflection:**");
    expect(result).toContain("I mean psychological harm like anxiety and depression.");
  });

  it("includes session statistics with correct counts", () => {
    const result = exportMarkdown(buildTree(), "input_1");
    expect(result).toContain("Questions generated:** 2");
    expect(result).toContain("Reflections written:** 1");
  });

  it("includes total processing time when present", () => {
    const result = exportMarkdown(buildTree(), "input_1");
    // Total: 3200 + 4100 = 7300 ms → 7.3 s
    expect(result).toContain("7.3s");
  });

  it("omits processing time line when no nodes have timing data", () => {
    const nodes: Record<string, ExplorationNode> = {
      input_1: {
        id: "input_1",
        type: "input",
        text: "Topic",
        parentId: null,
        depth: 0,
        metadata: {},
        children: [],
        isCollapsed: false,
      },
    };
    const result = exportMarkdown(nodes, "input_1");
    expect(result).not.toContain("Total processing time");
  });
});

// ── exportMermaid ─────────────────────────────────────────────

describe("exportMermaid", () => {
  it("returns empty string when rootId is null", () => {
    expect(exportMermaid({}, null)).toBe("");
  });

  it("returns empty string when root node is missing", () => {
    expect(exportMermaid({}, "nonexistent_id")).toBe("");
  });

  it("starts with graph TD declaration", () => {
    const result = exportMermaid(buildTree(), "input_1");
    expect(result.startsWith("graph TD")).toBe(true);
  });

  it("generates parent-to-child edges for all relationships", () => {
    const result = exportMermaid(buildTree(), "input_1");
    expect(result).toContain("input_1 --> q_1");
    expect(result).toContain("q_1 --> ref_1");
    expect(result).toContain("ref_1 --> q_2");
  });

  it("uses stadium shape for input nodes", () => {
    const result = exportMermaid(buildTree(), "input_1");
    // Stadium shape: ([...])
    expect(result).toMatch(/input_1\(\[".+"\]\)/);
  });

  it("applies node style classes", () => {
    const result = exportMermaid(buildTree(), "input_1");
    expect(result).toContain("classDef inputNode");
    expect(result).toContain("classDef questionNode");
    expect(result).toContain("classDef reflectionNode");
  });

  it("truncates long node text to 60 characters", () => {
    const longText = "A".repeat(80);
    const nodes: Record<string, ExplorationNode> = {
      input_1: {
        id: "input_1",
        type: "input",
        text: longText,
        parentId: null,
        depth: 0,
        metadata: {},
        children: [],
        isCollapsed: false,
      },
    };
    const result = exportMermaid(nodes, "input_1");
    // The truncated text should end with "..." and not contain the full 80-char string
    expect(result).toContain("...");
    expect(result).not.toContain(longText);
  });
});
