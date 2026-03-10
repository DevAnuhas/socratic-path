/**
 * Unit tests for the Zustand exploration store (lib/store.ts).
 *
 * Tests cover:
 * - getAncestryPath(): traverses parent_node_id chain back to root
 * - submitReflection(): optimistic node creation + rollback on API failure
 *
 * All network calls are mocked; no live backend is required.
 */

import { useAppStore } from "../../lib/store";
import { exploreQuestion, saveExploration } from "../../lib/api";
import type { ExplorationNode } from "../../lib/types";

// ── Module mocks ────────────────────────────────────────────

jest.mock("../../lib/api", () => ({
  exploreQuestion: jest.fn(),
  saveExploration: jest.fn().mockResolvedValue({ id: "saved-1" }),
  listExplorations: jest.fn().mockResolvedValue([]),
  getExploration: jest.fn(),
  deleteExploration: jest.fn(),
  checkHealth: jest.fn(),
}));

jest.mock("../../lib/supabase", () => ({
  createClient: jest.fn(() => ({
    auth: {
      getSession: jest.fn().mockResolvedValue({ data: { session: null } }),
    },
  })),
}));

const mockExploreQuestion = exploreQuestion as jest.Mock;
const mockSaveExploration = saveExploration as jest.Mock;

// ── Helpers ──────────────────────────────────────────────────

function makeInputNode(overrides: Partial<ExplorationNode> = {}): ExplorationNode {
  return {
    id: "input_1",
    type: "input",
    text: "Initial topic",
    parentId: null,
    depth: 0,
    metadata: {},
    children: [],
    isCollapsed: false,
    ...overrides,
  };
}

function makeQuestionNode(overrides: Partial<ExplorationNode> = {}): ExplorationNode {
  return {
    id: "q_1",
    type: "question",
    text: "What do you mean by that?",
    parentId: "input_1",
    depth: 0,
    metadata: { questionType: "clarity" as const },
    children: [],
    isCollapsed: false,
    ...overrides,
  };
}

function makeExploreResponse(questionText = "What is your evidence?") {
  return {
    input_classification: {
      input_type: "factual" as const,
      core_thesis: null,
      confidence: 0.9,
      reasoning: "",
    },
    pipeline_path: "wikipedia" as const,
    keyphrases: [],
    context_sources: [],
    questions: [
      { id: "q_server_1", type: "clarity", text: questionText, related_keyphrases: [] },
    ],
    depth_nudge: null,
    processing_time_ms: 1500,
  };
}

// ── Setup ────────────────────────────────────────────────────

beforeEach(() => {
  useAppStore.getState().reset();
  jest.clearAllMocks();
  mockSaveExploration.mockResolvedValue({ id: "saved-1" });
});

// ── getAncestryPath ───────────────────────────────────────────

describe("getAncestryPath", () => {
  it("returns empty array for an unknown node ID", () => {
    const path = useAppStore.getState().getAncestryPath("nonexistent");
    expect(path).toEqual([]);
  });

  it("returns a single-element path for the root input node", () => {
    useAppStore.setState({
      nodes: { input_1: makeInputNode() },
      rootId: "input_1",
    });
    const path = useAppStore.getState().getAncestryPath("input_1");
    expect(path).toEqual([{ role: "input", text: "Initial topic" }]);
  });

  it("returns [input, question] for a direct child of root", () => {
    useAppStore.setState({
      nodes: {
        input_1: makeInputNode({ children: ["q_1"] }),
        q_1: makeQuestionNode(),
      },
      rootId: "input_1",
    });
    const path = useAppStore.getState().getAncestryPath("q_1");
    expect(path).toEqual([
      { role: "input", text: "Initial topic" },
      { role: "question", text: "What do you mean by that?" },
    ]);
  });

  it("returns the full chain for a three-level tree", () => {
    useAppStore.setState({
      nodes: {
        input_1: makeInputNode({ children: ["q_1"] }),
        q_1: makeQuestionNode({ children: ["ref_1"] }),
        ref_1: {
          id: "ref_1",
          type: "reflection",
          text: "I think it is because of X.",
          parentId: "q_1",
          depth: 1,
          metadata: {},
          children: ["q_2"],
          isCollapsed: false,
        },
        q_2: {
          id: "q_2",
          type: "question",
          text: "What evidence supports X?",
          parentId: "ref_1",
          depth: 1,
          metadata: { questionType: "reasons_evidence" as const },
          children: [],
          isCollapsed: false,
        },
      },
      rootId: "input_1",
    });
    const path = useAppStore.getState().getAncestryPath("q_2");
    expect(path).toEqual([
      { role: "input", text: "Initial topic" },
      { role: "question", text: "What do you mean by that?" },
      { role: "reflection", text: "I think it is because of X." },
      { role: "question", text: "What evidence supports X?" },
    ]);
  });
});

// ── submitReflection — optimistic rollback ────────────────────

describe("submitReflection", () => {
  it("rolls back the optimistic reflection node when the API call fails", async () => {
    // Set up a store with one question node
    useAppStore.setState({
      nodes: {
        input_1: makeInputNode({ children: ["q_1"] }),
        q_1: makeQuestionNode(),
      },
      rootId: "input_1",
      generationStage: "idle",
      topic: "Initial topic",
    });

    mockExploreQuestion.mockRejectedValueOnce(new Error("Network error"));

    await useAppStore.getState().submitReflection("q_1", "My reflection text");

    const state = useAppStore.getState();

    // No reflection node should remain after rollback
    const reflectionNodes = Object.values(state.nodes).filter(
      (n) => n.type === "reflection"
    );
    expect(reflectionNodes).toHaveLength(0);

    // The question node's children must be empty (reflection removed)
    expect(state.nodes["q_1"].children).toHaveLength(0);

    // Error state should be set
    expect(state.error).toBe("Network error");
    expect(state.generationStage).toBe("idle");
  });

  it("adds question nodes to the reflection after a successful API call", async () => {
    useAppStore.setState({
      nodes: {
        input_1: makeInputNode({ children: ["q_1"] }),
        q_1: makeQuestionNode(),
      },
      rootId: "input_1",
      generationStage: "idle",
      topic: "Initial topic",
    });

    mockExploreQuestion.mockResolvedValueOnce(makeExploreResponse("What is your evidence?"));

    await useAppStore.getState().submitReflection("q_1", "My reflection text");

    const state = useAppStore.getState();

    // A reflection node should be present
    const reflectionNodes = Object.values(state.nodes).filter(
      (n) => n.type === "reflection"
    );
    expect(reflectionNodes).toHaveLength(1);
    expect(reflectionNodes[0].text).toBe("My reflection text");

    // The reflection node should have a child question
    const childQuestions = Object.values(state.nodes).filter(
      (n) => n.type === "question" && n.parentId === reflectionNodes[0].id
    );
    expect(childQuestions).toHaveLength(1);
    expect(childQuestions[0].text).toBe("What is your evidence?");

    // Generation stage should be idle
    expect(state.generationStage).toBe("idle");
    expect(state.error).toBeNull();
  });
});
