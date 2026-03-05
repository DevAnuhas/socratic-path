import { create } from "zustand";
import { persist } from "zustand/middleware";
import { exploreQuestion } from "./api";
import type {
  ExplorationNode,
  QuestionType,
  GenerationStage,
  AncestryNode,
  Keyphrase,
  ContextSource,
  InputClassification,
} from "./types";
import { ALL_QUESTION_TYPES } from "./types";

// ── Node ID generator ───────────────────────────────────────

let _nodeCounter = 0;

function nextNodeId(prefix: string): string {
  return `${prefix}_${++_nodeCounter}`;
}

// ── Store Interface ─────────────────────────────────────────

interface SocraticStore {
  // Exploration tree
  nodes: Record<string, ExplorationNode>;
  rootId: string | null;

  // Active UI state
  activeReflectionId: string | null;
  selectedNodeId: string | null;
  generationStage: GenerationStage;

  // Input
  topic: string;
  selectedTypes: QuestionType[];

  // Last classification (for InputPanel indicator)
  lastClassification: InputClassification | null;

  // Error state
  error: string | null;
  lastFailedAction: (() => Promise<void>) | null;

  // Actions — input
  setTopic: (topic: string) => void;
  toggleType: (type: QuestionType) => void;

  // Actions — core exploration
  submitInitialInput: () => Promise<void>;
  submitReflection: (
    questionId: string,
    reflectionText: string,
  ) => Promise<void>;

  // Actions — graph navigation
  selectNode: (nodeId: string | null) => void;
  setActiveReflection: (questionId: string | null) => void;
  collapseSubtree: (nodeId: string) => void;
  expandSubtree: (nodeId: string) => void;

  // Utilities
  getAncestryPath: (nodeId: string) => AncestryNode[];
  getRootKeyphrases: () => Keyphrase[];
  getRootSources: () => ContextSource[];
  getQuestionsByParent: (parentId: string) => ExplorationNode[];
  getAllQuestionNodes: () => ExplorationNode[];
  dismissError: () => void;
  retryLast: () => Promise<void>;
  reset: () => void;
}

// ── Initial state (extracted for reset) ─────────────────────

const initialState = {
  nodes: {} as Record<string, ExplorationNode>,
  rootId: null as string | null,
  activeReflectionId: null as string | null,
  selectedNodeId: null as string | null,
  generationStage: "idle" as GenerationStage,
  topic: "",
  selectedTypes: [...ALL_QUESTION_TYPES] as QuestionType[],
  lastClassification: null as InputClassification | null,
  error: null as string | null,
  lastFailedAction: null as (() => Promise<void>) | null,
};

// ── Store ───────────────────────────────────────────────────

export const useAppStore = create<SocraticStore>()(
  persist(
    (set, get) => ({
      ...initialState,

      // ── Input actions ──────────────────────────────────────

      setTopic: (topic) => set({ topic }),

      toggleType: (type) =>
        set((state) => {
          const has = state.selectedTypes.includes(type);
          if (has && state.selectedTypes.length === 1) return state;
          return {
            selectedTypes: has
              ? state.selectedTypes.filter((t) => t !== type)
              : [...state.selectedTypes, type],
          };
        }),

      // ── Core exploration ───────────────────────────────────

      submitInitialInput: async () => {
        const { topic, selectedTypes, generationStage } = get();
        if (!topic.trim() || generationStage !== "idle") return;

        // Clear previous exploration tree for fresh start
        _nodeCounter = 0;
        set({
          nodes: {},
          rootId: null,
          error: null,
          generationStage: "classifying",
          activeReflectionId: null,
          selectedNodeId: null,
          lastClassification: null,
          lastFailedAction: null,
        });

        // Advance stage indicators on a timer (visual feedback)
        const timer1 = setTimeout(
          () => set({ generationStage: "gathering-context" }),
          1200,
        );
        const timer2 = setTimeout(
          () => set({ generationStage: "generating-questions" }),
          2800,
        );

        try {
          const data = await exploreQuestion({
            text: topic.trim(),
            parent_question_id: null,
            ancestry: [],
            depth: 0,
            question_types: selectedTypes,
          });

          clearTimeout(timer1);
          clearTimeout(timer2);

          // Build tree from response
          const rootId = nextNodeId("input");
          const nodes: Record<string, ExplorationNode> = {};

          // Root input node
          nodes[rootId] = {
            id: rootId,
            type: "input",
            text: topic.trim(),
            parentId: null,
            depth: 0,
            metadata: {
              inputType: data.input_classification.input_type,
              pipelinePath: data.pipeline_path,
              keyphrases: data.keyphrases,
              sources: data.context_sources,
              processingTimeMs: data.processing_time_ms,
              depthNudge: data.depth_nudge,
            },
            children: [],
            isCollapsed: false,
          };

          // Question nodes as children of root
          for (const q of data.questions) {
            const qId = nextNodeId("q");
            nodes[qId] = {
              id: qId,
              type: "question",
              text: q.text,
              parentId: rootId,
              depth: 0,
              metadata: {
                questionType: q.type as QuestionType,
              },
              children: [],
              isCollapsed: false,
            };
            nodes[rootId].children.push(qId);
          }

          set({
            nodes,
            rootId,
            generationStage: "idle",
            lastClassification: data.input_classification,
          });
        } catch (err) {
          clearTimeout(timer1);
          clearTimeout(timer2);
          const message =
            err instanceof Error
              ? err.message
              : "Failed to generate questions";
          set({
            error: message,
            generationStage: "idle",
            lastFailedAction: () => get().submitInitialInput(),
          });
        }
      },

      submitReflection: async (questionId, reflectionText) => {
        const state = get();
        if (state.generationStage !== "idle") return;
        const questionNode = state.nodes[questionId];
        if (!questionNode) return;

        // Compute ancestry path from root to this question
        const ancestry = state.getAncestryPath(questionId);
        // Add the reflection itself
        ancestry.push({ role: "reflection", text: reflectionText });

        const refId = nextNodeId("ref");
        const refDepth = questionNode.depth + 1;

        // Optimistically add reflection node to tree
        set((prev) => {
          const updatedNodes = { ...prev.nodes };

          updatedNodes[refId] = {
            id: refId,
            type: "reflection",
            text: reflectionText,
            parentId: questionId,
            depth: refDepth,
            metadata: {},
            children: [],
            isCollapsed: false,
          };

          updatedNodes[questionId] = {
            ...updatedNodes[questionId],
            children: [...updatedNodes[questionId].children, refId],
          };

          return {
            nodes: updatedNodes,
            activeReflectionId: null,
            generationStage: "classifying",
            error: null,
          };
        });

        // Stage timers
        const timer1 = setTimeout(
          () => set({ generationStage: "gathering-context" }),
          1200,
        );
        const timer2 = setTimeout(
          () => set({ generationStage: "generating-questions" }),
          2800,
        );

        try {
          const data = await exploreQuestion({
            text: reflectionText.trim(),
            parent_question_id: questionId,
            ancestry,
            depth: refDepth,
            question_types: state.selectedTypes,
          });

          clearTimeout(timer1);
          clearTimeout(timer2);

          set((prev) => {
            const updatedNodes = { ...prev.nodes };

            // Update reflection node with response metadata
            updatedNodes[refId] = {
              ...updatedNodes[refId],
              metadata: {
                inputType: data.input_classification.input_type,
                pipelinePath: data.pipeline_path,
                keyphrases: data.keyphrases,
                sources: data.context_sources,
                processingTimeMs: data.processing_time_ms,
                depthNudge: data.depth_nudge,
              },
            };

            // Create child question nodes
            for (const q of data.questions) {
              const qId = nextNodeId("q");
              updatedNodes[qId] = {
                id: qId,
                type: "question",
                text: q.text,
                parentId: refId,
                depth: refDepth,
                metadata: {
                  questionType: q.type as QuestionType,
                },
                children: [],
                isCollapsed: false,
              };
              updatedNodes[refId] = {
                ...updatedNodes[refId],
                children: [...updatedNodes[refId].children, qId],
              };
            }

            return {
              nodes: updatedNodes,
              generationStage: "idle",
              lastClassification: data.input_classification,
            };
          });
        } catch (err) {
          clearTimeout(timer1);
          clearTimeout(timer2);
          const message =
            err instanceof Error
              ? err.message
              : "Failed to generate follow-up questions";

          // Roll back the optimistic reflection node
          set((prev) => {
            const updatedNodes = { ...prev.nodes };
            // Remove the reflection node
            delete updatedNodes[refId];
            // Remove refId from parent's children
            if (updatedNodes[questionId]) {
              updatedNodes[questionId] = {
                ...updatedNodes[questionId],
                children: updatedNodes[questionId].children.filter(
                  (id) => id !== refId,
                ),
              };
            }
            return {
              nodes: updatedNodes,
              error: message,
              generationStage: "idle",
              lastFailedAction: () =>
                get().submitReflection(questionId, reflectionText),
            };
          });
        }
      },

      // ── Graph navigation ───────────────────────────────────

      selectNode: (nodeId) => set({ selectedNodeId: nodeId }),

      setActiveReflection: (questionId) =>
        set({ activeReflectionId: questionId }),

      collapseSubtree: (nodeId) =>
        set((state) => {
          const node = state.nodes[nodeId];
          if (!node) return state;
          return {
            nodes: {
              ...state.nodes,
              [nodeId]: { ...node, isCollapsed: true },
            },
          };
        }),

      expandSubtree: (nodeId) =>
        set((state) => {
          const node = state.nodes[nodeId];
          if (!node) return state;
          return {
            nodes: {
              ...state.nodes,
              [nodeId]: { ...node, isCollapsed: false },
            },
          };
        }),

      // ── Utilities ──────────────────────────────────────────

      getAncestryPath: (nodeId) => {
        const { nodes } = get();
        const path: AncestryNode[] = [];
        let currentId: string | null = nodeId;

        while (currentId) {
          const node: ExplorationNode | undefined = nodes[currentId];
          if (!node) break;
          path.unshift({ role: node.type, text: node.text });
          currentId = node.parentId;
        }

        return path;
      },

      getRootKeyphrases: () => {
        const { nodes, rootId } = get();
        if (!rootId) return [];
        return nodes[rootId]?.metadata.keyphrases ?? [];
      },

      getRootSources: () => {
        const { nodes, rootId } = get();
        if (!rootId) return [];
        return nodes[rootId]?.metadata.sources ?? [];
      },

      getQuestionsByParent: (parentId) => {
        const { nodes } = get();
        const parent = nodes[parentId];
        if (!parent) return [];
        return parent.children
          .map((id) => nodes[id])
          .filter((n) => n?.type === "question");
      },

      getAllQuestionNodes: () => {
        const { nodes } = get();
        return Object.values(nodes).filter((n) => n.type === "question");
      },

      dismissError: () => set({ error: null, lastFailedAction: null }),

      retryLast: async () => {
        const action = get().lastFailedAction;
        if (!action) return;
        set({ error: null, lastFailedAction: null });
        await action();
      },

      reset: () => {
        _nodeCounter = 0;
        set({ ...initialState, selectedTypes: [...ALL_QUESTION_TYPES] });
      },
    }),
    {
      name: "socratic-path-exploration",
      partialize: (state) => ({
        nodes: state.nodes,
        rootId: state.rootId,
        topic: state.topic,
        selectedTypes: state.selectedTypes,
        lastClassification: state.lastClassification,
      }),
      onRehydrateStorage: () => (state) => {
        // Restore the node counter to avoid ID collisions
        if (state?.nodes) {
          const maxId = Object.keys(state.nodes).reduce((max, key) => {
            const num = parseInt(key.split("_").pop() || "0", 10);
            return isNaN(num) ? max : Math.max(max, num);
          }, 0);
          _nodeCounter = maxId;
        }
      },
    },
  ),
);
