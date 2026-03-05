import dagre from "@dagrejs/dagre";
import type { Node, Edge } from "@xyflow/react";
import type { ExplorationNode, QuestionType } from "./types";
import type { InputNodeData } from "@/components/nodes/InputNode";
import type { QuestionNodeData } from "@/components/nodes/QuestionNode";
import type { ReflectionNodeData } from "@/components/nodes/ReflectionNode";

// ── Node dimensions by type ─────────────────────────────────

const NODE_DIMENSIONS: Record<string, { width: number; height: number }> = {
  explorationInput: { width: 280, height: 120 },
  explorationQuestion: { width: 260, height: 130 },
  explorationReflection: { width: 240, height: 100 },
};

// ── Layout configuration ────────────────────────────────────

const GRAPH_CONFIG = {
  rankdir: "TB" as const, // top-to-bottom
  ranksep: 80, // vertical spacing between ranks
  nodesep: 40, // horizontal spacing between nodes
  marginx: 20,
  marginy: 20,
};

// ── Build React Flow graph from exploration tree ────────────

export function buildExplorationGraph(
  nodes: Record<string, ExplorationNode>,
  rootId: string | null,
  callbacks: {
    onExploreQuestion?: (nodeId: string) => void;
  },
): { nodes: Node[]; edges: Edge[] } {
  if (!rootId || !nodes[rootId]) {
    return { nodes: [], edges: [] };
  }

  const flowNodes: Node[] = [];
  const flowEdges: Edge[] = [];

  // Recursive tree walk to collect nodes and edges
  function walk(nodeId: string) {
    const node = nodes[nodeId];
    if (!node) return;

    // Skip collapsed subtrees (but still show the collapsed node itself)
    const flowNode = toFlowNode(node, nodes, callbacks);
    if (flowNode) {
      flowNodes.push(flowNode);
    }

    // Add edges from parent to this node
    if (node.parentId) {
      flowEdges.push({
        id: `edge-${node.parentId}-${node.id}`,
        source: node.parentId,
        target: node.id,
        type: "bezier",
        animated: false,
        style: {
          stroke: edgeColor(node),
          strokeWidth: 1.5,
        },
      });
    }

    // Recurse into children (unless collapsed)
    if (!node.isCollapsed) {
      for (const childId of node.children) {
        walk(childId);
      }
    }
  }

  walk(rootId);

  // Apply dagre layout
  return applyDagreLayout(flowNodes, flowEdges);
}

// ── Convert ExplorationNode → React Flow Node ───────────────

function toFlowNode(
  node: ExplorationNode,
  allNodes: Record<string, ExplorationNode>,
  callbacks: { onExploreQuestion?: (nodeId: string) => void },
): Node | null {
  switch (node.type) {
    case "input": {
      const data: InputNodeData = {
        label: node.text,
        inputType: node.metadata.inputType,
        pipelinePath: node.metadata.pipelinePath,
        isRoot: node.parentId === null,
      };
      return {
        id: node.id,
        type: "explorationInput",
        position: { x: 0, y: 0 }, // dagre will set this
        data,
      };
    }

    case "question": {
      const isExplored = node.children.length > 0;
      const data: QuestionNodeData = {
        label: node.text,
        questionType: (node.metadata.questionType ?? "clarity") as QuestionType,
        isExplored,
        nodeId: node.id,
        onExplore: callbacks.onExploreQuestion,
      };
      return {
        id: node.id,
        type: "explorationQuestion",
        position: { x: 0, y: 0 },
        data,
      };
    }

    case "reflection": {
      const childQuestionCount = node.children.filter(
        (id) => allNodes[id]?.type === "question",
      ).length;
      const data: ReflectionNodeData = {
        label: node.text,
        childCount: childQuestionCount,
      };
      return {
        id: node.id,
        type: "explorationReflection",
        position: { x: 0, y: 0 },
        data,
      };
    }

    default:
      return null;
  }
}

// ── Dagre auto-layout ───────────────────────────────────────

function applyDagreLayout(
  flowNodes: Node[],
  flowEdges: Edge[],
): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph(GRAPH_CONFIG);

  // Add nodes with dimensions
  for (const node of flowNodes) {
    const dims =
      NODE_DIMENSIONS[node.type ?? "explorationQuestion"] ??
      NODE_DIMENSIONS.explorationQuestion;
    g.setNode(node.id, { width: dims.width, height: dims.height });
  }

  // Add edges
  for (const edge of flowEdges) {
    g.setEdge(edge.source, edge.target);
  }

  dagre.layout(g);

  // Apply computed positions (dagre gives centre coords, React Flow uses top-left)
  const positionedNodes = flowNodes.map((node) => {
    const dagreNode = g.node(node.id);
    const dims =
      NODE_DIMENSIONS[node.type ?? "explorationQuestion"] ??
      NODE_DIMENSIONS.explorationQuestion;

    return {
      ...node,
      position: {
        x: dagreNode.x - dims.width / 2,
        y: dagreNode.y - dims.height / 2,
      },
    };
  });

  return { nodes: positionedNodes, edges: flowEdges };
}

// ── Edge styling helpers ────────────────────────────────────

function edgeColor(targetNode: ExplorationNode): string {
  switch (targetNode.type) {
    case "question":
      return "#d4d0cb"; // muted warm gray
    case "reflection":
      return "#a8a29e"; // slightly darker for reflection connections
    default:
      return "#d4d0cb";
  }
}
