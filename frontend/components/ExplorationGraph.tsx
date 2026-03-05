"use client";

import { useMemo, useCallback, useEffect, useRef } from "react";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  type NodeTypes,
  useNodesState,
  useEdgesState,
  useReactFlow,
  ReactFlowProvider,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { useAppStore } from "@/lib/store";
import { buildExplorationGraph } from "@/lib/graph-layout";
import { InputNode } from "./nodes/InputNode";
import { QuestionNode } from "./nodes/QuestionNode";
import { ReflectionNode } from "./nodes/ReflectionNode";

const nodeTypes: NodeTypes = {
  explorationInput: InputNode,
  explorationQuestion: QuestionNode,
  explorationReflection: ReflectionNode,
};

function ExplorationGraphInner() {
  const storeNodes = useAppStore((s) => s.nodes);
  const rootId = useAppStore((s) => s.rootId);
  const setActiveReflection = useAppStore((s) => s.setActiveReflection);

  const { fitView } = useReactFlow();
  const prevNodeCount = useRef(0);

  const onExploreQuestion = useCallback(
    (nodeId: string) => {
      setActiveReflection(nodeId);
      // Scroll to the question card below the graph
      const el = document.getElementById(`question-card-${nodeId}`);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    },
    [setActiveReflection],
  );

  const { nodes: layoutNodes, edges: layoutEdges } = useMemo(
    () =>
      buildExplorationGraph(storeNodes, rootId, {
        onExploreQuestion,
      }),
    [storeNodes, rootId, onExploreQuestion],
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(layoutNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(layoutEdges);

  // Sync when layout changes (new nodes added via reflection)
  useEffect(() => {
    setNodes(layoutNodes);
    setEdges(layoutEdges);

    // Fit view when new nodes appear
    if (layoutNodes.length !== prevNodeCount.current) {
      prevNodeCount.current = layoutNodes.length;
      // Small delay to let React Flow render nodes before fitting
      requestAnimationFrame(() => {
        fitView({ padding: 0.2, duration: 400 });
      });
    }
  }, [layoutNodes, layoutEdges, setNodes, setEdges, fitView]);

  if (layoutNodes.length === 0) return null;

  return (
    <div className="h-[500px] w-full overflow-hidden rounded-lg border bg-white">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.3}
        maxZoom={1.5}
        proOptions={{ hideAttribution: true }}
        className="bg-[#fdfcfb]"
        nodesDraggable
        nodesConnectable={false}
        elementsSelectable
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="#e5e2dd"
        />
        <Controls
          showInteractive={false}
          className="border-border! shadow-sm! [&>button]:border-border! [&>button]:bg-white!"
        />
      </ReactFlow>
    </div>
  );
}

export function ExplorationGraph() {
  return (
    <ReactFlowProvider>
      <ExplorationGraphInner />
    </ReactFlowProvider>
  );
}
