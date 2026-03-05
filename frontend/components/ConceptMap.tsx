"use client";

import { useMemo, useCallback } from "react";
import {
	ReactFlow,
	Background,
	BackgroundVariant,
	Controls,
	type Node,
	type Edge,
	type NodeTypes,
	useNodesState,
	useEdgesState,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { useAppStore } from "@/lib/store";
import { TopicNode } from "./nodes/TopicNode";
import { KeyphraseNode } from "./nodes/KeyphraseNode";
import type { Keyphrase, ContextSource } from "@/lib/types";

const nodeTypes: NodeTypes = {
	topic: TopicNode,
	keyphrase: KeyphraseNode,
};

/**
 * Compute radial positions for keyphrase nodes around a centre point.
 * Distributes nodes evenly in a circle with the topic at centre.
 */
function buildLayout(
	topic: string,
	keyphrases: Keyphrase[],
	sources: ContextSource[],
	selectedKeyphrase: string | null,
	onSelect: (kp: string) => void,
	onExplore: (kp: string) => void,
): { nodes: Node[]; edges: Edge[] } {
	const cx = 0;
	const cy = 0;
	const radius = 200;

	const sourceKeyphrases = new Set(
		sources.map((s) => s.keyphrase.toLowerCase()),
	);

	const topicNode: Node = {
		id: "topic",
		type: "topic",
		position: { x: cx, y: cy },
		data: { label: topic },
		draggable: true,
	};

	const kpNodes: Node[] = keyphrases.map((kp, i) => {
		const angle = (2 * Math.PI * i) / keyphrases.length - Math.PI / 2;
		return {
			id: `kp-${i}`,
			type: "keyphrase",
			position: {
				x: cx + radius * Math.cos(angle),
				y: cy + radius * Math.sin(angle),
			},
			data: {
				label: kp.text,
				score: kp.score,
				isSelected: selectedKeyphrase === kp.text,
				hasContext: sourceKeyphrases.has(kp.text.toLowerCase()),
				onSelect,
				onExplore,
			},
			draggable: true,
		};
	});

	// Determine handle pairing based on angle
	const edges: Edge[] = keyphrases.map((kp, i) => {
		const angle = (2 * Math.PI * i) / keyphrases.length - Math.PI / 2;
		const normAngle = ((angle % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);

		// Pick source handle on topic side closest to the keyphrase
		let sourceHandle: string | undefined;
		let targetHandle: string | undefined;

		if (normAngle >= 0 && normAngle < Math.PI / 4) {
			sourceHandle = undefined; // right (default)
			targetHandle = undefined; // left (default)
		} else if (normAngle < (3 * Math.PI) / 4) {
			sourceHandle = "bottom";
			targetHandle = "top";
		} else if (normAngle < (5 * Math.PI) / 4) {
			sourceHandle = "left";
			targetHandle = "right";
		} else if (normAngle < (7 * Math.PI) / 4) {
			sourceHandle = "top";
			targetHandle = "bottom";
		} else {
			sourceHandle = undefined;
			targetHandle = undefined;
		}

		return {
			id: `edge-${i}`,
			source: "topic",
			target: `kp-${i}`,
			sourceHandle,
			targetHandle,
			animated: selectedKeyphrase === kp.text,
			style: {
				stroke: selectedKeyphrase === kp.text ? "#1a1a1a" : "#d4d0cb",
				strokeWidth: selectedKeyphrase === kp.text ? 2 : 1.5,
			},
		};
	});

	return { nodes: [topicNode, ...kpNodes], edges };
}

export function ConceptMap() {
	const {
		topic,
		keyphrases,
		sources,
		selectedKeyphrase,
		selectKeyphrase,
		setTopic,
		generate,
	} = useAppStore();

	const handleSelect = useCallback(
		(kp: string) => selectKeyphrase(kp),
		[selectKeyphrase],
	);

	const handleExplore = useCallback(
		(kp: string) => {
			setTopic(kp);
			// Small delay so the store updates before generating
			setTimeout(() => {
				useAppStore.getState().generate();
			}, 50);
		},
		[setTopic],
	);

	const { nodes: initialNodes, edges: initialEdges } = useMemo(
		() =>
			buildLayout(
				topic,
				keyphrases,
				sources,
				selectedKeyphrase,
				handleSelect,
				handleExplore,
			),
		[
			topic,
			keyphrases,
			sources,
			selectedKeyphrase,
			handleSelect,
			handleExplore,
		],
	);

	const [nodes, , onNodesChange] = useNodesState(initialNodes);
	const [edges, , onEdgesChange] = useEdgesState(initialEdges);

	// Sync when data changes (new generation)
	// We use key prop on ReactFlow to force remount on topic change
	const flowKey = `${topic}-${keyphrases.length}-${selectedKeyphrase}`;

	if (keyphrases.length === 0) return null;

	return (
		<div className="h-[500px] w-full overflow-hidden rounded-lg border bg-white">
			<ReactFlow
				key={flowKey}
				nodes={initialNodes}
				edges={initialEdges}
				onNodesChange={onNodesChange}
				onEdgesChange={onEdgesChange}
				nodeTypes={nodeTypes}
				fitView
				fitViewOptions={{ padding: 0.3 }}
				minZoom={0.5}
				maxZoom={1.5}
				proOptions={{ hideAttribution: true }}
				className="bg-[#fdfcfb]"
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
