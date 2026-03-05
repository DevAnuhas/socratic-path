export interface Keyphrase {
  text: string;
  score: number;
}

export interface ContextSource {
  keyphrase: string;
  summary: string;
  url: string | null;
}

export interface Question {
  id: string;
  type: QuestionType;
  text: string;
  related_keyphrases: string[];
}

export interface GenerateRequest {
  topic: string;
  question_types: QuestionType[];
}

export interface GenerateResponse {
  topic: string;
  keyphrases: Keyphrase[];
  context_sources: ContextSource[];
  questions: Question[];
  processing_time_ms: number;
}

export type QuestionType =
  | "clarity"
  | "reasons_evidence"
  | "implication_consequences"
  | "alternate_viewpoints_perspectives"
  | "assumptions";

export const QUESTION_TYPE_CONFIG: Record<
  QuestionType,
  { label: string; color: string; bg: string; border: string }
> = {
  clarity: {
    label: "Clarity",
    color: "text-blue-600",
    bg: "bg-blue-50",
    border: "border-blue-200",
  },
  reasons_evidence: {
    label: "Evidence",
    color: "text-emerald-600",
    bg: "bg-emerald-50",
    border: "border-emerald-200",
  },
  implication_consequences: {
    label: "Implications",
    color: "text-amber-600",
    bg: "bg-amber-50",
    border: "border-amber-200",
  },
  alternate_viewpoints_perspectives: {
    label: "Viewpoints",
    color: "text-pink-600",
    bg: "bg-pink-50",
    border: "border-pink-200",
  },
  assumptions: {
    label: "Assumptions",
    color: "text-violet-600",
    bg: "bg-violet-50",
    border: "border-violet-200",
  },
};

export const ALL_QUESTION_TYPES: QuestionType[] = [
  "clarity",
  "reasons_evidence",
  "implication_consequences",
  "alternate_viewpoints_perspectives",
  "assumptions",
];

// ── Exploration Tree Types ──────────────────────────────────

export type InputType = "argumentative" | "factual" | "opinion" | "vague";
export type PipelinePath = "wikipedia" | "gemini" | "fallback";
export type NodeRole = "input" | "question" | "reflection";
export type GenerationStage =
  | "idle"
  | "classifying"
  | "gathering-context"
  | "generating-questions";

export interface ExplorationNode {
  id: string;
  type: NodeRole;
  text: string;
  parentId: string | null;
  depth: number;
  metadata: {
    questionType?: QuestionType;
    inputType?: InputType;
    pipelinePath?: PipelinePath;
    keyphrases?: Keyphrase[];
    sources?: ContextSource[];
    processingTimeMs?: number;
    depthNudge?: string | null;
  };
  children: string[];
  isCollapsed: boolean;
}

export interface AncestryNode {
  role: NodeRole;
  text: string;
}

export interface InputClassification {
  input_type: InputType;
  core_thesis: string | null;
  confidence: number;
  reasoning: string;
}

// ── Explore API Types ───────────────────────────────────────

export interface ExploreRequest {
  text: string;
  parent_question_id: string | null;
  ancestry: AncestryNode[];
  depth: number;
  question_types: QuestionType[];
}

export interface ExploreResponse {
  input_classification: InputClassification;
  pipeline_path: PipelinePath;
  keyphrases: Keyphrase[];
  context_sources: ContextSource[];
  questions: Question[];
  depth_nudge: string | null;
  processing_time_ms: number;
}

// ── Exploration Persistence Types ─────────────────────────

export interface ExplorationSummary {
  id: string;
  title: string;
  root_node_id: string;
  node_count: number;
  created_at: string;
  updated_at: string;
}

export interface ExplorationNodeRow {
  node_id: string;
  node_type: NodeRole;
  text: string;
  parent_node_id: string | null;
  depth: number;
  metadata: Record<string, unknown>;
  children: string[];
  sort_order: number;
}

export interface ExplorationDetail {
  id: string;
  title: string;
  root_node_id: string;
  node_count: number;
  created_at: string;
  updated_at: string;
  nodes: ExplorationNodeRow[];
}

export interface SaveExplorationPayload {
  exploration_id?: string;
  title: string;
  root_node_id: string;
  nodes: {
    node_id: string;
    node_type: NodeRole;
    text: string;
    parent_node_id: string | null;
    depth: number;
    metadata: Record<string, unknown>;
    children: string[];
  }[];
}
