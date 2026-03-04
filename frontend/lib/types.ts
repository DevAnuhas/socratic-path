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
