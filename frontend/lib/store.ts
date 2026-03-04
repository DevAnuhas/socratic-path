import { create } from "zustand";
import { generateQuestions } from "./api";
import type {
  Keyphrase,
  ContextSource,
  Question,
  QuestionType,
  GenerateResponse,
} from "./types";
import { ALL_QUESTION_TYPES } from "./types";

interface AppState {
  // Input
  topic: string;
  selectedTypes: QuestionType[];

  // Results
  keyphrases: Keyphrase[];
  questions: Question[];
  sources: ContextSource[];
  processingTimeMs: number | null;

  // UI
  isLoading: boolean;
  error: string | null;
  selectedKeyphrase: string | null;
  hasGenerated: boolean;

  // Actions
  setTopic: (topic: string) => void;
  toggleType: (type: QuestionType) => void;
  selectKeyphrase: (keyphrase: string | null) => void;
  generate: () => Promise<void>;
  reset: () => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  // Initial state
  topic: "",
  selectedTypes: [...ALL_QUESTION_TYPES],
  keyphrases: [],
  questions: [],
  sources: [],
  processingTimeMs: null,
  isLoading: false,
  error: null,
  selectedKeyphrase: null,
  hasGenerated: false,

  setTopic: (topic) => set({ topic }),

  toggleType: (type) =>
    set((state) => {
      const has = state.selectedTypes.includes(type);
      if (has && state.selectedTypes.length === 1) return state; // keep at least one
      return {
        selectedTypes: has
          ? state.selectedTypes.filter((t) => t !== type)
          : [...state.selectedTypes, type],
      };
    }),

  selectKeyphrase: (keyphrase) =>
    set((state) => ({
      selectedKeyphrase:
        state.selectedKeyphrase === keyphrase ? null : keyphrase,
    })),

  generate: async () => {
    const { topic, selectedTypes } = get();
    if (!topic.trim()) return;

    set({
      isLoading: true,
      error: null,
      selectedKeyphrase: null,
    });

    try {
      const data: GenerateResponse = await generateQuestions({
        topic: topic.trim(),
        question_types: selectedTypes,
      });
      set({
        keyphrases: data.keyphrases,
        questions: data.questions,
        sources: data.context_sources,
        processingTimeMs: data.processing_time_ms,
        isLoading: false,
        hasGenerated: true,
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to generate questions";
      set({ error: message, isLoading: false });
    }
  },

  reset: () =>
    set({
      topic: "",
      keyphrases: [],
      questions: [],
      sources: [],
      processingTimeMs: null,
      error: null,
      selectedKeyphrase: null,
      hasGenerated: false,
    }),
}));
