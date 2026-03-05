"""
Gemini service for input classification and context generation.

This service acts as an intelligent preprocessing layer for the T5 LoRA model.
It does NOT generate Socratic questions — that remains the fine-tuned model's job.

Responsibilities:
  1. Classify user input (argumentative, factual, opinion, vague)
  2. Generate argumentatively relevant context for non-factual inputs
  3. Summarize deep ancestry chains that exceed T5's 512-token input limit
"""

import json
import logging
import os
from typing import Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


# ── Prompt templates ──────────────────────────────────────────────────────────

_CLASSIFY_PROMPT = """\
You are an input classifier for a Socratic questioning system.

Classify the following user input into exactly ONE of these categories:

- "argumentative": The user makes a claim, takes a position, or presents a thesis they believe in. Example: "I think AI will replace most jobs."
- "factual": The user asks about or mentions a factual/academic topic without taking a strong personal position. Example: "Climate change effects on agriculture."
- "opinion": The user expresses a preference or feeling without supporting reasoning. Example: "I like football."
- "vague": The input is too short, unclear, or lacks enough substance for meaningful Socratic questioning. Example: "technology" or "hello"

Also extract the core claim or thesis if one exists.

Respond with ONLY a JSON object (no markdown fences):
{
  "input_type": "argumentative" | "factual" | "opinion" | "vague",
  "core_thesis": "the main claim or topic, rephrased as a clear statement" or null,
  "confidence": 0.0 to 1.0,
  "reasoning": "one sentence explaining why"
}

User input: """

_CONTEXT_PROMPT = """\
You are a context generator for a Socratic questioning system. Your job is to provide relevant background information that will help challenge and probe the user's argument.

The user's input has been classified as: {input_type}
Core thesis: {core_thesis}

Generate 2-3 paragraphs of relevant context that includes:
1. Key counter-arguments or alternative perspectives to the user's position
2. Relevant facts, statistics, or evidence that could challenge or support the claim
3. Related concepts the user may not have considered

Do NOT generate questions. Only provide factual, balanced context that a Socratic questioning model can use to generate probing questions.

Keep the total response under 500 words.

User input: {user_input}"""

_ANCESTRY_SUMMARY_PROMPT = """\
Summarize the following Socratic dialogue thread into a concise paragraph (max 150 words). Preserve the key arguments, claims, and counter-points. This summary will be used as context for generating follow-up questions.

Dialogue thread:
{ancestry_text}"""


class GeminiService:
    """Gemini for input classification and context generation."""

    def __init__(self):
        self._model = None
        self._is_loaded = False

    def load(self) -> None:
        """Configure the Gemini API client."""
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning(
                "No GEMINI_API_KEY or GOOGLE_API_KEY found in environment. "
                "Gemini features will be disabled — all inputs will route to Wikipedia pipeline."
            )
            self._is_loaded = False
            return

        self._client = genai.Client(api_key=api_key)
        self._model_name = "gemini-2.5-flash"
        self._is_loaded = True
        logger.info("Gemini 2.5 Flash initialized successfully.")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def classify_input(self, text: str) -> dict:
        """
        Classify user input into: argumentative, factual, opinion, vague.

        Returns dict with keys: input_type, core_thesis, confidence, reasoning.
        Falls back to 'factual' if Gemini is unavailable or fails.
        """
        if not self._is_loaded:
            return _fallback_classification(text)

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=_CLASSIFY_PROMPT + text,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=256,
                ),
            )
            raw = response.text.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            result = json.loads(raw)

            # Validate required fields
            valid_types = {"argumentative", "factual", "opinion", "vague"}
            if result.get("input_type") not in valid_types:
                result["input_type"] = "factual"

            return {
                "input_type": result.get("input_type", "factual"),
                "core_thesis": result.get("core_thesis"),
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": result.get("reasoning", ""),
            }

        except Exception as exc:
            logger.warning("Gemini classification failed, falling back: %s", exc)
            return _fallback_classification(text)

    def generate_context(
        self,
        user_input: str,
        input_type: str,
        core_thesis: Optional[str] = None,
    ) -> str:
        """
        Generate argumentatively relevant context for the T5 model.

        For argumentative inputs: counter-arguments, evidence, related concepts.
        For opinion inputs: reformulated as a challengeable claim + context.
        Returns empty string if Gemini is unavailable.
        """
        if not self._is_loaded:
            return ""

        try:
            prompt = _CONTEXT_PROMPT.format(
                input_type=input_type,
                core_thesis=core_thesis or user_input,
                user_input=user_input,
            )

            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=600,
                ),
            )
            return response.text.strip()

        except Exception as exc:
            logger.warning("Gemini context generation failed: %s", exc)
            return ""

    def summarize_ancestry(self, ancestry: list[dict]) -> str:
        """
        Compress a deep ancestry chain into a concise summary.

        Used when exploration depth > 3 to fit within T5's 512-token input limit.
        Each ancestry item has 'role' (input/question/reflection) and 'text'.
        """
        if not self._is_loaded or not ancestry:
            # Fallback: just concatenate and truncate
            parts = [f"[{a['role']}] {a['text']}" for a in ancestry]
            return " ".join(parts)[:400]

        try:
            ancestry_text = "\n".join(
                f"[{a['role'].upper()}]: {a['text']}" for a in ancestry
            )

            prompt = _ANCESTRY_SUMMARY_PROMPT.format(ancestry_text=ancestry_text)

            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=300,
                ),
            )
            return response.text.strip()

        except Exception as exc:
            logger.warning("Gemini ancestry summarization failed: %s", exc)
            parts = [f"[{a['role']}] {a['text']}" for a in ancestry]
            return " ".join(parts)[:400]


def _fallback_classification(text: str) -> dict:
    """
    Rule-based fallback when Gemini is unavailable.

    Simple heuristics:
    - Very short input → vague
    - Contains opinion indicators → argumentative
    - Otherwise → factual (sends to Wikipedia pipeline)
    """
    text_lower = text.strip().lower()

    if len(text_lower) < 15:
        return {
            "input_type": "vague",
            "core_thesis": None,
            "confidence": 0.6,
            "reasoning": "Input too short for meaningful classification.",
        }

    opinion_markers = [
        "i think", "i believe", "in my opinion", "i feel",
        "should", "must", "better than", "worse than",
        "is the best", "is the worst", "i like", "i hate",
        "we should", "they should", "it's wrong", "it's right",
        "i disagree", "i agree", "obviously", "clearly",
    ]

    if any(marker in text_lower for marker in opinion_markers):
        return {
            "input_type": "argumentative",
            "core_thesis": text.strip(),
            "confidence": 0.5,
            "reasoning": "Contains opinion/position markers (rule-based fallback).",
        }

    return {
        "input_type": "factual",
        "core_thesis": None,
        "confidence": 0.4,
        "reasoning": "Default classification (rule-based fallback).",
    }
