"""
Context Router — routes user input to the appropriate context generation strategy.

The smart router classifies input via Gemini Flash, then selects the best
context strategy for the T5 LoRA model:

  - Factual/Academic → KeyBERT + Wikipedia (grounded, citation-worthy)
  - Argumentative/Opinion → Gemini-generated context (argument-aware)
  - Vague → returns minimal context with a flag for the frontend

For exploration (depth > 0), the router also handles ancestry context:
  - Depth 1-3: Full ancestry chain included as-is
  - Depth > 3: Ancestry summarized via Gemini to fit T5's 512-token limit
"""

import logging
from dataclasses import dataclass, field

from backend.services.gemini_service import GeminiService
from backend.services.keyphrase import KeyphraseService
from backend.services.wikipedia import WikipediaService

logger = logging.getLogger(__name__)

# Depth threshold beyond which ancestry gets summarized
_SUMMARIZE_DEPTH = 3


@dataclass
class ContextResult:
    """Output of the context routing decision."""

    # What was determined about the input
    input_type: str                     # argumentative | factual | opinion | vague
    core_thesis: str | None          # Extracted thesis (if argumentative/opinion)
    classification_confidence: float

    # The context generated for T5
    combined_context: str               # Ready to feed to QuestionGenerationService
    pipeline_path: str                  # "wikipedia" | "gemini" | "fallback"

    # Metadata for the frontend
    keyphrases: list = field(default_factory=list)       # (text, score) tuples
    context_sources: list = field(default_factory=list)  # WikiResult dicts

    # Depth guidance
    depth_nudge: str | None = None   # Soft warning message if depth > 3


class ContextRouter:
    """Routes input to the appropriate context generation strategy."""

    def __init__(
        self,
        gemini_service: GeminiService,
        keyphrase_service: KeyphraseService,
        wikipedia_service: WikipediaService,
    ):
        self._gemini = gemini_service
        self._keyphrase = keyphrase_service
        self._wikipedia = wikipedia_service

    def route(
        self,
        text: str,
        ancestry: list[dict] | None = None,
        depth: int = 0,
    ) -> ContextResult:
        """
        Classify input and generate appropriate context.

        Args:
            text: The user's current input (topic or reflection)
            ancestry: List of {role, text} dicts from root to current node
            depth: Current depth in the exploration tree (0 = initial input)

        Returns:
            ContextResult with classified input, generated context, and metadata.
        """
        # Step 1: Classify the input
        classification = self._gemini.classify_input(text)
        input_type = classification["input_type"]
        core_thesis = classification["core_thesis"]
        confidence = classification["confidence"]

        logger.info(
            "Input classified as '%s' (confidence=%.2f): %s",
            input_type, confidence, classification.get("reasoning", ""),
        )

        # Step 2: Build ancestry context if exploring (depth > 0)
        ancestry_context = ""
        if ancestry and depth > 0:
            ancestry_context = self._build_ancestry_context(ancestry, depth)

        # Step 3: Route to appropriate context strategy
        if input_type in ("argumentative", "opinion"):
            result = self._gemini_path(text, input_type, core_thesis, ancestry_context)
        elif input_type == "vague":
            result = self._vague_path(text, ancestry_context)
        else:
            # factual → Wikipedia
            result = self._wikipedia_path(text, ancestry_context)

        # Step 4: Set classification metadata
        result.input_type = input_type
        result.core_thesis = core_thesis
        result.classification_confidence = confidence

        # Step 5: Add depth nudge if deep exploration
        if depth >= 6:
            result.depth_nudge = (
                "This is a very deep thread. You might discover new insights "
                "by exploring different assumptions in your original argument."
            )
        elif depth >= 4:
            result.depth_nudge = (
                "You're exploring deeply here. Consider branching to a sibling "
                "question to broaden your perspective."
            )

        return result

    def _build_ancestry_context(self, ancestry: list[dict], depth: int) -> str:
        """Build context from the ancestry chain, summarizing if too deep."""
        if depth <= _SUMMARIZE_DEPTH:
            # Full ancestry fits within token budget
            parts = []
            for node in ancestry:
                role = node["role"].upper()
                parts.append(f"[{role}]: {node['text']}")
            return "\n".join(parts)
        else:
            # Deep exploration: summarize distant ancestry, keep recent in full
            recent = ancestry[-2:] if len(ancestry) >= 2 else ancestry
            distant = ancestry[:-2] if len(ancestry) > 2 else []

            summary = ""
            if distant:
                summary = self._gemini.summarize_ancestry(distant)

            recent_text = "\n".join(
                f"[{n['role'].upper()}]: {n['text']}" for n in recent
            )

            if summary:
                return f"Earlier discussion summary: {summary}\n\nRecent:\n{recent_text}"
            return recent_text

    def _gemini_path(
        self,
        text: str,
        input_type: str,
        core_thesis: str | None,
        ancestry_context: str,
    ) -> ContextResult:
        """Generate context via Gemini for argumentative/opinion inputs."""
        gemini_context = self._gemini.generate_context(text, input_type, core_thesis)

        if not gemini_context:
            # Gemini failed — fall back to Wikipedia
            logger.warning("Gemini context generation failed, falling back to Wikipedia")
            return self._wikipedia_path(text, ancestry_context)

        # Also extract keyphrases for the concept map
        try:
            keyphrases = self._keyphrase.extract(text, top_n=5)
        except Exception:
            keyphrases = [(text.strip(), 1.0)]

        # Combine: ancestry (if any) + Gemini-generated context
        combined = gemini_context
        if ancestry_context:
            combined = f"Conversation so far:\n{ancestry_context}\n\nRelevant context:\n{gemini_context}"

        return ContextResult(
            input_type=input_type,
            core_thesis=core_thesis,
            classification_confidence=0.0,  # Set by caller
            combined_context=combined,
            pipeline_path="gemini",
            keyphrases=keyphrases,
            context_sources=[],
        )

    def _wikipedia_path(self, text: str, ancestry_context: str) -> ContextResult:
        """Generate context via KeyBERT + Wikipedia for factual inputs."""
        # Extract keyphrases
        try:
            keyphrases = self._keyphrase.extract(text, top_n=5)
        except Exception as exc:
            logger.warning("Keyphrase extraction failed: %s", exc)
            keyphrases = [(text.strip(), 1.0)]

        keyphrase_texts = [kp for kp, _ in keyphrases]

        # Wikipedia lookup (top 2 to avoid context bloat)
        context_sources = []
        wiki_context = ""
        try:
            wiki_results = self._wikipedia.retrieve_batch(keyphrase_texts, max_lookups=2)
            context_sources = wiki_results
            wiki_context = " ".join(wr["summary"] for wr in wiki_results)
        except Exception as exc:
            logger.warning("Wikipedia retrieval failed: %s", exc)

        # Combine: ancestry (if any) + Wikipedia context
        combined = wiki_context
        if ancestry_context:
            combined = f"Conversation so far:\n{ancestry_context}\n\nBackground information:\n{wiki_context}"

        return ContextResult(
            input_type="factual",
            core_thesis=None,
            classification_confidence=0.0,
            combined_context=combined,
            pipeline_path="wikipedia",
            keyphrases=keyphrases,
            context_sources=context_sources,
        )

    def _vague_path(self, text: str, ancestry_context: str) -> ContextResult:
        """Handle vague inputs with minimal context."""
        combined = ancestry_context if ancestry_context else ""

        return ContextResult(
            input_type="vague",
            core_thesis=None,
            classification_confidence=0.0,
            combined_context=combined,
            pipeline_path="fallback",
            keyphrases=[(text.strip(), 1.0)] if text.strip() else [],
            context_sources=[],
        )
