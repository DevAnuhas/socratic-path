"""
Unit tests for ContextRouter — smart input routing and ancestry context propagation.

Tests cover:
- Argumentative input routes to the Gemini context path
- Factual input routes to the Wikipedia context path
- Gemini failure triggers automatic fallback to Wikipedia
- Vague input returns fallback path without external API calls
- Shallow ancestry (depth <= 3) is included verbatim in the combined context
- Deep ancestry (depth > 3) triggers Gemini summarisation, preserving recent nodes
- Depth nudges are surfaced at the correct thresholds
"""

from unittest.mock import MagicMock, call

import pytest

from backend.services.context_router import ContextRouter, _SUMMARIZE_DEPTH


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def mock_gemini():
    svc = MagicMock()
    svc.classify_input.return_value = {
        "input_type": "argumentative",
        "core_thesis": "Social media is net harmful",
        "confidence": 0.9,
        "reasoning": "Contains a normative claim",
    }
    svc.generate_context.return_value = "Counter-arguments: privacy concerns, addiction..."
    svc.summarize_ancestry.return_value = "Earlier: user discussed mental health impacts."
    return svc


@pytest.fixture
def mock_keyphrase():
    svc = MagicMock()
    svc.extract.return_value = [("social media", 0.8), ("mental health", 0.7)]
    return svc


@pytest.fixture
def mock_wikipedia():
    svc = MagicMock()
    svc.retrieve_batch.return_value = [
        {
            "keyphrase": "social media",
            "summary": "Social media are interactive technologies...",
            "url": "https://en.wikipedia.org/wiki/Social_media",
        }
    ]
    return svc


@pytest.fixture
def router(mock_gemini, mock_keyphrase, mock_wikipedia):
    return ContextRouter(
        gemini_service=mock_gemini,
        keyphrase_service=mock_keyphrase,
        wikipedia_service=mock_wikipedia,
    )


# ── Routing decisions ────────────────────────────────────────


def test_argumentative_input_routes_to_gemini(router, mock_gemini):
    result = router.route("I think social media does more harm than good")
    assert result.pipeline_path == "gemini"
    mock_gemini.generate_context.assert_called_once()


def test_opinion_input_routes_to_gemini(router, mock_gemini):
    mock_gemini.classify_input.return_value = {
        "input_type": "opinion",
        "core_thesis": "Remote work is better",
        "confidence": 0.85,
        "reasoning": "Subjective preference stated",
    }
    result = router.route("I believe remote work is always more productive")
    assert result.pipeline_path == "gemini"
    mock_gemini.generate_context.assert_called_once()


def test_factual_input_routes_to_wikipedia(router, mock_gemini, mock_wikipedia):
    mock_gemini.classify_input.return_value = {
        "input_type": "factual",
        "core_thesis": None,
        "confidence": 0.88,
        "reasoning": "Descriptive claim about a natural process",
    }
    result = router.route("What causes the greenhouse effect?")
    assert result.pipeline_path == "wikipedia"
    mock_wikipedia.retrieve_batch.assert_called_once()


def test_gemini_failure_falls_back_to_wikipedia(router, mock_gemini, mock_wikipedia):
    """When GeminiService.generate_context returns None, the router falls back to Wikipedia."""
    mock_gemini.generate_context.return_value = None
    result = router.route("I believe social media is harmful")
    assert result.pipeline_path == "wikipedia"
    mock_wikipedia.retrieve_batch.assert_called_once()


def test_vague_input_returns_fallback_path(router, mock_gemini):
    mock_gemini.classify_input.return_value = {
        "input_type": "vague",
        "core_thesis": None,
        "confidence": 0.3,
        "reasoning": "Input lacks sufficient content",
    }
    result = router.route("hmm")
    assert result.pipeline_path == "fallback"
    mock_gemini.generate_context.assert_not_called()


# ── Ancestry context propagation ─────────────────────────────


def test_shallow_ancestry_included_verbatim(router):
    """At depth <= _SUMMARIZE_DEPTH, the full ancestry chain is embedded in the context."""
    ancestry = [
        {"role": "input", "text": "Tell me about climate change"},
        {"role": "question", "text": "What evidence supports this claim?"},
    ]
    result = router.route("I think it is urgent", ancestry=ancestry, depth=2)
    assert "[INPUT]: Tell me about climate change" in result.combined_context
    assert "[QUESTION]: What evidence supports this claim?" in result.combined_context


def test_deep_ancestry_triggers_gemini_summarisation(router, mock_gemini):
    """At depth > _SUMMARIZE_DEPTH, distant nodes are compressed via Gemini summarise_ancestry."""
    ancestry = [
        {"role": "input", "text": "Initial topic"},
        {"role": "question", "text": "Q1"},
        {"role": "reflection", "text": "R1 (distant)"},
        {"role": "question", "text": "Q2 (recent)"},
        {"role": "reflection", "text": "R2 (recent)"},
    ]
    router.route("Follow-up thought", ancestry=ancestry, depth=_SUMMARIZE_DEPTH + 1)
    mock_gemini.summarize_ancestry.assert_called_once()


def test_deep_ancestry_preserves_recent_nodes_verbatim(router, mock_gemini):
    """The two most recent ancestry nodes are always included verbatim, never summarised."""
    ancestry = [
        {"role": "input", "text": "Initial topic"},
        {"role": "question", "text": "Q1"},
        {"role": "reflection", "text": "Distant reflection"},
        {"role": "question", "text": "Most recent question"},
        {"role": "reflection", "text": "Most recent reflection"},
    ]
    result = router.route("New thought", ancestry=ancestry, depth=_SUMMARIZE_DEPTH + 1)
    assert "Most recent question" in result.combined_context
    assert "Most recent reflection" in result.combined_context


def test_depth_zero_skips_ancestry_building(router, mock_gemini):
    """At depth 0 (initial input), ancestry context is empty and not requested."""
    result = router.route("A fresh topic", ancestry=None, depth=0)
    mock_gemini.summarize_ancestry.assert_not_called()


# ── Depth nudges ─────────────────────────────────────────────


def test_no_depth_nudge_at_shallow_depth(router):
    result = router.route("Basic topic", depth=1)
    assert result.depth_nudge is None


def test_depth_nudge_appears_at_depth_4(router):
    result = router.route("Deep topic", depth=4)
    assert result.depth_nudge is not None


def test_depth_nudge_appears_at_depth_6(router):
    result = router.route("Very deep topic", depth=6)
    assert result.depth_nudge is not None


# ── Classification metadata propagated to result ─────────────


def test_classification_metadata_in_result(router, mock_gemini):
    mock_gemini.classify_input.return_value = {
        "input_type": "argumentative",
        "core_thesis": "AI replaces jobs",
        "confidence": 0.92,
        "reasoning": "Strong causal claim",
    }
    result = router.route("AI will definitely replace most jobs")
    assert result.input_type == "argumentative"
    assert result.classification_confidence == pytest.approx(0.92)
