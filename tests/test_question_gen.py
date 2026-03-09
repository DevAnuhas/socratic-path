"""
Unit tests for QuestionGenerationService (backend/services/question_gen.py).

The T5 model weights are never loaded. Tests inject mock tokenizer and model
objects directly onto the service instance, isolating the prompt construction
logic, context truncation, output post-processing, and async dispatch from the
actual inference path.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.question_gen import QuestionGenerationService


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_service(decoded_output: str = "What is your evidence?") -> QuestionGenerationService:
    """Return a QuestionGenerationService with mock model and tokenizer injected.

    The tokenizer mock records the prompt it receives and returns a dict of
    MagicMock tensors (each with a .to() method). The model mock returns a
    MagicMock from .generate() whose [0] element is passed to tokenizer.decode().
    """
    svc = QuestionGenerationService()

    mock_tokenizer = MagicMock()
    # tokenizer(prompt, ...) → dict of mock tensors; each .to(device) is a MagicMock
    mock_tokenizer.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }
    mock_tokenizer.decode.return_value = decoded_output

    mock_model = MagicMock()
    # model.generate(...) → subscriptable mock so outputs[0] works
    mock_model.generate.return_value = MagicMock()
    mock_model.num_parameters.return_value = 226_000_000

    svc.tokenizer = mock_tokenizer
    svc.model = mock_model
    return svc


# ── is_loaded ─────────────────────────────────────────────────────────────────


def test_is_loaded_false_before_model_set():
    svc = QuestionGenerationService()
    assert svc.is_loaded is False


def test_is_loaded_true_after_model_injected():
    svc = _make_service()
    assert svc.is_loaded is True


# ── Prompt construction ───────────────────────────────────────────────────────


def test_prompt_contains_question_type_and_user_input():
    svc = _make_service()
    svc.generate("Social media is harmful", "reasons_evidence")
    prompt = svc.tokenizer.call_args[0][0]
    assert "reasons_evidence: Social media is harmful" in prompt


def test_prompt_prefixed_with_generate_instruction():
    svc = _make_service()
    svc.generate("Taxes should be higher", "clarity")
    prompt = svc.tokenizer.call_args[0][0]
    assert prompt.startswith("Generate a Socratic question for this context:")


def test_no_context_omits_background_section():
    svc = _make_service()
    svc.generate("A claim", "clarity", retrieved_context="")
    prompt = svc.tokenizer.call_args[0][0]
    assert "Background information" not in prompt


def test_context_appended_when_provided():
    svc = _make_service()
    svc.generate("A claim", "clarity", retrieved_context="Some wiki text")
    prompt = svc.tokenizer.call_args[0][0]
    assert "Background information: Some wiki text" in prompt


# ── Context truncation ────────────────────────────────────────────────────────


def test_long_context_is_truncated():
    """
    With user_input of 100 chars: max_ctx = min(400, max(100, 500 - 100)) = 400.
    Only the first 400 chars of a 500-char context string should appear.
    """
    svc = _make_service()
    user_input = "x" * 100
    long_context = "y" * 500
    svc.generate(user_input, "clarity", retrieved_context=long_context)
    prompt = svc.tokenizer.call_args[0][0]
    assert "y" * 400 in prompt
    assert "y" * 401 not in prompt


def test_very_short_user_input_caps_context_at_400():
    """
    With user_input of 10 chars: max_ctx = min(400, max(100, 490)) = 400.
    Context is still capped at 400 even when user_input is short.
    """
    svc = _make_service()
    user_input = "x" * 10
    long_context = "y" * 500
    svc.generate(user_input, "clarity", retrieved_context=long_context)
    prompt = svc.tokenizer.call_args[0][0]
    assert "y" * 400 in prompt
    assert "y" * 401 not in prompt


# ── Output post-processing ────────────────────────────────────────────────────


def test_question_prefix_stripped_from_output():
    svc = _make_service(decoded_output="[Question] What is your evidence?")
    result = svc.generate("A claim", "reasons_evidence")
    assert result == "What is your evidence?"


def test_output_is_stripped_of_whitespace():
    svc = _make_service(decoded_output="  Why do you think that?  ")
    result = svc.generate("A claim", "clarity")
    assert result == "Why do you think that?"


def test_clean_output_returned_unchanged():
    svc = _make_service(decoded_output="What do you mean by that?")
    result = svc.generate("A claim", "clarity")
    assert result == "What do you mean by that?"


# ── Async dispatch ────────────────────────────────────────────────────────────


async def test_generate_async_returns_same_output_as_generate():
    svc = _make_service(decoded_output="Why do you think that?")
    result = await svc.generate_async("A claim", "clarity")
    assert result == "Why do you think that?"


async def test_generate_async_passes_context_to_generate():
    svc = _make_service()
    await svc.generate_async("A claim", "clarity", retrieved_context="Background text")
    prompt = svc.tokenizer.call_args[0][0]
    assert "Background information: Background text" in prompt
