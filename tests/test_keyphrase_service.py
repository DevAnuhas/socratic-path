"""
Unit tests for KeyphraseService — POS-tag nominal phrase filtering.

Tests cover:
- _is_nominal(): accepts noun phrases, rejects verb/prep/adverb-led phrases
- KeyphraseService.extract(): fallback on short input, fallback when all filtered
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.services.keyphrase import KeyphraseService, _is_nominal


# ── _is_nominal() ────────────────────────────────────────────


@patch("backend.services.keyphrase.word_tokenize")
@patch("backend.services.keyphrase.pos_tag")
def test_is_nominal_accepts_single_noun(mock_tag, mock_tok):
    mock_tok.return_value = ["inequality"]
    mock_tag.return_value = [("inequality", "NN")]
    assert _is_nominal("inequality") is True


@patch("backend.services.keyphrase.word_tokenize")
@patch("backend.services.keyphrase.pos_tag")
def test_is_nominal_accepts_compound_noun(mock_tag, mock_tok):
    mock_tok.return_value = ["climate", "change"]
    mock_tag.return_value = [("climate", "NN"), ("change", "NN")]
    assert _is_nominal("climate change") is True


@patch("backend.services.keyphrase.word_tokenize")
@patch("backend.services.keyphrase.pos_tag")
def test_is_nominal_accepts_proper_noun(mock_tag, mock_tok):
    mock_tok.return_value = ["United", "Nations"]
    mock_tag.return_value = [("United", "NNP"), ("Nations", "NNPS")]
    assert _is_nominal("United Nations") is True


@patch("backend.services.keyphrase.word_tokenize")
@patch("backend.services.keyphrase.pos_tag")
def test_is_nominal_rejects_verb_phrase(mock_tag, mock_tok):
    """Multi-word phrase starting with a verb is rejected."""
    mock_tok.return_value = ["think", "carefully"]
    mock_tag.return_value = [("think", "VB"), ("carefully", "RB")]
    assert _is_nominal("think carefully") is False


@patch("backend.services.keyphrase.word_tokenize")
@patch("backend.services.keyphrase.pos_tag")
def test_is_nominal_rejects_preposition_led_phrase(mock_tag, mock_tok):
    """Phrase beginning with a preposition is rejected even if it contains a noun."""
    mock_tok.return_value = ["in", "the", "economy"]
    mock_tag.return_value = [("in", "IN"), ("the", "DT"), ("economy", "NN")]
    assert _is_nominal("in the economy") is False


@patch("backend.services.keyphrase.word_tokenize")
@patch("backend.services.keyphrase.pos_tag")
def test_is_nominal_rejects_phrase_without_noun(mock_tag, mock_tok):
    """Phrase with no noun token is rejected regardless of position."""
    mock_tok.return_value = ["does", "not"]
    mock_tag.return_value = [("does", "VBZ"), ("not", "RB")]
    assert _is_nominal("does not") is False


@patch("backend.services.keyphrase.word_tokenize")
@patch("backend.services.keyphrase.pos_tag")
def test_is_nominal_accepts_adjective_noun_compound(mock_tag, mock_tok):
    """Adjective-led phrases that contain a noun are accepted (JJ is not a bad leading tag)."""
    mock_tok.return_value = ["economic", "inequality"]
    mock_tag.return_value = [("economic", "JJ"), ("inequality", "NN")]
    assert _is_nominal("economic inequality") is True


# ── KeyphraseService.extract() ───────────────────────────────


def test_extract_falls_back_on_short_input():
    """Input shorter than 10 characters bypasses KeyBERT and returns raw text."""
    svc = KeyphraseService()
    svc.model = MagicMock()
    result = svc.extract("hi", top_n=5)
    assert result == [("hi", 1.0)]
    svc.model.extract_keywords.assert_not_called()


def test_extract_falls_back_on_empty_input():
    """Empty input returns an empty list."""
    svc = KeyphraseService()
    svc.model = MagicMock()
    result = svc.extract("", top_n=5)
    assert result == []


@patch("backend.services.keyphrase._is_nominal", return_value=False)
def test_extract_falls_back_when_all_filtered(mock_nominal):
    """If every KeyBERT candidate fails the nominal filter, return raw text."""
    svc = KeyphraseService()
    svc.model = MagicMock()
    svc.model.extract_keywords.return_value = [
        ("doing well", 0.7),
        ("going strong", 0.6),
        ("remains unclear", 0.5),
    ]
    result = svc.extract("social media doing well going strong remains unclear", top_n=2)
    assert len(result) == 1
    assert result[0][1] == 1.0  # fallback score is 1.0


def test_extract_requests_extra_candidates_for_filtering():
    """extract() requests top_n + 3 candidates from KeyBERT to allow for filtering."""
    svc = KeyphraseService()
    svc.model = MagicMock()
    svc.model.extract_keywords.return_value = []

    with patch("backend.services.keyphrase._is_nominal", return_value=True):
        svc.extract("photosynthesis converts sunlight into glucose", top_n=3)

    call_kwargs = svc.model.extract_keywords.call_args.kwargs
    assert call_kwargs.get("top_n") == 6  # top_n + 3 = 3 + 3
