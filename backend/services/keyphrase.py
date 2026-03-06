import logging

import nltk
from nltk import pos_tag, word_tokenize
from keybert import KeyBERT

logger = logging.getLogger(__name__)

# POS tags that indicate nominal (noun-like) content worth looking up
_NOMINAL_TAGS = {"NN", "NNS", "NNP", "NNPS"}

# Tags that indicate junk leading words in multi-word phrases —
_BAD_LEADING_TAGS = {
    "IN",   # preposition ("like", "because", "for")
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  # verbs
    "RB", "RBR", "RBS",  # adverbs
    "PRP", "PRP$",  # pronouns
    "MD",   # modal ("can", "would")
    "CC",   # conjunction ("and", "but")
    "DT",   # determiner ("the", "a")
    "UH",   # interjection
    "EX",   # existential "there"
    "TO",   # "to"
}


def _is_nominal(phrase: str) -> bool:
    """
    Return True if the phrase is a valid nominal keyphrase worth looking up.

    Filters out junk conversational phrases that KeyBERT sometimes extracts from conversational input.

    Rules:
    1. Must contain at least one noun (NN/NNS/NNP/NNPS).
    2. Multi-word phrases must not start with a verb, preposition, or adverb
    """
    try:
        tokens = word_tokenize(phrase)
        tagged = pos_tag(tokens)
    except Exception:
        # If NLTK fails (missing data, etc.), let the phrase through
        return True

    if not tagged:
        return False

    tags = [tag for _, tag in tagged]

    # Must contain at least one noun
    if not any(t in _NOMINAL_TAGS for t in tags):
        return False

    # Multi-word: reject if the first word is a verb/preposition/adverb/etc.
    if len(tagged) > 1 and tags[0] in _BAD_LEADING_TAGS:
        return False

    return True


def find_related_keyphrases(question_text: str, keyphrases: list[str]) -> list[str]:
    """Simple word-overlap heuristic to link questions to keyphrases."""
    q_lower = question_text.lower()
    related = []
    for kp in keyphrases:
        words = kp.lower().split()
        if any(w in q_lower for w in words if len(w) > 2):
            related.append(kp)
    return related if related else keyphrases[:1]


class KeyphraseService:
    """Extracts keyphrases from text using KeyBERT (all-MiniLM-L6-v2)."""

    def __init__(self):
        self.model = None

    def load(self) -> None:
        logger.info("Loading KeyBERT (all-MiniLM-L6-v2)...")
        self.model = KeyBERT(model="all-MiniLM-L6-v2")

        # Ensure NLTK POS tagger data is available
        for resource in ["averaged_perceptron_tagger_eng", "punkt_tab"]:
            try:
                nltk.data.find(f"taggers/{resource}" if "tagger" in resource else f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        logger.info("KeyBERT loaded")

    def extract(self, text: str, top_n: int = 5) -> list[tuple[str, float]]:
        """
        Extract keyphrases with MMR diversity, then filter to nominal phrases.

        Returns list of (phrase, score) tuples.
        Falls back to using the raw text as a single keyphrase if the
        input is too short for meaningful extraction.
        """
        if not text or len(text.strip()) < 10:
            return [(text.strip(), 1.0)] if text and text.strip() else []

        # Ask for more candidates than needed so filtering still yields enough
        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n + 3,
            use_mmr=True,
            diversity=0.5,
        )

        if not keywords:
            return [(text.strip(), 1.0)]

        # Filter to nominal keyphrases only
        filtered = [
            (kw, float(score))
            for kw, score in keywords
            if _is_nominal(kw)
        ]

        if not filtered:
            # All keyphrases were junk — fall back to raw text
            logger.warning("All extracted keyphrases filtered as non-nominal, using raw text")
            return [(text.strip(), 1.0)]

        return filtered[:top_n]
