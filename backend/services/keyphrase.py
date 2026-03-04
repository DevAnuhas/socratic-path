import logging
from typing import List, Tuple

from keybert import KeyBERT

logger = logging.getLogger(__name__)


class KeyphraseService:
    """Extracts keyphrases from text using KeyBERT (all-MiniLM-L6-v2)."""

    def __init__(self):
        self.model = None

    def load(self) -> None:
        logger.info("Loading KeyBERT (all-MiniLM-L6-v2)...")
        self.model = KeyBERT(model="all-MiniLM-L6-v2")
        logger.info("KeyBERT loaded")

    def extract(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Extract keyphrases with MMR diversity.

        Returns list of (phrase, score) tuples.
        Falls back to using the raw text as a single keyphrase if the
        input is too short for meaningful extraction.
        """
        if not text or len(text.strip()) < 10:
            return [(text.strip(), 1.0)] if text and text.strip() else []

        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n,
            use_mmr=True,
            diversity=0.5,
        )

        if not keywords:
            return [(text.strip(), 1.0)]

        return [(kw, float(score)) for kw, score in keywords]
