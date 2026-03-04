import logging
from typing import List, Optional, TypedDict

import wikipediaapi

logger = logging.getLogger(__name__)


class WikiResult(TypedDict):
    keyphrase: str
    summary: str
    url: Optional[str]


class WikipediaService:
    """Retrieves factual summaries from Wikipedia for keyphrases."""

    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            user_agent="SocraticPath/1.0 (dissertation; contact: anuhas0123@gmail.com)",
            language="en",
        )

    def lookup(self, keyphrase: str) -> Optional[WikiResult]:
        """
        Fetch a short Wikipedia summary for a single keyphrase.

        Returns None if no page is found or the request fails.
        Truncates to ~3 sentences (up to 400 chars at a sentence boundary).
        """
        try:
            page = self.wiki.page(keyphrase)
            if not page.exists():
                return None

            summary = page.summary[:400]
            last_period = summary.rfind(".")
            if last_period > 150:
                summary = summary[: last_period + 1]

            return WikiResult(
                keyphrase=keyphrase,
                summary=summary,
                url=page.fullurl,
            )
        except Exception as exc:
            logger.warning("Wikipedia lookup failed for '%s': %s", keyphrase, exc)
            return None

    def retrieve_batch(
        self, keyphrases: List[str], max_lookups: int = 3
    ) -> List[WikiResult]:
        """
        Look up Wikipedia context for up to `max_lookups` keyphrases.

        Limits lookups to keep prompt length manageable and response time low.
        """
        results: List[WikiResult] = []
        for kp in keyphrases[:max_lookups]:
            result = self.lookup(kp)
            if result:
                results.append(result)
        return results
