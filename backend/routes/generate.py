import time
import uuid
import logging
from typing import List

from fastapi import APIRouter, HTTPException

from backend.schemas.models import (
    VALID_QUESTION_TYPES,
    GenerateRequest,
    GenerateResponse,
    Keyphrase,
    ContextSource,
    Question,
)
from backend.services.keyphrase import KeyphraseService
from backend.services.wikipedia import WikipediaService
from backend.services.question_gen import QuestionGenerationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Services are injected from main.py at startup
keyphrase_service: KeyphraseService = None
wikipedia_service: WikipediaService = None
question_gen_service: QuestionGenerationService = None


def _find_related_keyphrases(question_text: str, keyphrases: List[str]) -> List[str]:
    """Simple word-overlap heuristic to link questions to keyphrases."""
    q_lower = question_text.lower()
    related = []
    for kp in keyphrases:
        words = kp.lower().split()
        if any(w in q_lower for w in words if len(w) > 2):
            related.append(kp)
    return related if related else keyphrases[:1]


@router.post("/generate", response_model=GenerateResponse)
async def generate_questions(request: GenerateRequest):
    if not question_gen_service or not question_gen_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Validate question types
    types = [t for t in request.question_types if t in VALID_QUESTION_TYPES]
    if not types:
        raise HTTPException(
            status_code=400,
            detail=f"No valid question types. Choose from: {VALID_QUESTION_TYPES}",
        )

    t0 = time.time()

    # Step 1: Extract keyphrases (graceful fallback to raw topic)
    try:
        raw_keyphrases = keyphrase_service.extract(request.topic, top_n=5)
    except Exception as exc:
        logger.warning("Keyphrase extraction failed, using raw topic: %s", exc)
        raw_keyphrases = [(request.topic.strip(), 1.0)]

    keyphrases = [Keyphrase(text=kp, score=score) for kp, score in raw_keyphrases]
    keyphrase_texts = [kp.text for kp in keyphrases]

    # Step 2: Retrieve Wikipedia context (non-critical — degrade gracefully)
    context_sources: List[ContextSource] = []
    combined_context = ""
    try:
        wiki_results = wikipedia_service.retrieve_batch(keyphrase_texts, max_lookups=3)
        context_sources = [
            ContextSource(keyphrase=wr["keyphrase"], summary=wr["summary"], url=wr["url"])
            for wr in wiki_results
        ]
        combined_context = " ".join(wr["summary"] for wr in wiki_results)
    except Exception as exc:
        logger.warning("Wikipedia retrieval failed, proceeding without context: %s", exc)

    # Step 3: Generate one question per requested type
    questions: List[Question] = []
    for i, qtype in enumerate(types):
        try:
            text = question_gen_service.generate(
                user_input=request.topic,
                question_type=qtype,
                retrieved_context=combined_context,
            )
        except Exception as exc:
            logger.error("Question generation failed for type '%s': %s", qtype, exc)
            raise HTTPException(
                status_code=500,
                detail=f"Question generation failed for type '{qtype}'. Please try again.",
            )

        related = _find_related_keyphrases(text, keyphrase_texts)
        questions.append(
            Question(
                id=f"q{i + 1}",
                type=qtype,
                text=text,
                related_keyphrases=related,
            )
        )

    elapsed_ms = (time.time() - t0) * 1000
    logger.info("Generated %d questions in %.0fms", len(questions), elapsed_ms)

    return GenerateResponse(
        topic=request.topic,
        keyphrases=keyphrases,
        context_sources=context_sources,
        questions=questions,
        processing_time_ms=round(elapsed_ms, 1),
    )
