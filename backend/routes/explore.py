"""
POST /api/explore — Recursive Socratic exploration endpoint.

Handles both initial topic submission and follow-up exploration (reflections
on generated questions). Uses the ContextRouter to intelligently select
between Wikipedia and Gemini-generated context based on input type.

This endpoint powers the branching exploration graph in the frontend.
"""

import time
import logging
from fastapi import APIRouter, Depends, HTTPException

from backend.auth import get_current_user
from backend.schemas.models import (
    VALID_QUESTION_TYPES,
    ExploreRequest,
    ExploreResponse,
    InputClassification,
    Keyphrase,
    ContextSource,
    Question,
)
from backend.services.context_router import ContextRouter
from backend.services.keyphrase import find_related_keyphrases
from backend.services.question_gen import QuestionGenerationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Services injected from main.py at startup
context_router: ContextRouter = None
question_gen_service: QuestionGenerationService = None


@router.post("/explore", response_model=ExploreResponse)
async def explore(
    request: ExploreRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Generate Socratic questions with intelligent context routing.

    Supports both initial exploration (depth=0) and recursive follow-up
    exploration (depth>0 with ancestry chain for context propagation).
    """
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

    # Step 1: Route input through the smart context pipeline
    ancestry_dicts = [{"role": a.role, "text": a.text} for a in request.ancestry]

    if not context_router:
        raise HTTPException(
            status_code=503,
            detail="Context router not initialized",
        )

    ctx = context_router.route(
        text=request.text,
        ancestry=ancestry_dicts if ancestry_dicts else None,
        depth=request.depth,
    )

    logger.info(
        "Context routed — type=%s, path=%s, depth=%d, context_len=%d",
        ctx.input_type, ctx.pipeline_path, request.depth, len(ctx.combined_context),
    )

    # Step 2: Generate one question per requested type using T5
    keyphrase_texts = [kp for kp, _ in ctx.keyphrases] if ctx.keyphrases else []
    questions: list[Question] = []

    for i, qtype in enumerate(types):
        try:
            text = question_gen_service.generate(
                user_input=request.text,
                question_type=qtype,
                retrieved_context=ctx.combined_context,
            )
        except Exception as exc:
            logger.error("Question generation failed for type '%s': %s", qtype, exc)
            raise HTTPException(
                status_code=500,
                detail=f"Question generation failed for type '{qtype}'. Please try again.",
            )

        related = find_related_keyphrases(text, keyphrase_texts) if keyphrase_texts else []
        questions.append(
            Question(
                id=f"q{request.depth}_{i + 1}",
                type=qtype,
                text=text,
                related_keyphrases=related,
            )
        )

    elapsed_ms = (time.time() - t0) * 1000
    logger.info(
        "Exploration complete — %d questions in %.0fms (path=%s, depth=%d)",
        len(questions), elapsed_ms, ctx.pipeline_path, request.depth,
    )

    # Build response
    keyphrases_out = [
        Keyphrase(text=kp, score=score) for kp, score in ctx.keyphrases
    ]

    context_sources_out = [
        ContextSource(
            keyphrase=src["keyphrase"],
            summary=src["summary"],
            url=src.get("url"),
        )
        for src in ctx.context_sources
    ]

    return ExploreResponse(
        input_classification=InputClassification(
            input_type=ctx.input_type,
            core_thesis=ctx.core_thesis,
            confidence=ctx.classification_confidence,
        ),
        pipeline_path=ctx.pipeline_path,
        keyphrases=keyphrases_out,
        context_sources=context_sources_out,
        questions=questions,
        depth_nudge=ctx.depth_nudge,
        processing_time_ms=round(elapsed_ms, 1),
    )
