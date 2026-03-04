from pydantic import BaseModel, Field
from typing import List, Optional


VALID_QUESTION_TYPES = [
    "clarity",
    "reasons_evidence",
    "implication_consequences",
    "alternate_viewpoints_perspectives",
    "assumptions",
]


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=2000)
    question_types: List[str] = Field(default=VALID_QUESTION_TYPES)


class Keyphrase(BaseModel):
    text: str
    score: float


class ContextSource(BaseModel):
    keyphrase: str
    summary: str
    url: Optional[str] = None


class Question(BaseModel):
    id: str
    type: str
    text: str
    related_keyphrases: List[str]


class GenerateResponse(BaseModel):
    topic: str
    keyphrases: List[Keyphrase]
    context_sources: List[ContextSource]
    questions: List[Question]
    processing_time_ms: float
