from pydantic import BaseModel, Field


VALID_QUESTION_TYPES = [
    "clarity",
    "reasons_evidence",
    "implication_consequences",
    "alternate_viewpoints_perspectives",
    "assumptions",
]


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=2000)
    question_types: list[str] = Field(default=VALID_QUESTION_TYPES)


class Keyphrase(BaseModel):
    text: str
    score: float


class ContextSource(BaseModel):
    keyphrase: str
    summary: str
    url: str | None = None


class Question(BaseModel):
    id: str
    type: str
    text: str
    related_keyphrases: list[str]


class GenerateResponse(BaseModel):
    topic: str
    keyphrases: list[Keyphrase]
    context_sources: list[ContextSource]
    questions: list[Question]
    processing_time_ms: float


class AncestryNode(BaseModel):
    role: str = Field(..., pattern="^(input|question|reflection)$")
    text: str = Field(..., min_length=1)


class ExploreRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    parent_question_id: str | None = None
    ancestry: list[AncestryNode] = Field(default_factory=list)
    depth: int = Field(default=0, ge=0)
    question_types: list[str] = Field(default=VALID_QUESTION_TYPES)


class InputClassification(BaseModel):
    input_type: str          # argumentative | factual | opinion | vague
    core_thesis: str | None = None
    confidence: float
    reasoning: str = ""


class ExploreResponse(BaseModel):
    input_classification: InputClassification
    pipeline_path: str       # "wikipedia" | "gemini" | "fallback"
    keyphrases: list[Keyphrase]
    context_sources: list[ContextSource]
    questions: list[Question]
    depth_nudge: str | None = None
    processing_time_ms: float
