import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.services.keyphrase import KeyphraseService
from backend.services.wikipedia import WikipediaService
from backend.services.question_gen import QuestionGenerationService
from backend.routes import generate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Instantiate services
_keyphrase_service = KeyphraseService()
_wikipedia_service = WikipediaService()
_question_gen_service = QuestionGenerationService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models at startup, clean up on shutdown."""
    logger.info("Starting SocraticPath backend...")
    _keyphrase_service.load()
    _question_gen_service.load()

    # Inject services into the route module
    generate.keyphrase_service = _keyphrase_service
    generate.wikipedia_service = _wikipedia_service
    generate.question_gen_service = _question_gen_service

    logger.info("All services ready.")
    yield
    logger.info("Shutting down SocraticPath backend.")


app = FastAPI(
    title="SocraticPath API",
    description="Generates Socratic questions from user-provided topics using a fine-tuned T5 model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate.router)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _question_gen_service.is_loaded,
    }
