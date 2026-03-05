import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Load .env from project root (one level up from backend/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

from fastapi.middleware.cors import CORSMiddleware

from backend.services.keyphrase import KeyphraseService
from backend.services.wikipedia import WikipediaService
from backend.services.question_gen import QuestionGenerationService
from backend.services.gemini_service import GeminiService
from backend.services.context_router import ContextRouter
from backend.routes import generate, explore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Instantiate services
_keyphrase_service = KeyphraseService()
_wikipedia_service = WikipediaService()
_question_gen_service = QuestionGenerationService()
_gemini_service = GeminiService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models at startup, clean up on shutdown."""
    logger.info("Starting SocraticPath backend...")
    _keyphrase_service.load()
    _question_gen_service.load()
    _gemini_service.load()

    # Build the context router (depends on all three context services)
    _context_router = ContextRouter(
        gemini_service=_gemini_service,
        keyphrase_service=_keyphrase_service,
        wikipedia_service=_wikipedia_service,
    )

    # Inject services into route modules
    generate.keyphrase_service = _keyphrase_service
    generate.wikipedia_service = _wikipedia_service
    generate.question_gen_service = _question_gen_service

    explore.context_router = _context_router
    explore.question_gen_service = _question_gen_service

    logger.info("All services ready.")
    yield
    logger.info("Shutting down SocraticPath backend.")


app = FastAPI(
    title="SocraticPath API",
    description="Generates Socratic questions from user-provided topics using a fine-tuned T5 model.",
    version="1.1.0",
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
app.include_router(explore.router)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _question_gen_service.is_loaded,
        "gemini_loaded": _gemini_service.is_loaded,
    }
