import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Generation config matching NB06 eval config (deterministic beam search)
GENERATION_CONFIG = dict(
    max_length=80,
    num_beams=4,
    do_sample=False,
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "t5-base-lora" / "merged"

# max_workers=1: one model instance, one inference at a time
_inference_executor = ThreadPoolExecutor(max_workers=1)


class QuestionGenerationService:
    """Loads the merged T5-base + LoRA model and generates Socratic questions."""

    def __init__(self, model_path: str | Path | None = None, use_fp16: bool = False):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._use_fp16 = use_fp16
        # MODEL_PATH env var allows override for deployment (e.g. /app/model)
        env_path = os.getenv("MODEL_PATH")
        if model_path:
            self._model_path = Path(model_path)
        elif env_path:
            self._model_path = Path(env_path)
        else:
            self._model_path = DEFAULT_MODEL_PATH

    def load(self) -> None:
        dtype = torch.float16 if self._use_fp16 else torch.float32
        logger.info(
            "Loading T5-base merged model from %s (dtype=%s) ...",
            self._model_path, dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(self._model_path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self._model_path),
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info(
            "Model loaded — %s params, vocab=%d",
            f"{self.model.num_parameters():,}",
            len(self.tokenizer),
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def generate(
        self,
        user_input: str,
        question_type: str,
        retrieved_context: str = "",
    ) -> str:
        """
        Generate a single Socratic question.

        The prompt is structured to anchor the model on the user's original
        statement. The training prompt format is preserved as the core:
          "Generate a Socratic question for this context: {type}: {text}"

        Retrieved Wikipedia context is appended as clearly-labelled background
        so the model can draw on factual knowledge without drifting away from
        the user's actual argument.
        """
        prompt = (
            f"Generate a Socratic question for this context: "
            f"{question_type}: {user_input}"
        )

        if retrieved_context:
            # Cap context to avoid drowning out the user's input
            max_ctx = min(400, max(100, 500 - len(user_input)))
            prompt += (
                f"\n\nBackground information: {retrieved_context[:max_ctx]}"
            )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **GENERATION_CONFIG)

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.replace("[Question]", "").strip()

    async def generate_async(
        self,
        user_input: str,
        question_type: str,
        retrieved_context: str = "",
    ) -> str:
        """Non-blocking wrapper around generate() for use in async request handlers.

        Offloads the CPU-bound T5 inference to a dedicated single-worker thread
        pool so the asyncio event loop is never blocked during generation.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _inference_executor,
            self.generate,
            user_input,
            question_type,
            retrieved_context,
        )
