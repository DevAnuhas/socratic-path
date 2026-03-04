# SocraticPath

AI-powered Socratic question generation using fine-tuned T5 models with LoRA adapters. Given a topic or argument, the system extracts keyphrases, retrieves Wikipedia context, and generates probing questions across five categories of Socratic inquiry.

**Author:** Anuhas Dissanayake
**Student ID:** CL/BSCSD/32/122

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Model Fine-Tuning](#model-fine-tuning)
- [Evaluation Results](#evaluation-results)
- [Backend](#backend)
- [Frontend](#frontend)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [Tech Stack](#tech-stack)

---

## Overview

SocraticPath implements an end-to-end pipeline for automatic Socratic question generation:

1. **Keyphrase Extraction** — KeyBERT (with all-MiniLM-L6-v2 embeddings) extracts the most salient keyphrases from the user's input
2. **Context Retrieval** — Wikipedia API fetches relevant article summaries for each keyphrase
3. **Question Generation** — A fine-tuned T5-base model with LoRA adapters generates questions across five Socratic categories
4. **Visualisation** — An interactive concept map and colour-coded question cards present the results

### Socratic Question Types

Based on Paul & Elder's taxonomy of Socratic questioning:

| Type                            | Description                                    |
| ------------------------------- | ---------------------------------------------- |
| **Clarity**                     | Questions that probe meaning and understanding |
| **Reasons & Evidence**          | Questions that challenge the basis of claims   |
| **Implications & Consequences** | Questions exploring outcomes and effects       |
| **Alternate Viewpoints**        | Questions introducing different perspectives   |
| **Assumptions**                 | Questions that surface hidden presuppositions  |

---

## Architecture

```
User Input (topic/argument)
        │
        ▼
┌─────────────────┐
│     KeyBERT     │  Keyphrase extraction (top-5, MMR diversity)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Wikipedia API  │  Context retrieval (up to 3 articles)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  T5-base + LoRA │  Question generation (5 types × beam search)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Next.js UI   │  Concept map + question cards + source panel
└─────────────────┘
```

---

## Repository Structure

```
socratic-path/
├── backend/                    # FastAPI REST API
│   ├── main.py                 # App entrypoint, CORS, lifespan
│   ├── routes/
│   │   └── generate.py         # POST /api/generate endpoint
│   ├── schemas/
│   │   └── models.py           # Pydantic request/response models
│   └── services/
│       ├── keyphrase.py        # KeyBERT keyphrase extraction
│       ├── wikipedia.py        # Wikipedia context retrieval
│       └── question_gen.py     # T5 model loading & inference
│
├── frontend/                   # Next.js 16 (App Router)
│   ├── app/
│   │   ├── layout.tsx          # Root layout with metadata
│   │   ├── page.tsx            # Main page composing all components
│   │   └── globals.css         # Tailwind v4 + custom theme
│   ├── components/
│   │   ├── InputPanel.tsx      # Topic input + type toggles + generate
│   │   ├── QuestionCards.tsx   # Colour-coded question display
│   │   ├── SourcePanel.tsx     # Wikipedia sources sidebar
│   │   ├── LoadingState.tsx    # Skeleton loading animation
│   │   ├── ConceptMap.tsx      # React Flow concept map
│   │   └── nodes/
│   │       ├── TopicNode.tsx   # Central topic node
│   │       └── KeyphraseNode.tsx # Keyphrase satellite nodes
│   └── lib/
│       ├── types.ts            # TypeScript types + question type config
│       ├── api.ts              # Axios API client
│       ├── store.ts            # Zustand state management
│       └── utils.ts            # cn() class merge utility
│
├── notebooks/                  # Jupyter notebooks (research pipeline)
│   ├── 01_data_inspection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_keyphrase_extraction.ipynb
│   └── 06_inference_pipeline.ipynb
│
├── scripts/                    # Training & evaluation scripts
│   ├── train_model.py          # LoRA fine-tuning script
│   ├── evaluate_model.py       # Model evaluation (ROUGE, BERTScore, BLEU)
│   ├── plot_training_curves.py # Training visualisation
│   └── run_all_training.sh     # Batch training runner
│
├── models/                     # Trained model artifacts (git-ignored)
│   ├── flan-t5-small-lora/     # FLAN-T5-small + LoRA (78M params)
│   ├── flan-t5-base-lora/      # FLAN-T5-base + LoRA (251M params)
│   └── t5-base-lora/           # T5-base + LoRA (226M params) ← best
│       ├── adapter/            # LoRA adapter weights
│       ├── checkpoints/        # Training checkpoints
│       ├── logs/               # TensorBoard logs
│       └── merged/             # Merged model for inference
│
├── datasets/                   # Dataset files (git-ignored)
│   ├── raw/                    # Original SocratiQ dataset
│   └── processed/              # Tokenised + split data
│
├── evaluation_results/         # Evaluation outputs & plots
│
├── pyproject.toml              # Python dependencies (uv)
└── uv.lock                     # Dependency lock file
```

---

## Model Fine-Tuning

### Dataset

The [SocratiQ dataset](https://huggingface.co/datasets/GWU-NLP/SocratiQ) (Ang et al., EACL 2023) contains Socratic questions paired with Wikipedia context passages, spanning five question types.

| Split      | Samples |
| ---------- | ------- |
| Train      | 84,582  |
| Validation | 10,573  |
| Test       | 10,573  |

### Training Configuration

Three models were fine-tuned using LoRA (Hu et al., 2022) on an AWS EC2 g5.2xlarge instance (NVIDIA A10G, 24GB VRAM):

| Parameter               | Value                        |
| ----------------------- | ---------------------------- |
| LoRA rank (r)           | 16                           |
| LoRA alpha              | 32                           |
| LoRA dropout            | 0.1                          |
| Target modules          | q, v (attention projections) |
| Learning rate           | 3e-4                         |
| Scheduler               | Cosine with warmup           |
| Warmup ratio            | 0.06                         |
| Epochs                  | 10                           |
| Batch size              | 16                           |
| Early stopping patience | 5                            |
| Precision               | bf16 (CUDA)                  |

### Prompt Format

```
Generate a Socratic question for this context: {question_type}: {input_text}
```

---

## Evaluation Results

All models evaluated on the held-out test set (10,573 samples):

| Model               | ROUGE-1   | ROUGE-2   | ROUGE-L   | BLEU-4    | BERTScore F1 |
| ------------------- | --------- | --------- | --------- | --------- | ------------ |
| FLAN-T5-small (78M) | 0.222     | 0.052     | 0.205     | 0.026     | 0.892        |
| FLAN-T5-base (251M) | 0.232     | 0.059     | 0.215     | 0.029     | 0.894        |
| **T5-base (226M)**  | **0.235** | **0.060** | **0.217** | **0.029** | **0.895**    |

### Comparison with Paper Baselines (Ang et al., 2023)

| Model                      | ROUGE-L   | BERTScore |
| -------------------------- | --------- | --------- |
| T5-p (paper, prompt-based) | 0.211     | 0.632     |
| ProphetNet-p (paper)       | 0.208     | 0.632     |
| GPT-p (paper)              | 0.187     | 0.615     |
| **T5-base + LoRA (ours)**  | **0.217** | **0.895** |

Our best model (T5-base + LoRA) exceeds the paper's T5-p baseline by **+0.006 ROUGE-L** and **+0.263 BERTScore F1**, demonstrating the effectiveness of parameter-efficient fine-tuning over prompt-based approaches.

### Key Finding

Standard T5-base outperforms instruction-tuned FLAN-T5 variants, suggesting that FLAN's instruction tuning provides no advantage when paired with task-specific LoRA fine-tuning.

---

## Backend

**Framework:** FastAPI

### API Endpoints

| Method | Path            | Description                             |
| ------ | --------------- | --------------------------------------- |
| POST   | `/api/generate` | Generate Socratic questions for a topic |
| GET    | `/health`       | Health check (model loaded status)      |

### POST /api/generate

**Request:**

```json
{
	"topic": "Climate change effects on agriculture",
	"question_types": [
		"clarity",
		"reasons_evidence",
		"implication_consequences",
		"alternate_viewpoints_perspectives",
		"assumptions"
	]
}
```

**Response:**

```json
{
	"keyphrases": [
		{ "text": "climate change", "score": 0.72 },
		{ "text": "agriculture", "score": 0.65 }
	],
	"questions": [
		{
			"question_type": "clarity",
			"text": "What do you mean by the effects of climate change on agriculture?",
			"related_keyphrases": ["climate change", "agriculture"]
		}
	],
	"sources": [
		{
			"keyphrase": "climate change",
			"title": "Climate change",
			"summary": "Climate change includes both human-driven global warming...",
			"url": "https://en.wikipedia.org/wiki/Climate_change"
		}
	],
	"processing_time_ms": 7823
}
```

### Services

- **KeyphraseService** — KeyBERT with all-MiniLM-L6-v2, MMR diversity (top-5 keyphrases)
- **WikipediaService** — Wikipedia API with summary truncation (~400 chars)
- **QuestionGenerationService** — Loads merged T5-base model, beam search (num_beams=4, max_length=80)

---

## Frontend

**Framework:** Next.js 16 (App Router) with TypeScript

### Components

- **InputPanel** — Topic textarea with Cmd+Enter submit, colour-coded question type toggle chips, generate/reset buttons
- **ConceptMap** — Interactive React Flow visualisation with radial layout. Central topic node surrounded by keyphrase nodes. Click a keyphrase to filter questions; hover to explore (re-generate with that keyphrase as topic)
- **QuestionCards** — Colour-coded cards grouped by Socratic question type with filtering support
- **SourcePanel** — Wikipedia source cards with clickable keyphrase tags and external links
- **LoadingState** — Shimmer skeleton placeholders matching the results layout

### State Management

Zustand store manages: topic, selected question types, keyphrases, questions, sources, loading state, selected keyphrase filter, and generation status.

---

## Setup & Installation

### Prerequisites

- Python 3.10+ with [uv](https://docs.astral.sh/uv/) package manager
- Node.js 18+ with npm
- Trained model artifacts in `models/t5-base-lora/merged/`

### Backend

```bash
# Install Python dependencies
uv sync

# Start the backend server
uv run uvicorn backend.main:app --reload --port 8000
```

The backend loads the T5 model on startup (~10-15s). Health check at `http://localhost:8000/health`.

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

Frontend runs at `http://localhost:3000` and proxies API calls to `http://localhost:8000`.

---

## Running the Application

1. Start the backend: `uv run uvicorn backend.main:app --reload --port 8000`
2. Start the frontend: `cd frontend && npm run dev`
3. Open `http://localhost:3000` in your browser
4. Enter a topic and click Generate

Typical inference time is 5-11 seconds on CPU (keyphrase extraction + Wikipedia retrieval + model inference).

---

## Tech Stack

| Layer                   | Technology                                   |
| ----------------------- | -------------------------------------------- |
| Model                   | T5-base + LoRA (PyTorch, Transformers, PEFT) |
| Keyphrase Extraction    | KeyBERT + sentence-transformers              |
| Context Retrieval       | Wikipedia API                                |
| Backend                 | FastAPI, Pydantic, Uvicorn                   |
| Frontend                | Next.js 16, React 19, TypeScript             |
| UI                      | Tailwind CSS v4, Lucide React icons          |
| Concept Map             | React Flow (@xyflow/react)                   |
| State Management        | Zustand                                      |
| Training Infrastructure | AWS EC2 g5.2xlarge (NVIDIA A10G)             |
| Package Management      | uv (Python), npm (Node.js)                   |

---

## References

- Ang, L. et al. (2023). _SocratiQ: A dataset for Socratic question generation_. EACL.
- Hu, E.J. et al. (2022). _LoRA: Low-Rank Adaptation of Large Language Models_. ICLR.
- Raffel, C. et al. (2020). _Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer_. JMLR.
- Wei, J. et al. (2022). _Finetuned Language Models Are Zero-Shot Learners_. ICLR.
- Lin, C.Y. (2004). _ROUGE: A Package for Automatic Evaluation of Summaries_. ACL Workshop.
- Zhang, T. et al. (2020). _BERTScore: Evaluating Text Generation with BERT_. ICLR.
