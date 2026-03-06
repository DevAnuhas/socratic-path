# SocraticPath

AI-powered Socratic question generation combining fine-tuned T5 models with intelligent context routing via Gemini 2.5 Flash. Users explore topics through a branching dialogue tree: the system classifies input, retrieves or generates context, and produces probing questions across five categories of Socratic inquiry.

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
- [Docker Deployment](#docker-deployment)
- [Tech Stack](#tech-stack)
- [References](#references)

---

## Overview

SocraticPath implements a recursive, branching exploration pipeline for Socratic question generation:

1. **Input Classification** — Gemini 2.5 Flash classifies the user's input as argumentative, factual, opinion, or vague
2. **Smart Context Routing** — Factual inputs go through KeyBERT + Wikipedia; argumentative/opinion inputs get Gemini-generated counter-argument context
3. **Keyphrase Extraction** — KeyBERT (with all-MiniLM-L6-v2 embeddings) extracts salient nominal keyphrases, filtered by POS tagging
4. **Question Generation** — A fine-tuned T5-base model with LoRA adapters generates questions across five Socratic categories
5. **Branching Exploration** — Users reflect on generated questions to trigger deeper exploration, building a tree of inquiry
6. **Ancestry Context Propagation** — The full branch history is sent to T5 for context-aware follow-up generation (summarised by Gemini at depth > 3 to fit T5's 512-token limit)
7. **Persistence** — Exploration trees are saved to Supabase and can be resumed or exported

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
User Input (topic / reflection)
        |
        v
+-------------------+
|   Gemini 2.5      |  Input classification
|   Flash           |  (argumentative | factual | opinion | vague)
+--------+----------+
         |
    +----+----+
    |         |
    v         v
+-------+  +----------+
| Wiki  |  |  Gemini  |   Context generation
| Path  |  |  Path    |   (factual vs argumentative)
+---+---+  +----+-----+
    |            |
    +-----+------+
          |
          v
+-------------------+
|  T5-base + LoRA   |  Socratic question generation
|  (fine-tuned)     |  (5 types x beam search)
+--------+----------+
         |
         v
+-------------------+
|    Next.js UI     |  Branching exploration graph
|    + Supabase     |  + question cards + persistence
+-------------------+

Wikipedia Path: KeyBERT keyphrase extraction -> Wikipedia API summaries
Gemini Path:    Counter-arguments + multi-perspective context generation

Ancestry (depth > 0):
  depth 1-3: Full ancestry chain included
  depth > 3: Distant nodes summarised by Gemini, recent 2 in full
```

---

## Repository Structure

```
socratic-path/
├── backend/                          # FastAPI REST API
│   ├── main.py                       # App entrypoint, lifespan, CORS, service injection
│   ├── auth.py                       # Supabase JWT validation (JWKS, ES256)
│   ├── routes/
│   │   ├── explore.py                # POST /api/explore (smart routing, auth required)
│   │   └── explorations.py           # CRUD for exploration persistence
│   ├── schemas/
│   │   └── models.py                 # Pydantic request/response models
│   └── services/
│       ├── keyphrase.py              # KeyBERT extraction + POS filtering
│       ├── wikipedia.py              # Wikipedia context retrieval
│       ├── question_gen.py           # T5 model loading & inference
│       ├── gemini_service.py         # Gemini 2.5 Flash (classify, context, summarise)
│       ├── context_router.py         # Smart routing orchestrator
│       └── database.py               # Supabase CRUD for explorations
│
├── frontend/                         # Next.js 16 (App Router)
│   ├── app/
│   │   ├── layout.tsx                # Root layout with AuthProvider
│   │   ├── page.tsx                  # Main page (auth-gated)
│   │   ├── login/page.tsx            # Google OAuth login
│   │   └── globals.css               # Tailwind v4 + custom theme
│   ├── components/
│   │   ├── InputPanel.tsx            # Topic input + type toggles + generate
│   │   ├── QuestionCards.tsx         # Colour-coded cards with reflection areas
│   │   ├── ExplorationGraph.tsx      # React Flow branching tree visualisation
│   │   ├── SourcePanel.tsx           # Wikipedia sources + pipeline indicator
│   │   ├── ProgressIndicator.tsx     # 3-stage pipeline progress display
│   │   ├── ExplorationsList.tsx      # Load/delete saved explorations
│   │   ├── ExportDialog.tsx          # Markdown/PNG export
│   │   ├── ReflectionPanel.tsx       # Reflection text input
│   │   ├── LoadingState.tsx          # Skeleton loading animation
│   │   ├── ConceptMap.tsx            # Legacy keyphrase concept map
│   │   └── nodes/
│   │       ├── InputNode.tsx         # User input node (blue)
│   │       ├── QuestionNode.tsx      # Generated question node (type-coloured)
│   │       ├── ReflectionNode.tsx    # User reflection node (orange)
│   │       └── TopicNode.tsx         # Root topic node
│   └── lib/
│       ├── store.ts                  # Zustand state management (persisted)
│       ├── api.ts                    # Axios client + JWT interceptor
│       ├── types.ts                  # TypeScript interfaces
│       ├── supabase.ts               # Supabase client initialisation
│       ├── auth-context.tsx          # React Context for Google OAuth
│       ├── graph-layout.ts           # Dagre layout for exploration tree
│       ├── export.ts                 # Markdown export logic
│       └── utils.ts                  # cn() class merge utility
│
├── supabase/
│   └── migrations/
│       └── 001_initial_schema.sql    # DB schema (profiles, explorations, nodes, RLS)
│
├── scripts/                          # Training & evaluation scripts
│   ├── train_model.py                # LoRA fine-tuning (FLAN-T5-small/base, T5-base)
│   ├── evaluate_model.py             # ROUGE, BERTScore, BLEU evaluation
│   ├── plot_training_curves.py       # Training curve visualisation
│   ├── test_e2e.py                   # End-to-end API testing (/api/explore)
│   └── run_all_training.sh           # Batch training runner
│
├── notebooks/                        # Jupyter notebooks (research pipeline)
│   ├── 01_data_inspection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_keyphrase_extraction.ipynb
│   └── 06_inference_pipeline.ipynb
│
├── models/                           # Trained model artifacts (git-ignored)
│   ├── flan-t5-small-lora/
│   ├── flan-t5-base-lora/
│   └── t5-base-lora/                 # Best model
│       ├── adapter/                  # LoRA adapter weights
│       ├── checkpoints/              # Training checkpoints
│       ├── logs/                     # TensorBoard logs
│       └── merged/                   # Merged model for inference
│
├── datasets/                         # Dataset files (git-ignored)
├── evaluation_results/               # Evaluation outputs & plots
├── docs/                             # Architecture & training decision docs
├── Dockerfile                        # Production container (CPU PyTorch)
├── pyproject.toml                    # Python deps (uv) — runtime + training extras
└── uv.lock                           # Dependency lock file
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

| Parameter               | Value                  |
| ----------------------- | ---------------------- |
| LoRA rank (r)           | 16                     |
| LoRA alpha              | 32                     |
| LoRA dropout            | 0.1                    |
| Target modules          | q, k, v, o (attention) |
| Learning rate           | 3e-4                   |
| Scheduler               | Cosine with warmup     |
| Warmup ratio            | 0.06                   |
| Epochs                  | 10                     |
| Batch size              | 16                     |
| Early stopping patience | 5                      |
| Precision               | bf16 (CUDA)            |

### Prompt Format

```
Generate a Socratic question for this context: {question_type}: {input_text}

Background information: {context}
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

**Framework:** FastAPI with lifespan-managed service injection

### API Endpoints

| Method | Path                     | Auth     | Description                                |
| ------ | ------------------------ | -------- | ------------------------------------------ |
| POST   | `/api/explore`           | Required | Generate questions (smart context routing) |
| GET    | `/api/explorations`      | Required | List user's saved explorations             |
| GET    | `/api/explorations/{id}` | Required | Fetch exploration with all nodes           |
| POST   | `/api/explorations`      | Required | Create/upsert exploration tree             |
| DELETE | `/api/explorations/{id}` | Required | Delete exploration (cascades to nodes)     |
| GET    | `/api/health`            | No       | Health check (model loaded status)         |

### POST /api/explore

The primary endpoint powering the branching exploration UI.

**Request:**

```json
{
 "text": "I think social media does more harm than good",
 "parent_question_id": null,
 "ancestry": [],
 "depth": 0,
 "question_types": ["clarity", "reasons_evidence", "assumptions"]
}
```

**Response:**

```json
{
 "input_classification": {
  "input_type": "opinion",
  "core_thesis": "Social media is net harmful",
  "confidence": 0.85,
  "reasoning": "Contains value judgement 'more harm than good'"
 },
 "pipeline_path": "gemini",
 "keyphrases": [
  { "text": "social media", "score": 0.72 },
  { "text": "harm", "score": 0.58 }
 ],
 "context_sources": [],
 "questions": [
  {
   "id": "q0_1",
   "type": "clarity",
   "text": "What specific aspects of social media are you referring to?",
   "related_keyphrases": ["social media"]
  }
 ],
 "depth_nudge": null,
 "processing_time_ms": 4230
}
```

### Services

| Service                       | Responsibility                                                                      |
| ----------------------------- | ----------------------------------------------------------------------------------- |
| **KeyphraseService**          | KeyBERT with all-MiniLM-L6-v2, MMR diversity, POS-tag filtering for nominal phrases |
| **WikipediaService**          | Wikipedia API with summary truncation (~400 chars at sentence boundary)             |
| **QuestionGenerationService** | Loads merged T5-base model, beam search (4 beams, max_length=80)                    |
| **GeminiService**             | Input classification, context generation for arguments, ancestry summarisation      |
| **ContextRouter**             | Orchestrates classification -> routing -> context generation                        |
| **DatabaseService**           | Supabase CRUD for explorations and nodes (service-role key)                         |

### Authentication

- **Method:** Supabase JWT validation via JWKS endpoint (ES256)
- **Dependency:** `get_current_user()` extracts user ID from JWT `sub` claim
- **Graceful fallback:** Auth is disabled if `SUPABASE_URL` is not set (local dev without auth)

---

## Frontend

**Framework:** Next.js 16 (App Router) with React 19 and TypeScript

### Authentication

- Google OAuth via Supabase Auth
- `AuthProvider` context wraps the app, providing `useAuth()` hook
- JWT attached to all API requests via Axios interceptor
- Unauthenticated users redirected to `/login`

### Pages

| Route    | Description                        |
| -------- | ---------------------------------- |
| `/`      | Main exploration page (auth-gated) |
| `/login` | Google OAuth sign-in               |

### Components

- **InputPanel** — Topic textarea (up to 5000 chars), colour-coded question type toggles, classification badge showing detected input type and pipeline path, generate/reset buttons
- **ExplorationGraph** — React Flow tree visualisation with Dagre hierarchical layout. Custom nodes for inputs (blue), questions (type-coloured), and reflections (orange). Clickable nodes for selection, auto-fit on new content
- **QuestionCards** — Colour-coded cards grouped by Socratic type. Each card has a reflection text area — submitting a reflection triggers deeper exploration (branching)
- **SourcePanel** — Wikipedia source cards with pipeline path indicator (Wikipedia / AI-Generated / Fallback)
- **ProgressIndicator** — 3-stage display: classifying -> gathering context -> generating questions
- **ExplorationsList** — Modal listing saved explorations (newest first) with load/delete actions
- **ExportDialog** — Export the full exploration tree as Markdown/PNG via html-to-image
- **ReflectionPanel** — Text input for reflecting on a generated question before branching deeper
- **LoadingState** — Shimmer skeleton placeholders matching the results layout

### State Management

Zustand store (persisted to localStorage as `socratic-path-exploration`) manages:

- **Exploration tree:** nodes, rootId, children relationships, collapse/expand state
- **Generation state:** topic, selected question types, generation stage (idle / classifying / gathering-context / generating-questions)
- **Classification:** last input classification result (type, thesis, confidence)
- **Persistence:** current exploration ID, save/load status
- **UI state:** selected node, active reflection, error messages, retry handler

Key actions: `submitInitialInput()`, `submitReflection(questionId, text)`, `saveCurrentExploration()`, `loadExploration()`, `collapseSubtree()`, `expandSubtree()`, `getAncestryPath()`.

---

## Setup & Installation

### Prerequisites

- Python 3.10+ with [uv](https://docs.astral.sh/uv/) package manager
- Node.js 18+ with npm
- A [Supabase](https://supabase.com/) project (for auth and persistence)
- A [Google AI](https://aistudio.google.com/) API key (for Gemini 2.5 Flash)
- Trained model artifacts in `models/t5-base-lora/merged/` (or use the HuggingFace Hub model `DevAnuhas/socraticpath-t5-base`)

### Environment Variables

**Backend** (`backend/.env`):

```env
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
FRONTEND_URL=http://localhost:3000
MODEL_PATH=models/t5-base-lora/merged   # optional, defaults to this path
```

**Frontend** (`frontend/.env`):

```env
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Database Setup

Run the migration in your Supabase SQL editor to create the required tables (profiles, explorations, exploration_nodes) with Row Level Security:

```bash
# Apply from supabase/migrations/001_initial_schema.sql
```

### Backend

```bash
# Install runtime dependencies
uv sync

# (Optional) Install training dependencies too
uv sync --extra training

# Start the backend server
uv run uvicorn backend.main:app --reload --port 8000
```

The backend loads the T5 model, KeyBERT, and initialises Gemini + Supabase on startup (~10-15s). Health check at `http://localhost:8000/api/health`.

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

Frontend runs at `http://localhost:3000`.

---

## Running the Application

1. Set up environment variables (see above)
2. Apply the Supabase migration
3. Start the backend: `uv run uvicorn backend.main:app --reload --port 8000`
4. Start the frontend: `cd frontend && npm run dev`
5. Open `http://localhost:3000` and sign in with Google
6. Enter a topic or argument and click Generate
7. Reflect on generated questions to explore deeper branches

Typical inference time is 3-8 seconds (classification + context routing + model inference).

---

## Docker Deployment

A production Dockerfile is included for CPU-only deployment:

```bash
docker build -t socratic-path .
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e SUPABASE_URL=your_url \
  -e SUPABASE_SERVICE_ROLE_KEY=your_key \
  -e FRONTEND_URL=https://your-frontend.com \
  socratic-path
```

The Docker image:

- Uses `python:3.11-slim` base
- Installs CPU-only PyTorch
- Downloads the merged T5 model from HuggingFace Hub (`DevAnuhas/socraticpath-t5-base`) at build time
- Pre-downloads NLTK data for POS tagging
- Runs on port 8000 via Uvicorn

---

## Tech Stack

| Layer                   | Technology                                          |
| ----------------------- | --------------------------------------------------- |
| Question Generation     | T5-base + LoRA (PyTorch, Transformers)              |
| Input Classification    | Gemini 2.5 Flash (google-genai)                     |
| Context Generation      | Gemini (argumentative) / Wikipedia API (factual)    |
| Keyphrase Extraction    | KeyBERT + sentence-transformers (all-MiniLM-L6-v2)  |
| Backend                 | FastAPI, Pydantic, Uvicorn                          |
| Frontend                | Next.js 16, React 19, TypeScript                    |
| UI                      | Tailwind CSS v4, Lucide React icons                 |
| Exploration Graph       | React Flow (@xyflow/react) + Dagre layout           |
| State Management        | Zustand 5 (persisted to localStorage)               |
| Authentication          | Supabase Auth (Google OAuth), JWT (python-jose)     |
| Database                | Supabase (PostgreSQL with Row Level Security)       |
| Export                  | html-to-image, react-markdown                       |
| Training Infrastructure | AWS EC2 g5.2xlarge (NVIDIA A10G), PEFT/LoRA         |
| Deployment              | Docker (CPU PyTorch), HuggingFace Hub model hosting |
| Package Management      | uv (Python), npm (Node.js)                          |

---

## References

- Ang, L. et al. (2023). _SocratiQ: A dataset for Socratic question generation_. EACL.
- Hu, E.J. et al. (2022). _LoRA: Low-Rank Adaptation of Large Language Models_. ICLR.
- Raffel, C. et al. (2020). _Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer_. JMLR.
- Wei, J. et al. (2022). _Finetuned Language Models Are Zero-Shot Learners_. ICLR.
- Lin, C.Y. (2004). _ROUGE: A Package for Automatic Evaluation of Summaries_. ACL Workshop.
- Zhang, T. et al. (2020). _BERTScore: Evaluating Text Generation with BERT_. ICLR.
