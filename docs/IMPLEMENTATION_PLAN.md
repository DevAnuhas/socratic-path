# SocraticPath Application Artifact — Implementation Plan

> **Optimized for:** 4-day build (March 4-7, 2026), live local demo, CS/SE examiner audience
> **Thesis writing:** March 8-9 (2 days reserved)
> **Submission deadline:** March 10, 2026

---

## Executive Summary

SocraticPath is a web application that generates Socratic questions from user-provided topics. The pipeline is: **User enters topic → KeyBERT extracts keyphrases → Wikipedia retrieves context → Fine-tuned T5-base generates Socratic questions → React Flow visualizes concept relationships.**

This plan strips the original ROADMAP down to what is feasible, demonstrable, and academically defensible in 4 days. Everything cut was either unnecessary (ChromaDB, KP20k), scope creep (Gemini), or overengineered for a dissertation demo.

---

## What Was Cut and Why

| Cut Item                                     | Original Purpose                                 | Why Cut                                                                                                                                                       |
| -------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ChromaDB / Vector Store**                  | RAG-style retrieval from training contexts       | Architecturally unnecessary. The model generates questions from user context, not from stored training data. Wikipedia API already provides external context. |
| **Gemini API integration**                   | Alternative context retrieval                    | Muddies the research contribution. The dissertation is about fine-tuned T5 models, not Gemini. Adds API key dependency and failure modes to a live demo.      |
| **KP20k dataset / keyphrase model training** | Custom keyphrase extraction                      | KeyBERT (pre-trained, already working in NB06) does this job perfectly. Training a keyphrase model adds zero research value.                                  |
| **Sentence-transformers / FAISS**            | Embedding and similarity search                  | Only needed for ChromaDB, which is cut.                                                                                                                       |
| **Multi-model selection in UI**              | Let user pick which model to use                 | Adds memory and complexity. Serve T5-base only (best performer). Model comparison belongs in the thesis, not the UI.                                          |
| **Full concept map spec**                    | Context menus, manual edges, PNG export, minimap | Overengineered for a demo. Pragmatic middle ground: interactive zoom/pan, click-to-expand, color-coded nodes, animated edges. Still visually impressive.      |
| **Cloud deployment**                         | Railway/Vercel hosting                           | Live local demo confirmed. No deployment needed.                                                                                                              |

---

## Architecture (Simplified)

```
┌─────────────────────────────────────────────────────┐
│                  Next.js Frontend                   │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐   │
│  │ Input Pane  │  │ Question     │  │ Concept   │   │
│  │ (topic +    │  │ Cards        │  │ Map       │   │
│  │  controls)  │  │ (color-coded)│  │ (React    │   │
│  │             │  │              │  │  Flow)    │   │
│  └─────────────┘  └──────────────┘  └───────────┘   │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (localhost)
┌──────────────────────▼──────────────────────────────┐
│                  FastAPI Backend                    │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │ Keyphrase    │  │ Wikipedia   │  │ Question   │  │
│  │ Service      │  │ Service     │  │ Generation │  │
│  │ (KeyBERT)   │  │ (REST API)  │  │ Service    │  │
│  │              │  │              │  │ (T5-base   │  │
│  │              │  │              │  │  + LoRA)   │  │
│  └──────────────┘  └─────────────┘  └────────────┘  │
└──────────────────────────────────────────────────────┘
```

### Data Flow (matches proposal diagram)

```
1. User enters topic (e.g., "Climate change effects on agriculture")
        │
2. KeyBERT extracts keyphrases → ["climate change", "agriculture", "crop yields"]
        │
3. Wikipedia API retrieves context paragraphs for each keyphrase
        │
4. Context + question type prefix → T5-base inference
   "Generate a Socratic question: reasons_evidence: {wikipedia_context}"
        │
5. Model generates questions for each of the 5 Socratic types
        │
6. Frontend displays:
   - Questions as color-coded cards
   - Keyphrases as concept map nodes
   - Wikipedia sources as attribution links
```

---

## Technology Stack (Final)

| Layer                | Technology                             | Justification                                      |
| -------------------- | -------------------------------------- | -------------------------------------------------- |
| Frontend             | Next.js 16 + TypeScript + Tailwind CSS | Already scaffolded, App Router, modern React       |
| Visualization        | @xyflow/react (React Flow)             | Already installed, purpose-built for node graphs   |
| State management     | Zustand                                | Already installed, minimal boilerplate             |
| Backend              | FastAPI + Uvicorn                      | Already scaffolded, async-native, auto-docs        |
| Keyphrase extraction | KeyBERT                                | Pre-trained, proven in NB06, zero training needed  |
| Context retrieval    | Wikipedia REST API                     | Free, no API key, reliable, already proven in NB06 |
| Question generation  | T5-base + LoRA (merged model)          | Best performer (ROUGE-L 0.217, BERTScore 0.895)    |
| Inference            | PyTorch (CPU)                          | Adequate for live demo (~2-5s latency)             |

---

## Model Serving Strategy

**Serve the merged T5-base + LoRA model only.**

- **Why T5-base?** Highest ROUGE-L (0.217), highest BERTScore (0.895), and your eval shows instruction tuning provides no advantage — this is a thesis-worthy finding.
- **Why merged?** Avoids the fragile adapter loading sequence (tokenizer → base → resize → adapter). Merged model is a single `AutoModelForSeq2SeqLM.from_pretrained()` call.
- **Why CPU?** Live demo on laptop. T5-base (226M params) runs fine on CPU with ~2-5s inference. No CUDA dependency to break during demo.

### Model Loading (startup)

The merged model already exists at `models/t5-base-lora/merged/` (~891MB safetensors + tokenizer). No merging step required.

```python
# One-time at server startup — no adapter dance needed
model = AutoModelForSeq2SeqLM.from_pretrained("../models/t5-base-lora/merged")
tokenizer = AutoTokenizer.from_pretrained("../models/t5-base-lora/merged")
model.eval()
```

---

## Backend Design

### Directory Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, CORS, startup event
│   ├── routes/
│   │   ├── __init__.py
│   │   └── generate.py      # POST /api/generate
│   ├── services/
│   │   ├── __init__.py
│   │   ├── keyphrase.py     # KeyBERT wrapper
│   │   ├── wikipedia.py     # Wikipedia REST API client
│   │   └── question_gen.py  # T5-base inference
│   └── schemas/
│       ├── __init__.py
│       └── models.py        # Pydantic request/response models
├── model_artifacts/          # Merged T5-base model
├── pyproject.toml
└── requirements.txt
```

### API Endpoints

**`POST /api/generate`** — The single core endpoint.

Request:

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

Response:

```json
{
	"topic": "Climate change effects on agriculture",
	"keyphrases": [
		{ "text": "climate change", "score": 0.85 },
		{ "text": "agriculture", "score": 0.72 },
		{ "text": "crop yields", "score": 0.68 }
	],
	"context_sources": [
		{
			"keyphrase": "climate change",
			"summary": "Climate change includes both...",
			"url": "https://en.wikipedia.org/wiki/Climate_change"
		}
	],
	"questions": [
		{
			"id": "q1",
			"type": "clarity",
			"text": "What specific aspects of agriculture are you referring to?",
			"related_keyphrases": ["agriculture"]
		},
		{
			"id": "q2",
			"type": "reasons_evidence",
			"text": "What evidence links rising temperatures to declining crop yields?",
			"related_keyphrases": ["climate change", "crop yields"]
		}
	]
}
```

**`GET /api/health`** — Health check (confirms model is loaded).

**`GET /docs`** — Auto-generated Swagger/OpenAPI docs (free from FastAPI).

### Service Layer

**KeyphraseService** — wraps KeyBERT:

- Extract top-N keyphrases from user topic
- Return keyphrases with relevance scores
- Fallback: if topic is too short for KeyBERT, use the topic itself as a keyphrase

**WikipediaService** — wraps Wikipedia REST API:

- For each keyphrase, fetch Wikipedia summary (first 2-3 sentences)
- Return summary text + article URL
- Timeout: 3 seconds per request, fail gracefully (skip keyphrase if Wikipedia is slow)
- No API key required

**QuestionGenerationService** — wraps T5-base inference:

- For each (question_type, context) pair, construct the prompt:
  `"Generate a Socratic question: {question_type}: {context}"`
- Run beam search (num_beams=4, max_length=128)
- Return generated question text
- Batch all 5 question types in one forward pass where possible

### Error Handling

- Wikipedia timeout → skip that keyphrase's context, generate question from topic alone
- Empty keyphrase extraction → use full topic as context
- Model generates empty/garbage → return the model's most common fallback ("What do you mean by that?") with a flag
- Input too long → truncate to 512 tokens (model's max input length)

---

## Frontend Design

### Directory Structure

```
frontend/src/
├── app/
│   ├── page.tsx              # Main application page
│   ├── layout.tsx            # Root layout
│   └── globals.css           # Tailwind + custom styles
├── components/
│   ├── InputPanel.tsx        # Topic input + question type selector + generate button
│   ├── QuestionCards.tsx     # List of generated questions, color-coded by type
│   ├── ConceptMap.tsx        # React Flow visualization
│   ├── SourcePanel.tsx       # Wikipedia source attribution
│   ├── LoadingState.tsx      # Skeleton/spinner during generation
│   └── nodes/
│       ├── TopicNode.tsx     # Central topic node (large, styled)
│       └── KeyphraseNode.tsx # Keyphrase concept nodes (medium, color-coded)
├── lib/
│   ├── api.ts               # Axios client for backend
│   ├── types.ts             # TypeScript interfaces matching backend schemas
│   └── store.ts             # Zustand store for app state
└── hooks/
    └── useGenerate.ts       # Hook wrapping the generate API call
```

### Page Layout

```
┌──────────────────────────────────────────────────────┐
│  SocraticPath                                 [logo] │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─ Input Panel ──────────────────────────────────┐  │
│  │ Enter a topic: [________________________]      │  │
│  │ Question types: [x] Clarity [x] Evidence ...   │  │
│  │                              [Generate]        │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌─ Concept Map (React Flow) ─────────────────────┐  │
│  │                                                │  │
│  │        [keyphrase1] ── [TOPIC] ── [keyphrase2] │  │
│  │                          │                     │  │
│  │                    [keyphrase3]                 │  │
│  │                                                │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌─ Questions ────────────────────────────────────┐  │
│  │ ┌─────────────────┐ ┌─────────────────┐       │  │
│  │ │ 🔵 Clarity      │ │ 🟢 Evidence     │       │  │
│  │ │ "What do you    │ │ "What data      │       │  │
│  │ │  mean by...?"   │ │  supports...?"  │       │  │
│  │ └─────────────────┘ └─────────────────┘       │  │
│  │ ┌─────────────────┐ ┌─────────────────┐       │  │
│  │ │ 🟠 Implications │ │ 🟣 Assumptions  │       │  │
│  │ │ "If this is     │ │ "Why do you     │       │  │
│  │ │  true, then?"   │ │  assume...?"    │       │  │
│  │ └─────────────────┘ └─────────────────┘       │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌─ Sources ──────────────────────────────────────┐  │
│  │ Wikipedia: Climate change, Agriculture, ...    │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### Concept Map Features (Pragmatic Middle Ground)

**Included:**

- Central topic node (large, prominent)
- Keyphrase nodes in a radial layout around the topic
- Color-coded edges by question type
- Animated edges when hovering
- Zoom and pan (built-in with React Flow)
- Click a keyphrase node → highlight its related questions below
- Click a keyphrase node → option to "Explore this concept" (re-runs the pipeline with that keyphrase as the new topic)

**Not included (cut for time):**

- Right-click context menus
- Manual edge creation by dragging
- PNG export
- Minimap
- Tooltips with definitions

### Color Coding (Consistent Across Map + Cards)

| Question Type        | Color  | Hex     | Tailwind Class |
| -------------------- | ------ | ------- | -------------- |
| Clarity              | Blue   | #3B82F6 | `blue-500`     |
| Assumptions          | Purple | #8B5CF6 | `violet-500`   |
| Reasons/Evidence     | Green  | #10B981 | `emerald-500`  |
| Implications         | Orange | #F59E0B | `amber-500`    |
| Alternate Viewpoints | Pink   | #EC4899 | `pink-500`     |

### State Management (Zustand)

```typescript
interface AppState {
	// Input
	topic: string;
	selectedTypes: string[];

	// Results
	keyphrases: Keyphrase[];
	questions: Question[];
	sources: Source[];

	// UI
	isLoading: boolean;
	error: string | null;
	selectedKeyphrase: string | null; // for concept map click-to-filter

	// Actions
	setTopic: (topic: string) => void;
	toggleType: (type: string) => void;
	generate: () => Promise<void>;
	selectKeyphrase: (keyphrase: string | null) => void;
}
```

---

## Build Order (4-Day Schedule)

### Day 1 (March 4): Backend — Core Pipeline

**Morning:**

1. Create backend directory structure
2. Implement `QuestionGenerationService` — model loading from `models/t5-base-lora/merged/` + inference
3. Test: model generates questions from hardcoded context
4. Implement `KeyphraseService` — KeyBERT wrapper

**Afternoon:** 5. Implement `WikipediaService` — REST API client with timeouts 6. Wire up `POST /api/generate` endpoint with all 3 services 7. Test: full pipeline works via `curl` or Swagger UI 8. Add `GET /api/health` endpoint

**Exit criteria:** `curl -X POST localhost:8000/api/generate -d '{"topic": "climate change"}' ` returns keyphrases + Wikipedia context + 5 generated questions.

### Day 2 (March 5): Frontend — Core UI

**Morning:**

1. Set up API client (`lib/api.ts`) and TypeScript types (`lib/types.ts`)
2. Create Zustand store (`lib/store.ts`)
3. Build `InputPanel` component (topic input, question type checkboxes, generate button)
4. Build `QuestionCards` component (color-coded cards displaying generated questions)

**Afternoon:** 5. Build `LoadingState` component (skeleton placeholders during generation) 6. Build `SourcePanel` component (Wikipedia attribution links) 7. Wire everything together on `page.tsx` 8. Test: full flow from topic input to question display works

**Exit criteria:** User can type a topic, click Generate, see loading state, then see color-coded question cards and source links.

### Day 3 (March 6): Frontend — Concept Map + Polish

**Morning:**

1. Build `ConceptMap` component with React Flow
2. Create `TopicNode` and `KeyphraseNode` custom node components
3. Implement radial layout algorithm for node positioning
4. Wire concept map to Zustand store (populated from API response)

**Afternoon:** 5. Add click-to-filter: clicking a keyphrase highlights related questions 6. Add "Explore this concept" interaction (re-runs pipeline with clicked keyphrase) 7. Add animated edges and color coding 8. Responsive layout and Tailwind polish

**Exit criteria:** Full working application with concept map, question cards, and source attribution. Interactive concept exploration works.

### Day 4 (March 7): Integration Testing + Hardening

**Morning:**

1. End-to-end testing: try 10+ diverse topics, verify pipeline handles each
2. Edge case handling:
   - Very short topics (1-2 words)
   - Very long topics (paragraph-length)
   - Topics with no Wikipedia results
   - Empty input validation
3. Error states in UI (display user-friendly messages)

**Afternoon:** 4. Performance check: ensure <5s end-to-end response time 5. Final UI polish (spacing, typography, loading animations) 6. Write brief README with setup instructions (for examiner) 7. Do a full dry-run demo rehearsal

**Exit criteria:** Application is stable, handles edge cases gracefully, looks polished, runs reliably from cold start.

---

## Risk Register

| Risk                                   | Likelihood | Impact | Mitigation                                                                                                                                                                                                    |
| -------------------------------------- | ---------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **T5-base inference too slow on CPU**  | Medium     | High   | Pre-load model at startup. If >10s, switch to FLAN-T5-small (3x faster, acceptable quality).                                                                                                                  |
| **Wikipedia API rate-limited or down** | Low        | Medium | Cache Wikipedia responses in-memory. If down during demo, have 2-3 pre-fetched topic responses as fallback.                                                                                                   |
| **KeyBERT extraction quality varies**  | Low        | Low    | If keyphrases are poor for a topic, the questions still generate (just less targeted). Not a demo-breaker.                                                                                                    |
| **Model generates generic fallbacks**  | Medium     | Medium | Expected for ~10% of inputs (eval shows 12% zero-ROUGE). This is actually a thesis talking point, not a bug. Frame as "the model appropriately falls back to clarification probes when context is ambiguous." |
| **React Flow layout looks cluttered**  | Low        | Medium | Limit to top-5 keyphrases. Radial layout with fixed spacing prevents overlap.                                                                                                                                 |
| **Laptop runs out of memory**          | Low        | High   | T5-base needs ~1GB RAM for model + ~500MB for PyTorch overhead. Close other apps before demo.                                                                                                                 |
| **Demo topic produces bad results**    | Medium     | High   | **Pre-test 5 reliable demo topics** the day before. Have a scripted demo path.                                                                                                                                |

### Pre-Tested Demo Topics (prepare these on Day 4)

1. "Climate change effects on agriculture"
2. "Artificial intelligence in healthcare"
3. "Social media impact on mental health"
4. "Renewable energy vs fossil fuels"
5. "Privacy concerns with facial recognition"

---

## What This Artifact Demonstrates (for Thesis)

| Marking Criterion                   | What the Artifact Shows                                                                                                                               |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Achievement of Objectives (25%)** | Working end-to-end pipeline: topic → keyphrases → context → Socratic questions. All 5 question types demonstrated. Interactive concept visualization. |
| **Use of Literature (15%)**         | Paul & Elder taxonomy (5 question types as UI categories), Ang et al. SocratiQ dataset (model training), KeyBERT for keyphrase extraction.            |
| **Methodology (20%)**               | Clear ML pipeline (data → training → evaluation → serving). LoRA fine-tuning. Proper train/val/test split. Automated + manual evaluation.             |
| **Analysis & Implementation (30%)** | Model comparison (3 models, instruction-tuning finding). BERTScore vs ROUGE analysis. Architecture decisions documented. Error handling.              |

---

## Dependencies to Install

### Backend (add to requirements.txt)

```
fastapi>=0.110.0
uvicorn>=0.29.0
pydantic>=2.6.0
transformers>=4.40.0
torch>=2.0.0
sentencepiece>=0.2.0
keybert>=0.8.0
httpx>=0.27.0
python-dotenv>=1.0.0
```

### Frontend (already installed per package.json)

- `@xyflow/react` — concept map
- `zustand` — state management
- `axios` — HTTP client
- `react-markdown` — rendering
- `lucide-react` — icons

---

## Post-Artifact: Thesis Writing (March 8-9)

The artifact directly feeds these thesis sections:

| Section                                | Content Source                                                                                                   |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| System Architecture                    | Architecture diagram from this plan                                                                              |
| Implementation                         | Code walkthrough of pipeline services                                                                            |
| Results: Model Evaluation              | `docs/evaluation_results.md` (already complete)                                                                  |
| Results: Application Demo              | Screenshots of working app                                                                                       |
| Discussion: Limitations                | Model generates shorter questions (~8 vs 11 words), generic fallbacks for ambiguous input, CPU inference latency |
| Discussion: Instruction Tuning Finding | T5-base outperforms FLAN-T5-base — instruction tuning unnecessary for task-specific LoRA                         |
| Future Work                            | Gemini integration, multi-turn Socratic dialogues, student response analysis                                     |

---

## Success Criteria

- [ ] Backend starts and loads T5-base model in <30 seconds
- [ ] `POST /api/generate` returns valid response for any English topic
- [ ] Frontend displays questions color-coded by all 5 Socratic types
- [ ] Concept map renders with clickable keyphrase nodes
- [ ] Wikipedia sources displayed with attribution links
- [ ] End-to-end response time <5 seconds on laptop
- [ ] Handles edge cases without crashing (empty input, long input, no Wikipedia results)
- [ ] 5 pre-tested demo topics produce good results
- [ ] Application can cold-start and demo in under 60 seconds
