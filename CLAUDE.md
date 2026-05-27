# ChordAI — Project Brief for Claude Code

## What this project is

ChordAI is a multi-model AI pipeline that generates complete songs. Given a
description or structured parameters (genre, decade, tempo, emotional vibe),
it produces a chord progression and matching song lyrics.

A dedicated **AI Planner model** interprets user input, resolves ambiguity,
enriches parameters, and decides which downstream models to call and in what
order. The user interacts through a Nuxt web UI. The backend is a FastAPI
server running on a GPU server. All AI models run locally via Ollama.

---

## High-level architecture

```
┌─────────────────────────────────────────────────────┐
│  Nuxt Frontend  (UI + client-side, TypeScript)      │
│  - Freetext + structured form input                 │
│  - Streams token output in real time via WebSocket  │
└────────────────────┬────────────────────────────────┘
                     │  WebSocket + REST (HTTP)
┌────────────────────▼────────────────────────────────┐
│  FastAPI Backend  (Python)                          │
│  - Manages WebSocket connections                   │
│  - Orchestrates the pipeline                       │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  Planner Model  (AI — Ollama: llama3.2)             │
│  - Interprets freetext and structured input        │
│  - Resolves ambiguity, infers missing parameters   │
│  - Outputs a validated PlannerDecision JSON object │
│  - Decides which models to call and in what order  │
└───────┬─────────────────────────┬───────────────────┘
        │                         │
┌───────▼──────────┐   ┌──────────▼───────────────────┐
│  Chord Model     │   │  Lyrics Model                │
│  sample_chords.py│   │  gemma4_lyrics.py            │
│  (existing)      │   │  (existing)                  │
└──────────────────┘   └──────────────────────────────┘
```

---

## Existing models (already written — do not rewrite these)

### 1. Chord Model — `sample_chords.py`

Generates chord progressions. Run from command line:

```bash
# Generate from scratch
python sample_chords.py --out_dir=out-chords --mode=generate --genre=pop --decade=2010

# Extend a seed progression
python sample_chords.py --out_dir=out-chords --mode=extend --genre=rock --decade=1990 \
    --seed_chords="Emin C G D"

# Generate next section
python sample_chords.py --out_dir=out-chords --mode=section --genre=pop --decade=2010 \
    --seed_chords="<verse> C G Amin F" --next_section=chorus
```

**Output** — a token string on stdout:
```
<bos> <genre:pop> <decade:2010> <intro> C <verse> F C E7 Amin <chorus> F C G C ... <eos>
```

### 2. Lyrics Model — `gemma4_lyrics.py`

Consumes the chord model token string and generates section-aware lyrics using
Gemma 4 via Ollama.

```bash
python gemma4_lyrics.py \
    --out_dir=out-chords \
    --mode=generate \
    --genre=pop \
    --decade=2010 \
    --tempo=110 \
    --vibe="hopeful"

# Or bypass chord model with a token string directly:
python gemma4_lyrics.py \
    --token_string="<bos> <genre:rock> <decade:1990> <verse> Emin C G D <chorus> C G D Emin <eos>" \
    --tempo=130 \
    --vibe="rebellious"
```

**Output** — saves `.txt` and `.json` to `--out_dir`. JSON structure:
```json
{
  "genre": "pop",
  "decade": 2010,
  "tempo_bpm": 110,
  "vibe": "hopeful",
  "sections": {
    "verse": ["F", "C", "G", "Am"],
    "chorus": ["F", "C", "G", "C"]
  },
  "raw_tokens": "<bos> <genre:pop> ...",
  "lyrics": "[Verse 1]\n(Chords: F C G Am)\n..."
}
```

### 3. Planner Model — `backend/planner.py`

**This is an AI model, not a rule-based class.**

Uses `llama3.2` via Ollama (fast, small, low latency — ideal for routing).
The Planner sits between the user and the downstream models. It receives raw
user input (freetext description and/or partial structured parameters), reasons
about it, fills in any missing values, and returns a validated `PlannerDecision`
object that the backend uses to drive the pipeline.

**Responsibilities:**
- Interpret ambiguous or freetext input (e.g. "something that sounds like a
  rainy Sunday morning" → `genre=folk, tempo=72, vibe=melancholic, decade=1970`)
- Infer missing parameters from context
- Select which mode to use (`generate`, `extend`, or `section`)
- Decide the execution order of downstream models
- Validate and normalise all values before they reach the chord/lyrics models

**Input to the Planner** — `PlannerInput`:
```json
{
  "freetext": "make me something melancholic, like early Radiohead",
  "genre": null,
  "decade": null,
  "tempo_bpm": null,
  "vibe": null,
  "mode": null,
  "seed_chords": ""
}
```
Any field can be null/empty — the Planner fills gaps using the freetext.
If both freetext and a structured field are provided, the structured field
takes priority (the user was explicit).

**Output from the Planner** — `PlannerDecision`:
```json
{
  "genre": "rock",
  "decade": 1990,
  "tempo_bpm": 88,
  "vibe": "melancholic and detached",
  "mode": "generate",
  "seed_chords": "",
  "next_section": "",
  "pipeline": ["chord_model", "lyrics_model"],
  "reasoning": "Early Radiohead is mid-90s alternative rock, slow tempo, melancholic..."
}
```

The `pipeline` array defines which models to call in order. This is how new
models will be added in the future — the Planner simply includes them in the
array and the backend executor handles the rest.

**Implementation details:**
- Uses `ollama.chat()` with `model="llama3.2"` (not Gemma 4 — keep the models separate)
- The system prompt instructs the model to respond ONLY with a valid JSON object
  matching the `PlannerDecision` schema — no preamble, no explanation
- Temperature: `0.3` (low — we want consistent, deterministic routing)
- If JSON parsing fails, retry once with an explicit correction prompt before
  raising an error
- The `reasoning` field is logged server-side for debugging but never sent to
  the frontend

---

## Shared data contracts

### SongSpec
Passed between chord model and lyrics model:
```python
@dataclass
class SongSpec:
    genre:      str                    # e.g. "pop"
    decade:     int                    # e.g. 2010
    tempo_bpm:  int                    # e.g. 110
    vibe:       str                    # e.g. "hopeful"
    sections:   dict[str, list[str]]  # e.g. {"verse": ["F","C","G"], ...}
    raw_tokens: str                    # original token string
```

### PlannerInput
What the frontend sends to the backend:
```python
class PlannerInput(BaseModel):
    freetext:   str        # optional natural language description
    genre:      str | None
    decade:     int | None
    tempo_bpm:  int | None
    vibe:       str | None
    mode:       str | None  # "generate" | "extend" | "section"
    seed_chords: str = ""
    next_section: str = ""
```

### PlannerDecision
What the Planner model returns (used internally by the backend):
```python
class PlannerDecision(BaseModel):
    genre:        str
    decade:       int
    tempo_bpm:    int
    vibe:         str
    mode:         str
    seed_chords:  str
    next_section: str
    pipeline:     list[str]   # e.g. ["chord_model", "lyrics_model"]
    reasoning:    str         # logged only, never sent to client
```

---

## What needs to be built

### A — FastAPI Backend (`/backend`)

**`main.py`** — FastAPI app entry point
- CORS configured for Nuxt dev server (localhost:3000)
- Mounts routers

**`routers/generate.py`** — REST + WebSocket endpoints
- `POST /api/generate` — accepts PlannerInput, returns full song (non-streaming)
- `WS  /ws/generate`  — same but streams status events + tokens to client

**`planner.py`** — The AI Planner model (see full spec above)

**`executor.py`** — Pipeline executor
- Reads the `pipeline` array from PlannerDecision
- Calls each model runner in order, passing outputs forward
- Emits WebSocket status events at each step:
  `{"stage": "planner", "status": "done"}`
  `{"stage": "chord_model", "status": "running"}`
  `{"stage": "lyrics_model", "status": "streaming", "token": "..."}`

**`models/chord_runner.py`** — wrapper around `sample_chords.py`
- Runs as subprocess, captures stdout
- Parses token string into SongSpec

**`models/lyrics_runner.py`** — wrapper around `gemma4_lyrics.py`
- Accepts SongSpec + PlannerDecision, runs lyrics model
- Streams tokens back via a callback or async generator

**`schemas.py`** — All Pydantic models (PlannerInput, PlannerDecision, SongSpec, etc.)

### B — Nuxt Frontend (`/frontend`)

Tech stack: Nuxt 3, TypeScript, Tailwind CSS

**Pages:**
- `/` — main song generator page

**Components (build one at a time):**
1. `SongForm.vue` — freetext description field + optional structured overrides
   (genre, decade, tempo slider, vibe). Structured fields are collapsed by
   default — the freetext field is the primary input.
2. `SongOutput.vue` — displays generated lyrics with section labels and chord annotations
3. `StreamingText.vue` — real-time token streaming display (WebSocket consumer)
4. `ChordDisplay.vue` — visual chord progression display per section
5. `StatusBar.vue` — shows pipeline progress using stage events from the backend:
   "Planner thinking... → Generating chords... → Writing lyrics..."

**Composables:**
- `useGenerate.ts` — wraps the WebSocket connection, handles stage events and
  token streaming, exposes reactive state to components

---

## Technology choices (decided — do not change)

| Layer | Technology | Reason |
|-------|-----------|--------|
| Frontend | Nuxt 3 + TypeScript | SSR/client hybrid, good DX |
| Styling | Tailwind CSS | utility-first, fast iteration |
| Backend | FastAPI (Python) | same language as models, WebSocket support |
| AI runtime | Ollama (local) | all models run on GPU server |
| Planner model | llama3.2 via Ollama | fast, small, low latency for routing |
| Lyrics model | Gemma 4 via Ollama | see gemma4_lyrics.py |
| Chord model | sample_chords.py | existing model, called via subprocess |
| Pipeline execution | executor.py | reads Planner's pipeline array, not hardcoded |

---

## Project folder structure

```
ChordAI/
├── CLAUDE.md                  ← this file
├── backend/
│   ├── main.py
│   ├── planner.py             ← AI model (llama3.2)
│   ├── executor.py            ← pipeline runner
│   ├── schemas.py
│   ├── routers/
│   │   └── generate.py
│   ├── models/
│   │   ├── chord_runner.py
│   │   └── lyrics_runner.py
│   └── requirements.txt
├── frontend/
│   ├── nuxt.config.ts
│   ├── tailwind.config.ts
│   ├── pages/
│   │   └── index.vue
│   ├── components/
│   │   ├── SongForm.vue
│   │   ├── SongOutput.vue
│   │   ├── StreamingText.vue
│   │   ├── ChordDisplay.vue
│   │   └── StatusBar.vue
│   └── composables/
│       └── useGenerate.ts
├── sample_chords.py           ← existing chord model (do not modify)
├── gemma4_lyrics.py           ← existing lyrics model (do not modify)
└── out-chords/                ← shared output directory for both models
```

---

## Dev setup

```bash
# Backend
cd backend
python3 -m venv venv && source venv/bin/activate
pip install fastapi uvicorn websockets pydantic ollama rich
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev          # runs on localhost:3000

# Ollama — pull both models used
ollama serve
ollama pull gemma4    # lyrics model
ollama pull llama3.2  # planner model
```

---

## How to work on this project

**Build one component at a time.** When asked to build a component, only create
that component and its direct dependencies. Do not scaffold the entire project
at once.

The typical build order will be:
1. `schemas.py` — all shared Pydantic models
2. `planner.py` — AI Planner model
3. `models/chord_runner.py` — chord model wrapper
4. `models/lyrics_runner.py` — lyrics model wrapper
5. `executor.py` — pipeline executor
6. `main.py` + `routers/generate.py` — FastAPI app + endpoints
7. Nuxt project setup + `nuxt.config.ts`
8. `SongForm.vue` component
9. `SongOutput.vue` component
10. `StreamingText.vue` component
11. `ChordDisplay.vue` component
12. `StatusBar.vue` + `useGenerate.ts`

---
∏
## Key constraints

- `sample_chords.py` and `gemma4_lyrics.py` must not be modified. Wrappers
  call them via subprocess.
- The Planner is an AI model (llama3.2 via Ollama) — not a rule-based class.
- The Planner must output strict JSON matching PlannerDecision — enforce this
  in the system prompt and validate with Pydantic on every response.
- The `pipeline` array in PlannerDecision drives execution — never hardcode
  the model call order in the executor.
- All AI inference runs locally via Ollama — no external API calls for models.
- Lyrics stream token by token to the browser via WebSocket.
- The frontend must degrade gracefully if the WebSocket drops — fall back to
  polling the REST endpoint.