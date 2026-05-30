# ChordAI

AI song generator. Describe a song in plain language; the system generates chord progressions, lyrics, and an album cover.

## Components

| Component | Stack | Purpose |
|-----------|-------|---------|
| [`chord-gen/`](chord-gen/) | Python, PyTorch | Custom chord-progression model — train and sample |
| [`server/`](server/) | Go, Gin | REST + WebSocket API; orchestrates the AI pipeline |
| [`client/`](client/) | Nuxt 3, Vue 3 | Chat-style web UI |
| [`models/`](models/) | Python, Ollama, Diffusers | Lyrics (Gemma 4) and album art generation |

## How it works

```
User prompt
  └─► Go server
        ├─► llama3.2  (Ollama)   — interprets freetext → genre, decade, BPM, vibe
        ├─► sample.py (PyTorch)  — generates chord progressions
        ├─► Gemma 4   (Ollama)   — writes section-aware lyrics
        └─► Dreamshaper XL       — generates 1024×1024 album cover image
```

---

## Setup

### Prerequisites

- Go 1.21+
- Node 18+ with npm
- Python 3.10+
- [Ollama](https://ollama.com) installed and on `$PATH`
- A CUDA GPU (recommended — required for image generation; strongly recommended for chord model training)

---

### 1. Chord model — train on first pull

The chord model checkpoint is not committed to the repo and must be trained locally.
See **[chord-gen/README.md](chord-gen/README.md)** for the full data + training pipeline.

```bash
cd chord-gen
make setup
make data     # ~5 min — streams dataset from HuggingFace
make train    # ~30 min on a single GPU
cd ..
```

If no checkpoint is present the server falls back to a hardcoded stub (genre-appropriate progressions without neural generation), so the rest of the stack can be tested before training completes.

---

### 2. Ollama models

```bash
ollama pull llama3.2   # planner — routes user input to structured parameters
ollama pull gemma4     # lyrics generator
```

Start Ollama before running the server:

```bash
ollama serve           # keep this running in a dedicated terminal
```

---

### 3. Python model dependencies

The image generator and lyrics generator live in `models/` and need their own dependencies:

```bash
pip install torch diffusers transformers accelerate ollama rich
```

For GPU image generation install the CUDA torch wheel — see <https://pytorch.org/get-started/locally/>.

---

### 4. Go server

```bash
cd server
go run main.go         # listens on :5555
```

---

### 5. Frontend

```bash
cd client
npm install
npm run dev            # listens on :3000
```

Open <http://localhost:3000> in your browser.

---

## Cluster setup (UiT IFI)

The project runs split across the entry node (c0-0) and a GPU node (c6-4).
See [HowToRunFrontendOnCluster.md](HowToRunFrontendOnCluster.md) for the full SSH-tunnel and startup instructions.
