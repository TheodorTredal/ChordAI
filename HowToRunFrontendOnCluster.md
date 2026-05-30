# Running ChordAI on the UiT IFI cluster

The cluster has no internet-facing GPU nodes, so the stack is split:

| Where | What runs |
|-------|-----------|
| **c6-4** (GPU node) | Go server, Ollama, Python models (chord/lyrics/image) |
| **c0-0** (entry node) | Nuxt frontend |
| **Your laptop** | Browser, optionally SVCO voice pipeline |

Traffic flows: `browser → SSH tunnel → c0-0:3000 (Nuxt) → c6-4:5555 (Go)`

---

## One-time setup

### On c6-4 — install Ollama

<<<<<<< Updated upstream
### Install ollama localy:
    1. ssh c6-4
    2. mkdir -p local_ollama && cd local_ollama
    3. Download and unpack the official Linux-binary:
        curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | tar --zstd -xvf -
    4. Add the binary to the PATH (run this, or add it to ~/.bashrc for permanent tilgang):
        export PATH=$HOME/local_ollama/bin:$PATH
    5. Downlaod the models:
        ollama pull llama3.2
        ollama pull gemma4

### install python dependencies:
#### Alternativ 1: Download globally on your user
    1. pip install requirements.txt
    2. pip install --user ollama rich


#### Alternativ 2: If you have a venv
    1. source env/bin/activate
    2. pip install requirements.txt
    3. pip install ollama rich



### Setup of the internal cluster routing

Inside of the nuxt.config.ts file make sure that:

1. The server url is this:
    serverUrl: 'http://localhost:8000'

2. and that nitro /api and /ws has the correct target and that changeOrigin and prependPath is set to true:

nitro:
    devProxy:
      '/api': {
        target: 'http://c6-4:5555/api',
        changeOrigin: true,
        prependPath: true
      },
      '/ws':{
        target: 'http://c6-4:5555/ws',
        ws: true,
        changeOrigin: true,
        prependPath: true
      }


---------------------------------------------------------------------------------------------------------

## How to start up

You will need three terminals for this


### Terminal 1: SSH-tunnel & Frontend

#### 1. Log into the cluster with port-forwarding from your Mac
ssh -L 8000:localhost:3000 brukernavn@ificluster.ifi.uit.no

#### 2. Navigate to the frontend folder:
cd ChordAI/client

#### 3. Start Nuxt bound til alle ports:
npm run dev -- --host 0.0.0.0

---------------------------------------------------------------------------------------------------------

#### Terminal 2: Ollama

Ollama needs to run in the background on the GPU node to enable the LLM

# 1. Log into the GPU node:
ssh brukernavn@ificluster.its.uit.no
=======
```bash
ssh <username>@ificluster.its.uit.no
>>>>>>> Stashed changes
ssh c6-4
cd ~/INF-3600/ChordAI
mkdir -p local_ollama && cd local_ollama
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | tar --zstd -xvf -
export PATH=$HOME/local_ollama/bin:$PATH   # add to ~/.bashrc to make permanent
cd ..
ollama pull llama3.2
ollama pull gemma4
```

### On c6-4 — Python dependencies

```bash
pip install -r models/requirements.txt
```

### On c0-0 — Node dependencies

```bash
ssh <username>@ificluster.its.uit.no
cd ~/INF-3600/ChordAI/client
npm install
```

### Chord model checkpoint

Train once on c6-4 (requires GPU, ~30 min):

```bash
cd ~/INF-3600/ChordAI/chord-gen
make setup && make data && make train
```

Without a checkpoint the server uses a rule-based fallback and the rest of the stack still works.

---

## Starting up (3 terminals)

### Terminal 1 — SSH tunnel + Nuxt frontend (on c0-0)

```bash
# From your laptop — open tunnel: local port 3000 → c0-0 port 3000
ssh -L 3000:localhost:3000 <username>@ificluster.its.uit.no

# Now on c0-0:
cd ~/INF-3600/ChordAI/client
CHORDAI_BACKEND_URL=http://c6-4:5555 npm run dev -- --host 0.0.0.0
```

Open http://localhost:3000 in your browser.

### Terminal 2 — Ollama (on c6-4)

```bash
ssh <username>@ificluster.its.uit.no && ssh c6-4
export PATH=$HOME/local_ollama/bin:$PATH
ollama serve
```

### Terminal 3 — Go server (on c6-4)

```bash
ssh <username>@ificluster.its.uit.no && ssh c6-4
cd ~/INF-3600/ChordAI/server
go run main.go
```

---

## Optional: SVCO voice pipeline (on your laptop)

SVCO needs microphone access so it runs locally. Point it at the cluster:

```bash
cd ~/INF-3600/ChordAI
pip install -r svco/requirements.txt

# The Go server is behind the SSH tunnel on localhost:5555 — forward it first:
# (add -L 5555:c6-4:5555 to your ssh command above, or open a separate tunnel)
ssh -L 5555:c6-4:5555 <username>@ificluster.its.uit.no

CHORDAI_SERVER=http://localhost:5555 \
CHORDAI_FRONTEND=http://localhost:3000 \
python -m svco.main
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHORDAI_BACKEND_URL` | `http://localhost:5555` | Go server URL (set in Nuxt at startup) |
| `CHORDAI_SERVER` | `http://localhost:5555` | Go server URL (set in SVCO) |
| `CHORDAI_FRONTEND` | `http://localhost:3000` | Nuxt URL (set in SVCO) |

All defaults are correct for local development. On the cluster only `CHORDAI_BACKEND_URL` needs to change (to `http://c6-4:5555`).
