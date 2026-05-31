# ChordAI Cluster Setup Guide

This document explains how to set up, route, and run the ChordAI application across a computing cluster. The system architecture is distributed across three environments to optimize performance and resource usage.

* **Your Local Computer:** Acts solely as the viewer, connecting via a secure SSH tunnel (port-forwarding).
* **Cluster Entry Node (`c0-0`):** Handles high-bandwidth internet downloads, frontend Nuxt routing, and acts as the gateway.
* **GPU Node (`c6-4`):** Handles heavy hardware-bound computations, including the Go backend server, Python scripts, and Ollama language models.

---

## Phase 1: High-Speed Downloads (Run on c0-0)

Because the entry node has high network bandwidth, all heavy downloads and standard frontend setups should be done here. The cluster utilizes a Network File System (NFS), meaning files downloaded to your user directory on c0-0 will automatically be available when you switch to the GPU node later.

**1. Log into the entry node:**
```bash
ssh yourusername@ificluster.ifi.uit.no
ssh c0-0
```

**2. Frontend Dependencies:**
Navigate to the client folder and pull down the Node packages required for the user interface.
```bash
cd ~/ChordAI/client
npm install
```

**3. Download Ollama & Models:**
Download the raw binary and extract it. To download the Llama and Gemma models, we must temporarily start the Ollama server in the background (using &), pull the models, and then terminate the server (kill %1) so it doesn't drain resources on the entry node. Afterwards install nuxt.

```bash
ssh c6-4
mkdir -p local_ollama && cd local_ollama
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | tar --zstd -xvf -
export PATH=$HOME/local_ollama/bin:$PATH
ollama pull llama3.2
ollama pull gemma4
```

**4. Install Nuxt:**
```bash
npm i nuxt
```

**5. Download Chord-Gen Datasets:**
Fetch the raw data used to train the music models. Do not run make setup here, as that creates a Python environment.

```bash
cd ~/ChordAI/chord-gen
make data
```

## Phase 2: Hardware Bindings & Training (Run on c6-4)
You must switch to the GPU node for this phase. Compiling software or installing Python dependencies on c0-0 will result in missing hardware bindings, causing the backend to crash or run extremely slowly.
**1. Access the GPU Node**
Open your terminal and establish an SSH connection directly to the computation node.
```bash
ssh yourusername@ificluster.ifi.uit.no
ssh c6-4
```

**2. Audio Processing Dependencies**

The chord detection engine requires ffmpeg to function correctly. Since administrator commands are unavailable, you must download the source code, configure it to install in your local user directory, and compile it

```bash

wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar xf ffmpeg-release-amd64-static.tar.xz
mkdir -p ~/bin
mv ffmpeg-*-static/ffmpeg ~/bin/
Add ffmpeg to path:
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**3. Train the Chord Generator**
Now that you are on the GPU node with the correct hardware bindings, you can safely create the virtual environment and train the models. This process will take a few hours.
```bash
cd ~/ChordAI/chord-gen
make setup
make train
make sample
```

## Daily Startup Guide
To run the full application, you will need to open four separate terminal windows on your local machine to manage the distributed services simultaneously.

### Terminal 1: SSH Tunnel & Frontend
Open a connection to the cluster while forwarding port 8000, log into the entry node, and start the Nuxt server. The --host 0.0.0.0 flag forces the server to broadcast on all network interfaces so your SSH tunnel can pick up the traffic.

```Bash
ssh -L 8000:localhost:3000 yourusername@ificluster.ifi.uit.no
ssh c0-0
cd ~/ChordAI/client
npm run dev -- --host 0.0.0.0
```
### Terminal 2: Ollama Engine
Open a new terminal, log into the cluster, and jump to the GPU node. Ensure your local installation is in your system PATH, and start the background service so the Go server can generate AI text and lyric responses.

```Bash
ssh yourusername@ificluster.ifi.uit.no
ssh c6-4
export PATH=$HOME/local_ollama/bin:$PATH
ollama serve
```
### Terminal 3: Go Backend (Generators & WebSockets)
Open a third terminal, connect to the GPU node, and start the primary Go backend. It will automatically bind to port 5555 and listen for standard API requests forwarded from the frontend proxy.

```Bash
ssh yourusername@ificluster.ifi.uit.no
ssh c6-4
cd ~/ChordAI/server
go run main.go
```
### Terminal 4: Python Audio API (Chord Recognizer)
Open a fourth terminal, connect to the GPU node, and start the SVCO audio transcription server. You must activate your virtual environment so the deep learning libraries (Madmom/PyTorch) are available. This server will handle audio file POST requests sent from the frontend to the /verify-chords endpoint.

```Bash
ssh yourusername@ificluster.ifi.uit.no
ssh c6-4
cd ~/ChordAI
source env/bin/activate
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
(Note: If your FastAPI server runs on port 8000, ensure your Nuxt proxy settings in nuxt.config.ts are configured to route audio uploads to http://c6-4:8000)
