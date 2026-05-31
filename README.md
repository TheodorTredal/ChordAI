# ChordAI

An AI-powered music orchestration platform. Describe a song in plain language to generate original chord progressions, lyrics, and album art, or use the built-in audio analysis engine to transcribe chords directly from audio files.

## Architecture & Components

ChordAI utilizes a decoupled architecture. This design allows the web-based frontend client and the Go-based orchestrator (REST API) to run on a standard computer or entry node, while the computationally heavy neural network generation can be routed to a dedicated GPU node.

The project is divided into several dedicated modules. **Please refer to the `README.md` files located inside the following directories for comprehensive technical explanations of their specific functions:**

* `chord-gen/`: Contains the custom Python/PyTorch chord-progression model, including data streaming and training pipelines.
* `models/`: Houses the Python scripts for image generation (Diffusers) and lyric generation (Gemma 4).
* `svco/`: Contains the offline audio processing and chord transcription engines.
* `client/`: Contains the Nuxt 3 / Vue 3 frontend web application.
* `server/`: The Go backend that orchestrates the AI pipeline 

## Components

| Component | Stack | Purpose | Documentation |
|-----------|-------|---------|---------------|
| [`client/`](client/) | Nuxt 3, Vue 3 | Chat-style web interface | [Read README](client/README.md) |
| [`server/`](server/) | Go, Gin framework | REST + WebSocket API; orchestrates the AI pipelines | [Read README](server/README.md) |
| [`chord-gen/`](chord-gen/) | Python, PyTorch | Custom sequence-based chord progression model (train & sample) | [Read README](chord-gen/README.md) |
| [`models/`](models/) | Python, Ollama, Diffusers | Dedicated script environment for lyric and album art generation | [Read README](models/README.md) |
| [`svco/`](svco/) | Python, Librosa | Offline audio processing and chord transcription engine | [Read README](svco/README.md) |



## Running the System on the Cluster

Running the generative models efficiently requires routing through dedicated university GPU nodes. For the complete guide on setting up the SSH tunnels, starting the local Ollama instance, and running the Nuxt frontend across the cluster nodes (c0-0 and c6-4), please refer to the [HowToRunFrontendOnCluster.md](HowToRunFrontendOnCluster.md) document. 

## Running Locally (Unsupported)

Due to hardware limitations, we do not have the resources to run the full AI stack locally. The system has not been executed or tested in a purely local environment during development, and we cannot guarantee that it will function correctly without the cluster's dedicated GPU capabilities. 

If you still wish to attempt a local setup, the process follows the exact same underlying logic as the cluster guide (`HowToRunFrontendOnCluster.md`), but requires the following prerequisites modifications:

### Prerequisites
- Go 1.21+
- Node 18+ (with npm)
- Python 3.10+
- [Ollama](https://ollama.com) installed and added to your system `$PATH`
- A CUDA-capable GPU (Required for image generation; strongly recommended for chord model training)

1. **Remove Remote Connections:** Skip all steps involving SSH tunneling and logging into `c0-0` or `c6-4`. 
2. **Local Installations:** You must install Ollama, Python, and Go directly onto your own machine and execute their respective start commands locally. 
3. **Frontend Adjustments:** When starting the Nuxt frontend, you can simply run the development server normally (`npm run dev`). You do not need to bind it to all network ports (`--host 0.0.0.0`) since you are not forwarding traffic from a remote server.
4. **Routing:** Ensure that the API target routes in your Nuxt configuration point to `localhost` instead of the cluster nodes. It is highly recommended to configure this using Environment Variables rather than hardcoding the URLs directly into the configuration files.
