# ChordAI Server

The Go backend acts as the core orchestrator and API gateway for the ChordAI platform. It manages network traffic, marshals internal data, and dispatches parallel execution tasks across the Python machine learning models and Ollama engines.



## Architecture & Codebase Layout

The server uses a modular structure built on top of the **Gin Web Framework**, decoupling endpoint routing from the low-level system execution wrappers.

```

server/
├── main.go             # Application entrypoint; initializes router and network listeners
├── routers/            # HTTP and WebSocket endpoint handlers
│   └── generate.go     # Manages pipeline triggers and client connections
├── schemas/            # Data transfer objects (DTOs) and structural JSON validation definitions
│   └── schemas.go

├── executor/           # OS-level process management wrapper
│   └── executor.go     # Executes underlying binaries and Python scripts asynchronously
└── models/             # Dedicated orchestration runners for individual AI domains
├── chord_runner.go # Interfaces with the PyTorch sequence generator
├── lyrics_runner.go# Interfaces with the Gemma 4 engine via Ollama
└── image_runner.go # Interfaces with Diffusers for album art graphics

```


## Core Infrastructure Responsibilities

### 1. API Routing & Connection Lifecycle
The code in `routers/` uses Gin to spin up high-performance routing matrices. It establishes bidirectional communication channels using WebSockets so that long-running AI pipeline updates (such as progressive lyric chunks or processing status) can be streamed back to the Nuxt frontend in real time, avoiding HTTP timeout failures.

### 2. Native Subprocess Execution
Since the heavy AI generation lives in Python environments, the `executor/` component wraps Go's native `os/exec` library. It provides thread-safe execution boundaries to run Python modules (like `sample.py` and `image_generator.py`) as background tasks. It captures standard output (`stdout`) and error flags (`stderr`), converting system errors into clean JSON payloads for the user interface.

### 3. Structural Data Mapping
The `schemas/` directory defines strict data contracts between the distributed subsystems. It enforces consistency across variable names, musical metrics (BPM, decade, genre), and raw audio file uploads before any computing power is spent on the AI workers.



## Network Mapping (Cluster vs. Local)

The server binds to port `:5555` by default. Depending on where you are running the system, traffic is handled in one of two ways:

* **Distributed Deployment (Cluster):** The server executes directly on the high-performance GPU node (`c6-4`). The frontend on node `c0-0` uses its built-in server proxy configuration (`nitro.devProxy`) to bridge internal network traffic seamlessly to `http://c6-4:5555`.
* **Local Deployment (Unsupported):** The server executes natively on localhost. The client configuration must be re-mapped via environment variables (`.env`) to target your local loopback address (`http://localhost:5555`) instead of the university cluster hostname.



## Standard Run Command

To initialize network bindings, compile dependencies, and boot up the orchestrator service, navigate to this directory and run:

```bash
go run main.go
```
