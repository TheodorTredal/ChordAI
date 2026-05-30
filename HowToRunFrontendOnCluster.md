
This document describes how to set up, route and run ChordAI (nuxt-frontend, GO-backend and Ollama/gemma), 
across the clusters entry node c0-0 and the GPU node c6-4


## Architecture 
Since heavy language models needs a decent GPU, is the system split up into three parts
1. Your Local computer
2. The cluster entry node (c0-0): Runs Nuxt-frontend and forwards GPU traffic
3. GPU-node (C6-4): Runs the GO backend, Python scripts and local Ollama.  

ps. If you want to change the GPU from c6-4 node go to "nuxt.config.ts"


## One Time Preperations

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
ssh -L 8000:localhost:3000 brukernavn@ificluster.its.uit.no

#### 2. Navigate to the frontend folder:
cd ChordAI/client

#### 3. Start Nuxt bound til alle ports:
npm run dev -- --host 0.0.0.0

---------------------------------------------------------------------------------------------------------

#### Terminal 2: Ollama

Ollama needs to run in the background on the GPU node to enable the LLM

# 1. Log into the GPU node:
ssh brukernavn@ificluster.its.uit.no
ssh c6-4

# 2. make sure that the terminal has Ollama in PATH
export PATH=$HOME/local_ollama/bin:$PATH

# 3. Start the LLM engine:
ollama serve


---------------------------------------------------------------------------------------------------------

#### Terminal 3: GO-backend

# 1. Logg inn til GPU-noden:
ssh brukernavn@ificluster.its.uit.no
ssh c6-4

# 2. Naviger til server-mappen:
cd INF-3600/DiffusionTester/ChordAI/server

# 3. Start the Go-serveren(will listen to standardport 5555):

go run main.go


