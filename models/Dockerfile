# Python image for all AI model scripts (lyric generator, image generator,
# and chord model inference via sample.py).
# Uses CUDA 12.1 base so GPU passthrough works with nvidia-container-toolkit.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3 resolve to 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Install model script deps first (layer cached unless requirements change)
COPY models/requirements.txt ./models/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r models/requirements.txt

# Install chord-gen deps
COPY chord-gen/requirements.txt ./chord-gen/requirements.txt
RUN pip install --no-cache-dir -r chord-gen/requirements.txt

# Copy the full project so subprocess paths resolve correctly
COPY models/ ./models/
COPY chord-gen/ ./chord-gen/

# out-chords is the shared output dir (mounted as a volume in compose)
RUN mkdir -p /app/out-chords

# This image is not a standalone service — it's used as a base layer
# referenced by the Go server via subprocess calls.
# The Go server mounts the project root; this image is just for
# `docker compose run models python3 models/lyric_generator_gemma4.py ...`
# or as a sidecar in future microservice splits.
CMD ["python3", "--version"]
