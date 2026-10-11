# Shared base image: CUDA + Python + uv. Per-tool images build FROM this. (placeholder)
# Pin a concrete CUDA/Python tag at implementation time.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace
# Real images copy the repo and `uv sync` the relevant package(s).
