# The image AgentCLI runs shell commands in (see sandbox/docker.py).
# Built once, on first use, as agentcli-sandbox:<version>.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ripgrep ca-certificates procps \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv \
    && mkdir -p /opt/venv /workspace && chmod 777 /opt/venv /workspace

# Project dependencies installed inside the sandbox go to /opt/venv, a Docker volume per
# project, so the workspace's own .venv (built for the host) is left alone.
ENV VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/opt/venv/.uv-cache \
    PIP_CACHE_DIR=/tmp/pip-cache \
    PATH=/opt/venv/bin:$PATH \
    HOME=/tmp

LABEL org.agentcli.sandbox="1"
WORKDIR /workspace
