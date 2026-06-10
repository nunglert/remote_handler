# syntax=docker/dockerfile:1
#
# Installation-pipeline image for remote_handler.
#
# Building this image IS the installation check: every target runs a real
# `pip install` of the package from the copied sources, so if dependency
# resolution or the build backend breaks, the image build fails. The CI test
# stage then runs the suite *inside* the built image, against the installed
# package.
#
# Targets:
#   dev   -- base package + [dev] test tooling. Fast, CPU-only. Must pass.
#   full  -- attempts every optional extra (mace/torch_sim/clinamen/quests).
#            Heavy and may require CUDA; built as an allow_failure CI job.
ARG PYTHON_VERSION=3.13

# --- base: interpreter + package sources, shared by every target -----------
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
# Only the metadata + package source are needed to install the package.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip

# --- dev: base install + test tooling (the must-pass CI path) --------------
FROM base AS dev
RUN pip install ".[dev]"
# Smoke-check that the installed package imports cleanly from site-packages.
RUN python -c "import remote_handler; from remote_handler import config, workflows; print('remote_handler import OK')"
# Bake tests in too, so `docker run <image> pytest` works standalone.
COPY test ./test
CMD ["pytest"]

# --- full: attempt every optional extra (heavy; may require CUDA) ----------
FROM base AS full
# git + compilers in case an extra (or a VCS dependency) needs to build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install ".[dev,docs,mace,torch_sim,clinamen,quests]"
RUN python -c "import remote_handler; print('full install OK')"
COPY test ./test
CMD ["pytest"]
