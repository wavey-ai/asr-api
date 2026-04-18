# syntax=docker/dockerfile:1.7

ARG CPU_RUNTIME_BASE_IMAGE=ghcr.io/wavey-ai/asr-api-cpu-base:main

FROM ${CPU_RUNTIME_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      libopus0 \
    && rm -rf /var/lib/apt/lists/*
