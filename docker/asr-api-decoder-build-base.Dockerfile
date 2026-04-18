# syntax=docker/dockerfile:1.7

ARG CPU_BUILD_BASE_IMAGE=ghcr.io/wavey-ai/asr-api-cpu-build-base:main

FROM ${CPU_BUILD_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      libopus-dev \
    && rm -rf /var/lib/apt/lists/*
