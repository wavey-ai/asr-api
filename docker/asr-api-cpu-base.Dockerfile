# syntax=docker/dockerfile:1.7

ARG RUNTIME_IMAGE=docker.io/ubuntu:22.04

FROM ${RUNTIME_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      libgomp1 \
      libopus0 \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib
ENV RUST_LOG=asr_api=info,web_service=info,upload_response=info
