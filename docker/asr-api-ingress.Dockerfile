# syntax=docker/dockerfile:1.7

ARG RUST_IMAGE=docker.io/rust:1.88.0-bookworm
ARG CPU_RUNTIME_BASE_IMAGE=ghcr.io/wavey-ai/asr-api-cpu-base:main

FROM ${RUST_IMAGE} AS build

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      clang \
      cmake \
      curl \
      git \
      libopus-dev \
      libprotobuf-dev \
      libssl-dev \
      pkg-config \
      protobuf-compiler \
      python3 \
      python3-dev \
      python3-pip \
      zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
ENV PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/local/lib/pkgconfig

WORKDIR /app

COPY Cargo.toml Cargo.lock /app/
COPY src /app/src

RUN --mount=type=secret,id=github_token,required=true \
    --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,target=/app/target,sharing=locked \
    set -eu; \
    token="$(cat /run/secrets/github_token)"; \
    git config --global url."https://x-access-token:${token}@github.com/".insteadOf "https://github.com/"; \
    cargo build --release --locked --no-default-features; \
    git config --global --unset-all url."https://x-access-token:${token}@github.com/".insteadOf; \
    mkdir -p /opt/asr-bin; \
    cp /app/target/release/asr-api /opt/asr-bin/asr-api

FROM ${CPU_RUNTIME_BASE_IMAGE} AS runtime

COPY --from=build /opt/asr-bin/asr-api /usr/local/bin/asr-api

EXPOSE 8443

ENTRYPOINT ["asr-api"]
