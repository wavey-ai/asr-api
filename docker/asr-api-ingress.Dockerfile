# syntax=docker/dockerfile:1.7

ARG RUST_IMAGE=rust:1.88.0-bookworm
ARG RUNTIME_IMAGE=ubuntu:22.04
ARG OPUS_VERSION=1.5.2

FROM ${RUST_IMAGE} AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG OPUS_VERSION

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      clang \
      cmake \
      curl \
      git \
      libprotobuf-dev \
      libssl-dev \
      pkg-config \
      protobuf-compiler \
      python3 \
      python3-dev \
      python3-pip \
      zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://downloads.xiph.org/releases/opus/opus-${OPUS_VERSION}.tar.gz" -o /tmp/opus.tar.gz \
    && tar -xzf /tmp/opus.tar.gz -C /tmp \
    && cd "/tmp/opus-${OPUS_VERSION}" \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && rm -rf "/tmp/opus-${OPUS_VERSION}" /tmp/opus.tar.gz

ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

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

FROM ${RUNTIME_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/lib/libopus.so* /usr/local/lib/
COPY --from=build /opt/asr-bin/asr-api /usr/local/bin/asr-api

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu
ENV RUST_LOG=asr_api=info,web_service=info,upload_response=info

EXPOSE 8443

ENTRYPOINT ["asr-api"]
