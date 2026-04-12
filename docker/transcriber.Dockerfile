# syntax=docker/dockerfile:1.7

ARG CUDA_DEVEL_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ARG PYTHON_SITE_PACKAGES=/usr/local/lib/python3.10/dist-packages
ARG TORCH_VERSION=2.7.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

FROM ${CUDA_DEVEL_IMAGE} AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_SITE_PACKAGES
ARG TORCH_VERSION
ARG TORCH_INDEX_URL

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      clang \
      cmake \
      curl \
      git \
      libopus-dev \
      libssl-dev \
      pkg-config \
      python3 \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.87.0

ENV PATH=/root/.cargo/bin:${PATH}
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN python3 -m pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION}

ENV LIBTORCH_USE_PYTORCH=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV LD_LIBRARY_PATH=${PYTHON_SITE_PACKAGES}/torch/lib

WORKDIR /app

COPY Cargo.toml Cargo.lock /app/
COPY src /app/src

RUN --mount=type=secret,id=github_token,required=true \
    set -eux; \
    token="$(cat /run/secrets/github_token)"; \
    git config --global url."https://x-access-token:${token}@github.com/".insteadOf "https://github.com/"; \
    cargo build --release --locked; \
    git config --global --unset-all url."https://x-access-token:${token}@github.com/".insteadOf

FROM ${CUDA_RUNTIME_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_SITE_PACKAGES
ARG TORCH_VERSION
ARG TORCH_INDEX_URL

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      libopus0 \
      python3 \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN python3 -m pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION}

ENV LIBTORCH_USE_PYTORCH=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV LD_LIBRARY_PATH=${PYTHON_SITE_PACKAGES}/torch/lib
ENV LD_PRELOAD=${PYTHON_SITE_PACKAGES}/torch/lib/libc10_cuda.so:${PYTHON_SITE_PACKAGES}/torch/lib/libtorch_cuda.so:${PYTHON_SITE_PACKAGES}/torch/lib/libtorch_cuda_linalg.so
ENV RUST_LOG=transcriber=info,web_service=info

COPY --from=build /app/target/release/transcriber /usr/local/bin/transcriber

EXPOSE 8443

ENTRYPOINT ["transcriber"]
