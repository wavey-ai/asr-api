# syntax=docker/dockerfile:1.7

ARG CUDA_DEVEL_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ARG OPUS_VERSION=1.5.2
ARG PYTHON_SITE_PACKAGES=/usr/local/lib/python3.10/dist-packages
ARG TORCH_VERSION=2.7.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

FROM ${CUDA_DEVEL_IMAGE} AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG OPUS_VERSION
ARG PYTHON_SITE_PACKAGES
ARG TORCH_VERSION
ARG TORCH_INDEX_URL

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      apt-transport-https \
      build-essential \
      ca-certificates \
      clang \
      cmake \
      curl \
      file \
      git \
      gnupg \
      libprotobuf-dev \
      libssl-dev \
      lsb-release \
      ninja-build \
      pkg-config \
      protobuf-compiler \
      python3 \
      python3-dev \
      python3-pip \
      software-properties-common \
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

RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.88.0

ENV PATH=/root/.cargo/bin:/usr/local/bin:${PATH}
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
ENV LIBTORCH_USE_PYTORCH=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV ORT_CUDA_VERSION=12
ENV LD_LIBRARY_PATH=/usr/local/lib:${PYTHON_SITE_PACKAGES}/torch/lib

RUN python3 -m pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION}

WORKDIR /app

COPY Cargo.toml Cargo.lock /app/
COPY src /app/src

RUN --mount=type=secret,id=github_token,required=true \
    set -eu; \
    token="$(cat /run/secrets/github_token)"; \
    git config --global url."https://x-access-token:${token}@github.com/".insteadOf "https://github.com/"; \
    cargo build --release --locked; \
    git config --global --unset-all url."https://x-access-token:${token}@github.com/".insteadOf; \
    mkdir -p /opt/asr-runtime-libs; \
    find /app/target/release \
      \( -name 'libonnxruntime_providers*.so*' -o -name 'libonnxruntime*.so*' \) \
      -exec cp -L {} /opt/asr-runtime-libs/ \; \
    && ls -l /opt/asr-runtime-libs

FROM ${CUDA_RUNTIME_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_SITE_PACKAGES

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/lib/libopus.so* /usr/local/lib/
COPY --from=build ${PYTHON_SITE_PACKAGES}/torch/lib /opt/libtorch/lib
COPY --from=build /opt/asr-runtime-libs/ /usr/local/lib/
COPY --from=build /app/target/release/asr-api /usr/local/bin/asr-api

ENV LIBTORCH_USE_PYTORCH=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/libtorch/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
ENV LD_PRELOAD=/opt/libtorch/lib/libc10_cuda.so:/opt/libtorch/lib/libtorch_cuda.so:/opt/libtorch/lib/libtorch_cuda_linalg.so
ENV RUST_LOG=asr_api=info,web_service=info,upload_response=info

EXPOSE 8443

ENTRYPOINT ["asr-api"]
