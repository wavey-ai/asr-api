# syntax=docker/dockerfile:1.7

ARG CUDA_DEVEL_IMAGE=docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ARG GPU_RUNTIME_BASE_IMAGE=ghcr.io/wavey-ai/asr-api-gpu-runtime-base:main

FROM ${GPU_RUNTIME_BASE_IMAGE} AS runtime-base

FROM ${CUDA_DEVEL_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      clang \
      cmake \
      curl \
      git \
      libgomp1 \
      libprotobuf-dev \
      libssl-dev \
      ninja-build \
      pkg-config \
      protobuf-compiler \
      unzip \
      zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.88.0

ENV PATH=/root/.cargo/bin:/usr/local/bin:${PATH}

COPY --from=runtime-base /opt/libtorch /opt/libtorch
COPY --from=runtime-base /opt/onnxruntime /opt/onnxruntime

RUN ldconfig /opt/libtorch/lib /opt/onnxruntime/lib

ENV LIBTORCH=/opt/libtorch
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV ORT_SKIP_DOWNLOAD=1
ENV ORT_PREFER_DYNAMIC_LINK=1
ENV ASR_ONNX_RUNTIME_LIB=/opt/onnxruntime/lib/libonnxruntime.so
ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
ENV ORT_CUDA_VERSION=12
ENV PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/local/lib/pkgconfig
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcusparseLt/12:/opt/libtorch/lib:/opt/onnxruntime/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib
