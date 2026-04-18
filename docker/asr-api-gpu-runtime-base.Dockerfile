# syntax=docker/dockerfile:1.7

ARG CUDA_RUNTIME_IMAGE=docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ARG LIBTORCH_VERSION=2.7.0
ARG LIBTORCH_CUDA_SUFFIX=cu128
ARG ONNXRUNTIME_VERSION=1.24.4

FROM ${CUDA_RUNTIME_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG LIBTORCH_VERSION
ARG LIBTORCH_CUDA_SUFFIX
ARG ONNXRUNTIME_VERSION
ARG LIBTORCH_ZIP_URL=https://download.pytorch.org/libtorch/${LIBTORCH_CUDA_SUFFIX}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_CUDA_SUFFIX}.zip
ARG ONNXRUNTIME_GPU_TGZ_URL=https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      libgomp1 \
      libcusparselt0-cuda-12 \
      unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "${LIBTORCH_ZIP_URL}" -o /tmp/libtorch.zip \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm -f /tmp/libtorch.zip

RUN curl -fsSL "${ONNXRUNTIME_GPU_TGZ_URL}" -o /tmp/onnxruntime.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/onnxruntime.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm -f /tmp/onnxruntime.tgz

RUN ldconfig /opt/libtorch/lib /opt/onnxruntime/lib

ENV LIBTORCH=/opt/libtorch
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV ORT_SKIP_DOWNLOAD=1
ENV ORT_PREFER_DYNAMIC_LINK=1
ENV ASR_ONNX_RUNTIME_LIB=/opt/onnxruntime/lib/libonnxruntime.so
ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcusparseLt/12:/opt/libtorch/lib:/opt/onnxruntime/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib
ENV LD_PRELOAD=/opt/libtorch/lib/libc10_cuda.so:/opt/libtorch/lib/libtorch_cuda.so:/opt/libtorch/lib/libtorch_cuda_linalg.so
ENV RUST_LOG=asr_api=info,web_service=info,upload_response=info
