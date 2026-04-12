# syntax=docker/dockerfile:1.7

ARG CUDA_DEVEL_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ARG OPUS_VERSION=1.5.2
ARG ONNXRUNTIME_COMMIT=986b66af96252488bcf885741623ba877964baca
ARG CUDA_ARCHITECTURES=89
ARG PYTHON_SITE_PACKAGES=/usr/local/lib/python3.10/dist-packages
ARG TORCH_VERSION=2.7.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

FROM ${CUDA_DEVEL_IMAGE} AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG OPUS_VERSION
ARG ONNXRUNTIME_COMMIT
ARG CUDA_ARCHITECTURES
ARG PYTHON_SITE_PACKAGES
ARG TORCH_VERSION
ARG TORCH_INDEX_URL

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      apt-transport-https \
      build-essential \
      ca-certificates \
      clang \
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

RUN ln -sfn /usr/local/cuda-12.8 /usr/local/cuda

RUN python3 -m pip install --no-cache-dir cmake==3.30.9 psutil

ENV PATH=/usr/local/bin:${PATH}

RUN git clone --recursive https://github.com/microsoft/onnxruntime.git /opt/onnxruntime \
    && cd /opt/onnxruntime \
    && git checkout "${ONNXRUNTIME_COMMIT}" \
    && git submodule update --init --recursive \
    && ./build.sh --allow_running_as_root --config Release --build_shared_lib --parallel \
      --use_cuda \
      --cuda_home /usr/local/cuda-12.8 \
      --cudnn_home /usr/lib/x86_64-linux-gnu \
      --skip_tests \
      --cmake_extra_defines CMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" ONNX_USE_LTO=OFF \
    && mkdir -p /usr/local/include/onnxruntime \
    && cp build/Linux/Release/libonnxruntime*.so* /usr/local/lib/ \
    && cp -r include/* /usr/local/include/onnxruntime/ \
    && ldconfig

RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.88.0

ENV PATH=/root/.cargo/bin:${PATH}
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

RUN python3 -m pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION}

ENV PATH=/usr/local/bin:/root/.cargo/bin:${PATH}

ENV LIBTORCH_USE_PYTORCH=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV ORT_LIB_LOCATION=/usr/local/lib
ENV ORT_PREFER_DYNAMIC_LINK=1
ENV ORT_SKIP_DOWNLOAD=1
ENV LD_LIBRARY_PATH=/usr/local/lib:${PYTHON_SITE_PACKAGES}/torch/lib

WORKDIR /app

COPY Cargo.toml Cargo.lock /app/
COPY src /app/src

RUN --mount=type=secret,id=github_token,required=true \
    set -eu; \
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
      python3 \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN python3 -m pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION}

ENV LIBTORCH_USE_PYTORCH=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV ORT_LIB_LOCATION=/usr/local/lib
ENV ORT_PREFER_DYNAMIC_LINK=1
ENV ORT_SKIP_DOWNLOAD=1
ENV LD_LIBRARY_PATH=/usr/local/lib:${PYTHON_SITE_PACKAGES}/torch/lib:/usr/lib/x86_64-linux-gnu
ENV LD_PRELOAD=${PYTHON_SITE_PACKAGES}/torch/lib/libc10_cuda.so:${PYTHON_SITE_PACKAGES}/torch/lib/libtorch_cuda.so:${PYTHON_SITE_PACKAGES}/torch/lib/libtorch_cuda_linalg.so
ENV RUST_LOG=asr_api=info,web_service=info,upload_response=info

COPY --from=build /usr/local/lib/libonnxruntime*.so* /usr/local/lib/
COPY --from=build /usr/local/lib/libopus.so* /usr/local/lib/
COPY --from=build /app/target/release/asr-api /usr/local/bin/asr-api

RUN ldconfig

EXPOSE 8443

ENTRYPOINT ["asr-api"]
