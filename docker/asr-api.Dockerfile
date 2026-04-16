# syntax=docker/dockerfile:1.7

ARG GPU_BUILD_BASE_IMAGE=ghcr.io/wavey-ai/asr-api-gpu-build-base:main
ARG GPU_RUNTIME_BASE_IMAGE=ghcr.io/wavey-ai/asr-api-gpu-runtime-base:main

FROM ${GPU_BUILD_BASE_IMAGE} AS build

RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.88.0

ENV PATH=/root/.cargo/bin:/usr/local/bin:${PATH}
ENV ORT_CUDA_VERSION=12

WORKDIR /app

COPY Cargo.toml Cargo.lock /app/
COPY src /app/src

RUN --mount=type=secret,id=github_token,required=true \
    --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    --mount=type=cache,target=/app/target,sharing=locked \
    set -eu; \
    token="$(cat /run/secrets/github_token)"; \
    git config --global url."https://x-access-token:${token}@github.com/".insteadOf "https://github.com/"; \
    cargo build --release --locked; \
    git config --global --unset-all url."https://x-access-token:${token}@github.com/".insteadOf; \
    mkdir -p /opt/asr-bin; \
    cp /app/target/release/asr-api /opt/asr-bin/asr-api

FROM ${GPU_RUNTIME_BASE_IMAGE} AS runtime

COPY --from=build /opt/asr-bin/asr-api /usr/local/bin/asr-api

EXPOSE 8443

ENTRYPOINT ["asr-api"]
