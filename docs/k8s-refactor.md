# Kubernetes Refactor Notes

## Goal

Deploy `transcriber` across multiple GPU nodes so uploads can land on any ingress replica and GPU-backed workers can scale horizontally across one or more Ada 4000 nodes.

## Current constraint

`upload-response` is currently process-local and in-memory:

- request slots live in a local `ChunkCache`
- response slots live in a local `ChunkCache`
- the waiter for the final HTTP response is a local oneshot channel

That is good for single-process or same-pod handoff. It is not a distributed cache.

## Recommendation

Do not make the request cache a global DaemonSet.

A DaemonSet is useful for node-local model assets, not for request bodies. A node-local request cache would force sticky routing and node-aware worker assignment, and it becomes awkward as soon as ingress and GPU workers are not guaranteed to land on the same node.

## Recommended split

### 1. Edge ingress deployment

Responsibilities:

- terminate public HTTP/2 and HTTP/3
- accept `POST /v1/listen`
- stream upload bytes into `upload-response`
- keep the client connection open until the final response is ready

Scale:

- normal `Deployment`
- multiple replicas behind a `Service` / ingress
- CPU-oriented autoscaling

### 2. Transcode stage

Responsibilities:

- tail `upload-response` request slots
- decode and resample through `soundkit-decoder`
- emit normalized `16 kHz mono f32` PCM chunks
- compute request metadata such as SHA-256 and duration

Recommended placement:

- first iteration: same pod as the edge ingress
- later, if needed, split into a separate worker deployment only after the PCM/result handoff is externalized

### 3. GPU transcription workers

Responsibilities:

- consume PCM chunk jobs
- run `asr-torch` featurization
- run `asr-onnx` decoding
- emit partial or final transcript data

Scale:

- `Deployment` pinned to GPU nodes
- `resources.limits.nvidia.com/gpu: 1`
- one pod per GPU is the simplest starting point

This is the layer that scales when more Ada 4000 nodes are added.

## What should be global

Use a real shared backend for cross-pod handoff:

- object storage for larger PCM chunk payloads or finalized normalized audio
- Redis / NATS JetStream / Kafka / Postgres for job manifests, progress, ownership, and final response routing

The exact store can vary, but it needs to be shared across pods and nodes.

## What should be node-local

Use a DaemonSet or equivalent node-local warmer only for model assets:

- download `encoder.onnx`, `decoder.onnx`, `joint.*.onnx`, `tokens.txt`, `vocab.txt`, `featurizer_cuda*.pt`
- stage them onto a `hostPath` or node-local PVC
- mount that path into GPU worker pods

That avoids repeated multi-gigabyte model downloads per pod and keeps cold starts predictable.

## Practical rollout order

1. Refactor `transcriber` so the current inline request path becomes a worker that can consume `upload-response` streams.
2. Keep ingress and worker in the same pod first, so the existing local `ChunkCache` model still works.
3. Add an internal PCM/job handoff that is externalized into shared storage.
4. Move GPU transcription to a separate GPU deployment.
5. Add a model-cache DaemonSet for GPU nodes.
6. Add autoscaling separately for CPU ingress and GPU workers.

## Short version

- `upload-response` is the right ingress primitive
- the request cache should stay local, not global
- cross-node scaling needs a shared job/result backend
- a DaemonSet makes sense for model cache warming, not for request-body cache
