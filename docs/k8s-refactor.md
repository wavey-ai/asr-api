# Kubernetes Refactor Notes

## Goal

Deploy `asr-api` across multiple GPU nodes so uploads can land on any ingress replica and GPU-backed workers can scale horizontally across one or more Ada 4000 nodes.

## Current constraint

`upload-response` is still ingress-local and in-memory:

- request slots live in a local `ChunkCache`
- response slots live in a local `ChunkCache`
- the waiter for the final HTTP response is a local oneshot channel

That is not a distributed cache. The first split therefore works by making workers talk back to the owning ingress pod over the internal `/_upload_response/...` API instead of trying to globalize the cache itself.

## Recommendation

Do not make the request cache a global DaemonSet.

A DaemonSet is useful for node-local model assets, not for request bodies. A node-local request cache would force sticky routing and node-aware worker assignment, and it becomes awkward as soon as ingress and GPU workers are not guaranteed to land on the same node.

## Recommended split

### 1. Edge ingress deployment

Responsibilities:

- terminate public HTTP/2 and HTTP/3
- accept `POST /v1/listen`
- decode and normalize uploaded audio on CPU
- stream normalized mono 16 kHz PCM into `upload-response`
- keep the client connection open until the final response is ready
- expose `/_upload_response/...` for remote workers

Scale:

- normal `Deployment`
- multiple replicas behind a `Service` / ingress
- CPU-oriented autoscaling

### 2. GPU transcription workers

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

## First split handoff

The first deployment does not add Redis or another external queue. Instead:

- ingress pods own the request/response cache
- GPU workers discover ingress pod IPs through a headless Service
- workers claim streams over the internal cache API
- workers read cached PCM slots and write final response slots back to the owning ingress pod

That is enough to split CPU ingest/transcode from GPU ASR while keeping the handoff inside the Wavey cache primitives.

## What eventually needs to be global

If the internal cache mesh stops scaling cleanly, the next externalization target is PCM/job manifests, not raw upload bytes:

- object storage for larger normalized PCM payloads
- NATS JetStream / Kafka / Postgres for manifests, progress, ownership, and final response routing

## What should be node-local

Use a DaemonSet or equivalent node-local warmer only for model assets:

- download `encoder.onnx`, `decoder.onnx`, `joint.*.onnx`, `tokens.txt`, `vocab.txt`, `featurizer_cuda*.pt`
- stage them onto a `hostPath` or node-local PVC
- mount that path into GPU worker pods

That avoids repeated multi-gigabyte model downloads per pod and keeps cold starts predictable.

## Practical rollout order

1. Refactor `asr-api` so the current inline request path becomes a worker that can consume `upload-response` streams.
2. Split CPU ingress from GPU workers over the internal cache API.
3. Add a model-cache DaemonSet for GPU nodes when cold starts matter.
4. Externalize PCM/job manifests only if the internal cache handoff becomes the bottleneck.
5. Add autoscaling separately for CPU ingress and GPU workers.

## Short version

- `upload-response` is the right ingress primitive
- the request cache should stay local, not global
- the first split can use the cache API itself as the cross-pod handoff
- a DaemonSet makes sense for model cache warming, not for request-body cache
