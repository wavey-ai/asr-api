# asr-api

`asr-api` serves Deepgram-style prerecorded transcription over Wavey's `web-service` stack.

It now supports three runtime roles:

1. `monolith`: one process owns `/v1/listen`, decodes audio to normalized PCM, and runs the local ASR worker against the in-process `upload-response` cache.
2. `ingress`: CPU-oriented front door. It accepts `POST /v1/listen`, decodes supported audio through `soundkit-decoder`, normalizes to mono 16 kHz PCM, writes those PCM chunks into `upload-response`, and waits for a worker response.
3. `worker`: GPU-oriented worker. It discovers ingress pods over the internal cache API, claims cached streams, runs `asr-torch` featurization plus `asr-onnx` decoding, then writes the final Deepgram-compatible JSON response back to the owning ingress pod.

The split is deliberate:

- audio decode / resample / downmix stays on CPU ingress nodes
- featurization stays with transcription on GPU nodes because `asr-torch` is CUDA-backed
- the shared handoff stays on `upload-response`; there is no Redis sidecar or external queue in this first split

## Environment

- `ASR_MODEL_DIR` (required): directory containing the split TDT ONNX files
  - encoder: `encoder.fp16.onnx`, `encoder.onnx`, or `encoder.int8.onnx`
  - decoder: `decoder.fp16.onnx`, `decoder.onnx`, or `decoder.int8.onnx`
  - joint encoder: `joint.enc.fp16.onnx`, `joint.enc.onnx`, or `joint.enc.int8.onnx`
  - joint predictor: `joint.pred.fp16.onnx`, `joint.pred.onnx`, or `joint.pred.int8.onnx`
  - joint net: `joint.joint_net.fp16.onnx`, `joint.joint_net.onnx`, or `joint.joint_net.int8.onnx`
  - vocabulary: `tokens.txt`, or `vocab.txt` / `ASR_VOCAB_PATH`
- `ASR_VOCAB_PATH`: optional fallback vocab path when `tokens.txt` is not present in the model dir
- `ASR_DEVICE_IDS`: comma-separated GPU ids, default `0`
- `ASR_TORCH_SESSIONS`: featurizer sessions per device, default `1`
- `ASR_ONNX_SESSIONS`: decoder sessions per device, default `1`
- `ASR_API_ROLE`: `monolith`, `ingress`, or `worker`
- `PORT`: TLS port, default `8443`
- `ENABLE_H3`: enable HTTP/3 in addition to HTTP/2
- `TLS_CERT_PATH` / `TLS_KEY_PATH`: optional PEM paths; if omitted the workspace's default local TLS material is used
- `CHUNK_SECONDS`: transcription window length, default `30`
- `OVERLAP_SECONDS`: overlap between adjacent windows, default `2`
- `FINAL_MIN_SECONDS`: minimum residual tail to keep, default `0.5`
- `UTT_SPLIT_SECONDS`: pause threshold used when `utterances=true`, default `0.8`
- `UPLOAD_RESPONSE_NUM_STREAMS`: in-memory upload-response stream slots, default `128`
- `UPLOAD_RESPONSE_SLOT_SIZE_KB`: per-slot cache size, default `64`
- `UPLOAD_RESPONSE_SLOTS_PER_STREAM`: max request/response slots per stream, default `16384`
- `UPLOAD_RESPONSE_TIMEOUT_MS`: listen request timeout while waiting for the worker response, default `30000`
- `UPLOAD_RESPONSE_WATCH_POLL_MS`: response watcher poll interval, default `1`
- `UPLOAD_RESPONSE_WORKER_POLL_MS`: local worker poll interval for cached streams, default `2`
- `UPLOAD_RESPONSE_MAX_INFLIGHT`: max simultaneously claimed cached streams for the monolith worker, default `2`
- `UPLOAD_RESPONSE_WORKER_ID`: local worker identity for cache claims, default `asr-api-monolith`
- `UPLOAD_RESPONSE_INGRESS_URLS`: optional comma-separated ingress origins for worker mode
- `UPLOAD_RESPONSE_DISCOVERY_DNS`: optional `host:port` to resolve into ingress pod IPs for worker mode
- `UPLOAD_RESPONSE_DISCOVERY_INTERVAL_MS`: ingress discovery refresh interval, default `2000`
- `UPLOAD_RESPONSE_INSECURE_TLS`: allow self-signed / internal TLS for worker mode

`ASR_MODEL_DIR` is only required for `monolith` and `worker`. Pure `ingress` mode does not load the model.

## Local Run

This repo currently builds `asr-torch` through `tch`, so local development expects PyTorch to be available through Python.

macOS example:

```bash
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export DYLD_LIBRARY_PATH="$(python3 - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"

cargo run -- \
  --role monolith \
  --model-dir /path/to/parakeet-tdt
```

Split example:

```bash
cargo run -- --role ingress
```

```bash
cargo run -- \
  --role worker \
  --model-dir /path/to/parakeet-tdt \
  --upload-response-ingress-urls https://127.0.0.1:8443 \
  --upload-response-insecure-tls
```

## Request

Upload raw audio bytes directly:

```bash
curl --http2 -k \
  -H 'Content-Type: audio/wav' \
  --data-binary @sample.wav \
  'https://localhost:8443/v1/listen?utterances=true&paragraphs=true'
```

The success shape mirrors Deepgram prerecorded responses:

- `metadata`
- `results.channels[0].alternatives[0].transcript`
- `results.channels[0].alternatives[0].words`
- optional `results.utterances`
- optional `results.channels[0].alternatives[0].paragraphs`

JSON URL jobs are not implemented yet. This service currently supports uploaded audio bodies only.

## Internal Cache API

Ingress and monolith roles also serve the `upload-response` cache API for inspection and worker handoff:

- `GET /_upload_response/streams`
- `GET /_upload_response/streams/{stream_id}`
- `GET /_upload_response/streams/{stream_id}/request/last`
- `GET /_upload_response/streams/{stream_id}/request/slots/{slot_id}`
- `GET /_upload_response/streams/{stream_id}/response/last`
- `GET /_upload_response/streams/{stream_id}/response/slots/{slot_id}`
- `PUT /_upload_response/streams/{stream_id}/readers/{worker_id}`
- `DELETE /_upload_response/streams/{stream_id}/readers/{worker_id}`
- `PUT /_upload_response/streams/{stream_id}/response/claim/{worker_id}`
- `DELETE /_upload_response/streams/{stream_id}/response/claim/{worker_id}`
- `PUT /_upload_response/streams/{stream_id}/response/headers`
- `PUT /_upload_response/streams/{stream_id}/response/body`
- `PUT /_upload_response/streams/{stream_id}/response/end`

## Verification

Local verification used while wiring the service:

```bash
env LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 cargo check
env LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
  DYLD_LIBRARY_PATH="$(python3 - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)" \
  cargo test --lib
```

## Deploy

The checked-in Kubernetes shape is now split:

- CPU ingress deployment: `deploy/k8s/transcriber/ingress-deployment.yaml`
- GPU worker deployment: `deploy/k8s/transcriber/worker-deployment.yaml`
- public service + headless discovery service: `deploy/k8s/transcriber/services.yaml`
- image build: `docker/transcriber.Dockerfile`
- image workflow: `.github/workflows/build-image.yml`
- deploy workflow: `.github/workflows/deploy-main.yml`

The worker image expects CUDA plus Python-installed PyTorch 2.7 at runtime so `tch` can load the traced featurizer modules. The checked-in worker ConfigMap leaves TensorRT disabled (`ASR_ONNX_TRT_COMPONENTS=none`) for the baseline CUDA deployment. Once the TensorRT-enabled image path is ready, switch that value to `encoder,joint_enc` or another explicit component set.

The build workflow also needs a repo secret named `WAVEY_AI_GH_TOKEN` so Docker can fetch the private `asr-onnx`, `asr-torch`, and `soundkit` dependencies during image build.

See also:

- `docs/k8s-refactor.md` for the recommended split between ingress, transcode, GPU workers, and node-local model cache
- `../asr-onnx/python/requirements-export-cu128-torch210.lock.txt` for the exact ONNX export environment
- `../asr-torch/python/requirements-trace-cu128-torch27.lock.txt` for the exact featurizer trace environment
