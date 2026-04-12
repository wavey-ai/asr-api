# transcriber

`transcriber` serves Deepgram-style prerecorded transcription over Wavey's `web-service` stack.

It accepts uploaded audio bytes on `POST /v1/listen`, decodes supported formats through `soundkit-decoder`, chunks the decoded mono 16 kHz PCM locally, runs `asr-torch` for mel features and `asr-onnx` for TDT decoding, then returns a Deepgram-compatible JSON payload with `results.channels[0].alternatives[0].words`.

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
- `PORT`: TLS port, default `8443`
- `ENABLE_H3`: enable HTTP/3 in addition to HTTP/2
- `TLS_CERT_PATH` / `TLS_KEY_PATH`: optional PEM paths; if omitted the workspace's default local TLS material is used
- `CHUNK_SECONDS`: transcription window length, default `30`
- `OVERLAP_SECONDS`: overlap between adjacent windows, default `2`
- `FINAL_MIN_SECONDS`: minimum residual tail to keep, default `0.5`
- `UTT_SPLIT_SECONDS`: pause threshold used when `utterances=true`, default `0.8`

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
  --model-dir /path/to/parakeet-tdt
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

The repo still contains the earlier LKE scaffolding under:

- image build: `docker/transcriber.Dockerfile`
- image workflow: `.github/workflows/build-image.yml`
- deploy workflow: `.github/workflows/deploy-main.yml`
- manifests: `deploy/k8s/transcriber/`

The Kubernetes config in this repo now points at the split ONNX model layout, but the container image path still needs a CUDA/libtorch-aware runtime before the deployed `asr-torch` stack will be usable in CI or LKE.

See also:

- `docs/k8s-refactor.md` for the recommended split between ingress, transcode, GPU workers, and node-local model cache
- `../asr-onnx/python/requirements-export-cu128-torch210.lock.txt` for the exact ONNX export environment
- `../asr-torch/python/requirements-trace-cu128-torch27.lock.txt` for the exact featurizer trace environment
