Current operating context

- Public API shape is Deepgram-compatible `POST /v1/listen`.
- The runtime path is `audio -> soundkit-decoder -> asr-torch -> asr-onnx -> Deepgram-shaped JSON`.
- Do not route new work back through `parakeet-rs` for serving. This repo now fronts the split ONNX/Torch stack directly.

Repos / local dependencies

- Canonical repo: `/Users/jamieb/wavey.ai/asr-api`
- Sibling repos used by the current stack:
  - `/Users/jamieb/wavey.ai/asr-onnx`
  - `/Users/jamieb/wavey.ai/asr-torch`
  - `/Users/jamieb/wavey.ai/soundkit`
  - `/Users/jamieb/wavey.ai/web-services`

Model assets

- Current target model family: `nvidia/parakeet-tdt-0.6b-v3`
- Export bundle location in object storage:
  - `s3://wavey.ai/models/parakeet-tdt-0.6b-v3/`
  - `s3://wavey.ai/models/parakeet-tdt-0.6b-v3.tar.gz`
- Expected local model directory contents:
  - `encoder.onnx`
  - `decoder.onnx`
  - `joint.enc.onnx`
  - `joint.pred.onnx`
  - `joint.joint_net.onnx`
  - `tokens.txt`
  - `vocab.txt`
  - `export.json`
  - `featurizer_cuda0.pt`

Important runtime constraints

- `asr-torch` is on `tch 0.20.0`, which expects PyTorch/libtorch `2.7.0`.
- If you change the torch runtime, retrace `featurizer_cuda0.pt` with that same runtime before debugging anything else.
- The traced featurizer and the libtorch runtime must match. Mixed versions fail in non-obvious ways.
- The ONNX path is CUDA-backed and expects GPU execution for realistic performance.

Known-good GPU host

- Host: `scratch-fm-gpu-de-fra-2-1`
- Public IP: `172.238.102.200`
- Known-good Python env: `/root/bench/venv-torch27`
- Known-good model dir on host: `/root/asr-export/out/parakeet-tdt-0.6b-v3`

Known-good host env

- `LIBTORCH_USE_PYTORCH=1`
- `LIBTORCH_BYPASS_VERSION_CHECK=1`
- `LD_LIBRARY_PATH=/root/bench/venv-torch27/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}`
- `LD_PRELOAD=/root/bench/venv-torch27/lib/python3.12/site-packages/torch/lib/libc10_cuda.so:/root/bench/venv-torch27/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so:/root/bench/venv-torch27/lib/python3.12/site-packages/torch/lib/libtorch_cuda_linalg.so:${LD_PRELOAD:-}`
- `ASR_MODEL_DIR=/root/asr-export/out/parakeet-tdt-0.6b-v3`
- `ASR_DEVICE_IDS=0`
- `ASR_TORCH_SESSIONS=1`
- `ASR_ONNX_SESSIONS=1`
- `PORT=8443`

GPU smoke test

1. Start the server with the env above.
2. Health check:

```bash
curl --http2 -k https://172.238.102.200:8443/healthz
```

3. Listen test:

```bash
curl --http2 -k \
  -H 'Content-Type: audio/wav' \
  --data-binary @sample.wav \
  'https://172.238.102.200:8443/v1/listen?utterances=true&paragraphs=true'
```

Current expected result on the bundled sample:

- Transcript: `A tusk is used to make costly gifts.`

Operational reminders

- `web-service` expects local TLS assets when running the current dev server path. If startup fails in TLS initialization, check sibling `web-services/tls/local.wavey.ai`.
- `upload-response` is not used for `/v1/listen` because the endpoint returns one final JSON payload, not a streamed transcript.
- If you need to regenerate model assets, do it from `asr-onnx/export/export_parakeet_tdt.py` and `asr-torch/trace_featurizer.py`, not by trying to split `parakeet-rs` artifacts after export.
