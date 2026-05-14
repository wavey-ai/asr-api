# asr-api

`asr-api` serves Deepgram-style prerecorded transcription over Wavey's `web-service` stack.

It now supports three runtime roles:

1. `ingress`: CPU-oriented front door. It accepts `POST /v1/listen`, writes raw request bytes plus request metadata into `upload-response`, and waits for a worker response.
2. `decoder`: CPU-oriented processing stage. It discovers ingress origins over the internal cache API, claims raw request streams, decodes / normalizes them to mono `16 kHz` `f32`, and writes that canonical PCM into the `decoded` stage lane.
3. `worker`: GPU-oriented response stage. It discovers ingress origins over the internal cache API, claims decoded streams, runs `asr-torch` featurization plus `asr-onnx` decoding, then writes the final Deepgram-compatible JSON response back to the owning ingress process.

The split is deliberate:

- ingress is only HTTP / WebSocket / cache ingress; it does not carry audio decode libraries
- audio decode / resample / downmix stays in a separate CPU decoder role
- featurization stays with transcription on GPU nodes because `asr-torch` is CUDA-backed
- the shared handoff stays on `upload-response`; there is no Redis sidecar or external queue in this split

## Environment

- `ASR_MODEL_DIR` (required): directory containing the ASR model bundle
  - Cohere: `encoder.onnx`, `encoder.onnx.data`, `decoder_prefill.onnx`, `decoder_prefill.onnx.data`, `decoder_cached_step.onnx`, `decoder_cached_step.onnx.data`, `tokenizer.json`, `tokenizer.model`, `config.json`, `generation_config.json`, and `preprocessor_config.json`
  - Parakeet / NeMo:
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
- `ASR_WORKER_COUNT`: local-orchestrator worker process count, default `1`
- `ASR_API_ROLE`: `ingress`, `decoder`, or `worker`
- `ASR_LOG_FORMAT`: `json`, `pretty`, or `compact`, default `json`
- `PORT`: TLS port, default `8443`
- `ENABLE_H3`: enable HTTP/3 in addition to HTTP/2
- `TLS_CERT_PATH` / `TLS_KEY_PATH`: optional PEM paths; if omitted the workspace's default local TLS material is used
- `CHUNK_SECONDS`: transcription window length, default `30`
- `OVERLAP_SECONDS`: overlap between adjacent windows, default `2`
- `FINAL_MIN_SECONDS`: minimum residual tail to keep, default `0.5`
- `UTT_SPLIT_SECONDS`: pause threshold used when `utterances=true`, default `0.8`
- `UPLOAD_RESPONSE_NUM_STREAMS`: in-memory upload-response stream slots, default `16`
- `UPLOAD_RESPONSE_SLOT_SIZE_KB`: per-slot cache size, default `32`
- `UPLOAD_RESPONSE_SLOTS_PER_STREAM`: max request/response slots per stream, default `1024`
- `UPLOAD_RESPONSE_TIMEOUT_MS`: listen request timeout while waiting for the worker response, default `30000`
- `UPLOAD_RESPONSE_WATCH_POLL_MS`: response watcher poll interval, default `1`
- `UPLOAD_RESPONSE_WORKER_POLL_MS`: local worker poll interval for cached streams, default `2`
- `UPLOAD_RESPONSE_MAX_INFLIGHT`: max simultaneously claimed cached streams per worker process, default `2`
- `UPLOAD_RESPONSE_WORKER_ID`: worker identity for cache claims, default `asr-api-worker`
- `UPLOAD_RESPONSE_WORKER_ID_PREFIX`: local-orchestrator worker identity prefix, default `asr-api-worker-local`
- `UPLOAD_RESPONSE_INGRESS_URLS`: optional comma-separated ingress origins for worker mode
- `UPLOAD_RESPONSE_DISCOVERY_DNS`: optional `host:port` to resolve into ingress origins for worker mode
- `UPLOAD_RESPONSE_DISCOVERY_INTERVAL_MS`: ingress discovery refresh interval, default `2000`
- `UPLOAD_RESPONSE_INSECURE_TLS`: allow self-signed / internal TLS for worker mode

`ASR_MODEL_DIR` is only required for `worker`. `ingress` and `decoder` mode do not load the model.

The repo no longer checks model payloads into Git. Sync them from the Wavey
bucket before starting a worker:

```bash
AWS_ACCESS_KEY_ID=... \
AWS_SECRET_ACCESS_KEY=... \
scripts/sync-model-from-bucket.sh \
  --model parakeet-tdt-0.6b-v3 \
  --dest /var/lib/asr-api/models/parakeet-tdt
```

That sync step is intended as model seeding / updating, not normal code deploy.
The helper records remote object metadata in the target directory and skips the
download when the local bundle is already current.

TensorRT engine caches can also be stored in the bucket after building them on
a compatible GPU host. Use a cache id that names the compatibility dimensions:
GPU family, CUDA/TensorRT/ONNX Runtime versions, precision, components, and
profile window. For the Ada 35s Cohere cache:

```bash
scripts/sync-trt-cache.sh pull \
  --model cohere-transcribe-03-2026 \
  --cache-id rtx4000-ada-ort1.23.2-trt10-fp16-all-35s \
  --dir /var/lib/asr-api/models/cohere-transcribe-03-2026/.trt_cache_all_35s_frames
```

Publish a cache only from the GPU host that built and validated it:

```bash
scripts/sync-trt-cache.sh push \
  --model cohere-transcribe-03-2026 \
  --cache-id rtx4000-ada-ort1.23.2-trt10-fp16-all-35s \
  --dir /var/lib/asr-api/models/cohere-transcribe-03-2026/.trt_cache_all_35s_frames
```

For standalone hosts, keep the model on persistent disk and restart only the
service binary during normal deploys. Re-run the sync helper only when you are
changing the model version, repairing a missing bundle, or forcing a refresh.

Correlation IDs use Wavey's snowflake generator and are propagated internally in `x-wavey-request-id`. If the client supplies a numeric `x-request-id`, `asr-api` will reuse it; otherwise ingress will mint one and carry it through the request path.

## Local Run

The Cohere backend links ONNX Runtime through the `ort` crate's downloaded
runtime artifacts. It does not require `ASR_ONNX_RUNTIME_LIB` or
`ORT_DYLIB_PATH` for normal builds.

On Apple Silicon, use the ONNX Runtime CoreML execution provider for the Apple
GPU/Metal path:

```bash
ASR_COHERE_COREML=true \
ASR_COHERE_COREML_COMPUTE_UNITS=cpu-and-gpu \
ASR_COHERE_COREML_CACHE_DIR=models/cohere-transcribe-03-2026/.coreml-cache-static \
  target/release/local-orchestrator --model-provider cohere --device-ids 0
```

Build only Cohere plus audio decode when you do not need the NeMo/Torch path:

```bash
cargo build --no-default-features --features cohere-backend,audio-decoder
```

The NeMo backend builds `asr-torch` through `tch`, so local development for that
path expects PyTorch to be available through Python.

macOS split example:

```bash
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export DYLD_LIBRARY_PATH="$(python3 - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"

cargo run -- \
  --role worker \
  --model-dir /path/to/parakeet-tdt \
  --upload-response-ingress-urls https://127.0.0.1:8443 \
  --upload-response-insecure-tls
```

```bash
cargo run -- --role ingress
```

```bash
cargo run -- --role decoder \
  --upload-response-ingress-urls https://127.0.0.1:8443 \
  --upload-response-insecure-tls
```

## Cohere Ada Benchmarks

These are point-in-time Cohere Transcribe measurements from the Linode NVIDIA
RTX 4000 Ada Generation host (`20475 MiB` VRAM) on `2026-05-14`. The
`asr-api` rows used release binaries, the split `ingress` / `decoder` /
`worker` upload-response path, the Cohere ONNX bundle synced from the bucket,
and Harvard `*.s16le` PCM files through `../asr-load --h2 --warmup`.

`Stage RTFx` is the measured load window after warmup. `Whole RTFx` includes
prewarm, warmup, and client/report overhead. `Response mean` is the
`asr-load` part-response mean for `asr-api` rows; the Second State row uses the
full-response mean reported from that harness.

| Runtime | Topology | Measured load | OK / fail | Stage RTFx | Whole RTFx | Mean TTFB | Response mean | GPU VRAM | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `asr-api` ONNX + TensorRT | `4` workers x `1` ONNX session, `max_inflight=1` each | `100`, c=`4` | `100 / 0` | `32.90` | `21.11` | `146.12 ms` | `298.80 ms` | `18681 MiB` | Best measured stage RTFx, but `27` internal stale response-claim warnings while clients still saw `100%` OK. |
| `asr-api` ONNX + TensorRT | `2` workers x `2` ONNX sessions, `max_inflight=2` each | `100`, c=`4` | `100 / 0` | `32.80` | `21.01` | `142.15 ms` | `314.26 ms` | `18347 MiB` | Similar throughput to `4x1`, slightly lower VRAM, `7` stale response-claim warnings. |
| `asr-api` ONNX + TensorRT | `1` worker x `4` ONNX sessions, `max_inflight=4` | `100`, c=`4` | `100 / 0` | `29.81` | `21.05` | `124.99 ms` | `284.68 ms` | `18182 MiB` | Fits, but does not improve whole-run throughput over split workers. |
| `asr-api` ONNX + TensorRT | `1` worker x `1` ONNX session | `100`, c=`1` | `100 / 0` | `16.58-16.77` | `13.00` | `53-60 ms` | `157-161 ms` | `~5100 MiB` | Hot-cache 35s TensorRT profile, no server errors. |
| `asr-api` ONNX + CUDA EP | `1` worker x `1` ONNX session, TensorRT disabled | `100`, c=`1` | `100 / 0` | `11.33` | `8.81` | `52.23 ms` | `237.84 ms` | `10514 MiB` | `ASR_COHERE_TRT_COMPONENTS=none`; clean CUDA-only baseline. |
| `asr-api` ONNX + CUDA EP | earlier CUDA baseline, `max_inflight=2` | `200`, c=`2` | `200 / 0` | `-` | `17.82` | `53.12 ms` | `255.94 ms` | `~10500 MiB` | User-run baseline before TensorRT tuning; load output did not include stage-window RTFx. |
| `asr-api` ONNX + decoder-only TensorRT | TensorRT only on decoder components | `100`, c=`1` | `100 / 0` | `8.22` | `-` | `-` | `~332 ms` | `~9400 MiB` | Slower than CUDA-only and full TensorRT; not a useful target. |
| Second State `cohere_transcribe_rs` | libtorch CUDA implementation | `100`, c=`1` | `100 / 0` | `-` | `10.99` | `-` | `~247.8 ms` | `~8700 MiB` | Baseline from `https://github.com/second-state/cohere_transcribe_rs`. |

Capacity observations from the same host:

| Runtime | Topology | Result |
| --- | --- | --- |
| ONNX + TensorRT | `4` total sessions | Fits in all tested shapes (`1x4`, `2x2`, `4x1`), using about `18.2-18.7 GiB` steady-state VRAM. |
| ONNX + TensorRT | `5` total sessions | Not tested as a target; expected to be too tight on a `20 GiB` Ada card without reducing per-session memory. |
| ONNX + CUDA EP | `1` worker x `2` ONNX sessions | A single worker with two CUDA sessions consumed about `20012 MiB`; effectively full-card. |
| ONNX + CUDA EP | `2` workers x `2` ONNX sessions | Failed startup. One worker survived at about `20012 MiB`; the other failed initializing `decoder_cached_step` with a CUDA BFCArena allocation error for `67108864` bytes. |

Memory efficiency matters more than the single-session footprint. The Second
State libtorch CUDA baseline used less memory than plain ONNX CUDA EP for one
request stream (`~8.7 GiB` vs `~10.5 GiB`), but it did not beat full TensorRT
throughput. Full TensorRT used about `~5.1 GiB` for one hot 35s session and
`18.2-18.7 GiB` for four total sessions, which is the only tested path that
kept the `20 GiB` Ada card below capacity while increasing useful concurrency.
Plain ONNX CUDA EP filled the card at two sessions and failed at the four
session topology.

The current practical takeaway is that full TensorRT is both faster and more
memory efficient on Ada. It is what makes four total Cohere ONNX sessions fit
on the `20 GiB` RTX 4000 Ada host; CUDA EP alone does not fit the same topology.

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

Ingress serves the `upload-response` cache API for inspection and worker handoff:

- `GET /_upload_response/streams`
- `GET /_upload_response/streams/{stream_id}`
- `GET /_upload_response/streams/{stream_id}/request/last`
- `GET /_upload_response/streams/{stream_id}/request/slots/{slot_id}`
- `GET /_upload_response/streams/{stream_id}/stages/{stage}/last`
- `GET /_upload_response/streams/{stream_id}/stages/{stage}/slots/{slot_id}`
- `GET /_upload_response/streams/{stream_id}/response/last`
- `GET /_upload_response/streams/{stream_id}/response/slots/{slot_id}`
- `PUT /_upload_response/streams/{stream_id}/readers/{worker_id}`
- `DELETE /_upload_response/streams/{stream_id}/readers/{worker_id}`
- `PUT /_upload_response/streams/{stream_id}/stages/{stage}/claim/{worker_id}`
- `DELETE /_upload_response/streams/{stream_id}/stages/{stage}/claim/{worker_id}`
- `PUT /_upload_response/streams/{stream_id}/stages/{stage}/head`
- `PUT /_upload_response/streams/{stream_id}/stages/{stage}/body`
- `PUT /_upload_response/streams/{stream_id}/stages/{stage}/control`
- `PUT /_upload_response/streams/{stream_id}/stages/{stage}/end`
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

## Runtime Notes

Worker orchestration uses `gpu-worker::upload_response`, and the shared stream transport comes directly from the sibling `web-services/upload-response` crate. Keep new cache claim, heartbeat, and remote-discovery behavior in those shared crates unless the behavior is ASR-specific.

`upload-response` cache sizing matters because `ChunkCache` eagerly allocates its ring buffers. The baseline config uses `16` streams, `32KB` slots, and `1024` slots per stream for the request ring, the `decoded` stage lane, and the response ring.

Ingress now stores raw upload bytes in the request ring, so request-ring pressure depends on the input codec and bitrate. The decoder stage writes canonical PCM into the `decoded` stage lane. With the current normalized audio format (`mono`, `16 kHz`, `f32`), that stage stores about `62.5 KiB/s` of audio. At the baseline setting of `1024 * 32 KiB`, each stream has about `32 MiB` of decoded-stage capacity, which is roughly `8.7 minutes` of decoded PCM before that lane starts wrapping. If you need to tolerate longer uploads or slower worker drain, increase `UPLOAD_RESPONSE_SLOTS_PER_STREAM` first. For example, `2048` slots is about `17.5 minutes` per stream, and `4096` slots is about `35 minutes` per stream at the same normalized audio format.

See also:

- `../asr-onnx/python/requirements-export-cu128-torch210.lock.txt` for the exact ONNX export environment
- `../asr-torch/python/requirements-trace-cu128-torch27.lock.txt` for the exact featurizer trace environment
