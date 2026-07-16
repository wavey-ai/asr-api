# asr-api

[`asr-api`](https://github.com/wavey-ai/asr-api) provides a
Deepgram-compatible `/v1/listen` API for ASR backends including Cohere
Transcribe and Parakeet/TDT. It supports buffered HTTP uploads, streaming HTTP
request bodies, WebSocket audio, and WebTransport over HTTP/3. It normalizes
input audio to mono `16 kHz` PCM, runs inference through the configured backend,
and returns Deepgram-compatible JSON.

## Useful For

- Deepgram-compatible ASR where clients send buffered HTTP uploads, streaming
  HTTP request bodies, WebSocket audio, or WebTransport over HTTP/3 to
  `/v1/listen`.
- GPU service experiments where ingress, decode, model execution, and response
  collection need to be measured as separate stages.
- Production-style benchmarking of Cohere Transcribe on Ada-class NVIDIA GPUs
  with ONNX Runtime, CUDA EP, TensorRT EP, and TensorRT engine caches.
- Apple Silicon development with a native MLX backend while keeping ONNX as the
  default serving path.
- Media research pipelines that resolve/download audio elsewhere, chunk it, and
  use ASR output as an intermediate artifact for structured notes, product
  analysis, search indexes, or QA.

## Architecture

The runtime has three roles:

1. `ingress`: accepts `/v1/listen` through the enabled `web-service`
   transports, including buffered HTTP uploads, streaming HTTP request bodies,
   WebSocket audio, and WebTransport over HTTP/3. It writes request bytes and
   metadata into
   [`upload-response`](https://github.com/wavey-ai/web-services/tree/main/upload-response)
   and waits for a worker response.
2. `decoder`: CPU processing stage. It discovers ingress origins, claims raw
   request streams, decodes/resamples/downmixes them to mono `16 kHz` `f32`,
   and writes canonical PCM to the `decoded` stage lane.
3. `worker`: model stage. It discovers ingress origins, claims decoded streams,
   runs Cohere inference through ONNX Runtime or MLX, and writes the final
   Deepgram-compatible response back to the owning ingress process.

The split keeps model libraries out of ingress, keeps audio codec work off the
GPU worker, and makes worker throughput visible at the cache/stage boundary.
The handoff is
[`upload-response`](https://github.com/wavey-ai/web-services/tree/main/upload-response);
there is no Redis sidecar or external queue.

See [ASR Capability Inventory](docs/asr-capability-inventory.md) for the
current Cohere and Parakeet backend capability matrix.

## Build

Default service build, Cohere ONNX plus audio decode:

```bash
cargo build --no-default-features --features cohere-backend,audio-decoder
```

Apple Silicon MLX build:

```bash
cd apple && swift build -c release && cd ..

MACOSX_DEPLOYMENT_TARGET=14.0 \
  cargo build --release --no-default-features --features cohere-mlx,audio-decoder \
  --bin cohere-debug
```

Parakeet ONNX/TDT build:

```bash
cargo build --no-default-features --features parakeet-backend,audio-decoder
```

Backends can be compiled together; Cohere ONNX remains the default runtime
unless `ASR_MODEL_PROVIDER` or `ASR_COHERE_BACKEND` selects another path.

## Model Artifacts

`ASR_MODEL_DIR` is required only for `worker`.

Cohere ONNX expects:

- `encoder.onnx`
- `encoder.onnx.data`
- `decoder_prefill.onnx`
- `decoder_prefill.onnx.data`
- `decoder_cached_step.onnx`
- `decoder_cached_step.onnx.data`
- `tokenizer.json`
- `tokenizer.model`
- `config.json`
- `generation_config.json`
- `preprocessor_config.json`

Cohere MLX expects a local copy of
[`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
from Hugging Face. Use a Hugging Face login that has access to Cohere's gated
model, then point `ASR_MODEL_DIR` at that directory.

The local directory should contain:

- `model.safetensors`
- `config.json`
- `preprocessor_config.json`
- `tokenizer.model`
- `vocab.json`

The `cohere-mlx` Rust backend starts the Swift runtime under `apple/` as a
persistent child process. The model is loaded once per worker and subsequent
audio windows use a newline-delimited request/response protocol over standard
input and output. Build that package with `swift build -c release` or set
`ASR_MLX_TRANSCRIBE_BIN`. The Swift package contains the Cohere encoder/decoder
MLX graph used by the local Apple Silicon runtime.

Parakeet ONNX/TDT expects:

- `encoder.onnx`
- `decoder.onnx`
- `joint.enc.onnx`
- `joint.pred.onnx`
- `joint.joint_net.onnx`
- `tokens.txt`

The ONNX and MLX artifacts can live in the same model directory when a host
needs both paths. Download and prepare the MLX directory with the setup script:

```bash
scripts/setup-cohere-mlx-model.sh --login
```

The script downloads
[`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026),
writes `vocab.json` from the downloaded `tokenizer.model`, and verifies the
files the MLX runtime loads. If you have already logged in to Hugging Face or
set `HF_TOKEN`, omit `--login`.

Model payloads are not checked into Git. For hosts that still use the Wavey
bucket mirror, seed or update the Cohere bundle with:

```bash
AWS_ACCESS_KEY_ID=... \
AWS_SECRET_ACCESS_KEY=... \
scripts/sync-model-from-bucket.sh \
  --model cohere-transcribe-03-2026 \
  --dest /var/lib/asr-api/models/cohere-transcribe-03-2026
```

The sync helper records remote object metadata in the destination and skips the
download when the local bundle is current.

## Runtime Configuration

Core service:

- `ASR_API_ROLE`: `ingress`, `decoder`, or `worker`
- `PORT`: TLS port, default `8443`
- `ENABLE_H3`: enable HTTP/3 in addition to HTTP/2
- `TLS_CERT_PATH` / `TLS_KEY_PATH`: optional PEM paths; if omitted the
  workspace default local TLS material is used
- `ASR_MODEL_DIR`: model directory, required for `worker`
- `ASR_DEVICE_IDS`: comma-separated GPU ids, default `0`
- `ASR_ONNX_SESSIONS`: ONNX sessions per device, default `1`
- `ASR_COHERE_MAX_NEW_TOKENS`: generation cap, default `384`
- `ASR_WORKER_COUNT`: local-orchestrator worker process count, default `1`
- `RUST_LOG`: tracing filter, default `info`
- `ASR_LOG_FORMAT`: `json`, `pretty`, or `compact`, default `json`

Audio/chunking:

- `CHUNK_SECONDS`: transcription window length, default `30`
- `OVERLAP_SECONDS`: overlap between adjacent windows, default `2`
- `FINAL_MIN_SECONDS`: minimum residual tail to keep, default `0.5`
- `UTT_SPLIT_SECONDS`: pause threshold used when `utterances=true`, default
  `0.8`

Backend selection and ONNX Runtime:

- `ASR_MODEL_PROVIDER`: `auto`, `cohere`, or `parakeet`; default `auto`
- `ASR_COHERE_BACKEND`: `onnx` or `mlx`, default `onnx`
- `ASR_ONNX_RUNTIME_LIB` / `ORT_DYLIB_PATH`: explicit ONNX Runtime dynamic
  library path
- `ASR_COHERE_FORCE_CPU`: force ONNX CPU EP for compare/debug runs
- `ASR_COHERE_TIMINGS`: emit per-window Cohere timing lines to stderr
- `ASR_COHERE_INTRA_THREADS`: ONNX intra-op threads. On macOS, unset lets ONNX
  Runtime choose its default pool; on other platforms unset preserves the
  previous single-thread behavior. Set `0` to skip explicit thread setting.
- `ASR_COHERE_PARALLEL_EXECUTION`: enable ONNX Runtime graph-level parallel
  execution
- `ASR_COHERE_INTER_THREADS`: ONNX inter-op threads when graph-level parallel
  execution is enabled

Parakeet:

- `ASR_ONNX_FORCE_CPU`: force the Parakeet ONNX/TDT path to CPU EP
- `ASR_PARAKEET_TIMINGS`: emit per-window Parakeet timing lines to stderr
- `ASR_PARAKEET_N_MELS`: default comes from `asr-onnx` config, currently `128`
- `ASR_PARAKEET_N_FFT`: default `512`
- `ASR_PARAKEET_WIN_LENGTH`: default `400`
- `ASR_PARAKEET_HOP_LENGTH`: default `160`
- `ASR_PARAKEET_PAD_TO`: default `0`; set `16` to mimic NeMo padding

Cohere word timestamps default to a generated-token frequency estimate. They
are monotonic and suitable for Deepgram-compatible `words` output, but they are
not model-derived CTC/TDT alignments.

Cohere ONNX can instead attach a Parakeet CTC ONNX forced-alignment
side-channel:

- `ASR_COHERE_TIMESTAMP_BACKEND`: `token-frequency` or `parakeet-ctc`
- `ASR_CTC_ALIGN_MODEL_DIR`: Parakeet CTC ONNX model directory
- `ASR_CTC_ALIGN_ONNX_FILE`: default `onnx/model.onnx`
- `ASR_CTC_ALIGN_EXECUTION_PROVIDER`: `auto`, `tensorrt`, `cuda`, or `cpu`
- `ASR_CTC_ALIGN_FORCE_CPU`: force CPU for local/macOS validation
- `ASR_CTC_ALIGN_TRT_CACHE_DIR`: TensorRT engine/timing cache directory
- `ASR_CTC_ALIGN_TRT_MIN_DURATION_S`, `ASR_CTC_ALIGN_TRT_OPT_DURATION_S`,
  `ASR_CTC_ALIGN_TRT_MAX_DURATION_S`: TensorRT dynamic-shape profile window
- `ASR_CTC_ALIGN_TRT_WORKSPACE_BYTES`: default `8 GiB`
- `ASR_CTC_ALIGN_TRT_FP16`: default disabled; enable only after validating
  finite logits for the specific CTC ONNX graph/runtime
- `ASR_CTC_ALIGN_TIMINGS`: emit CTC aligner timing/debug lines to stderr

The local macOS validation path uses ONNX Runtime CPU. TensorRT is not a macOS
path; build and benchmark TensorRT engines on a supported NVIDIA/CUDA host. The
NVIDIA TensorRT support matrix lists Linux, Windows, SBSA, and JetPack targets,
and no macOS target:
https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html

The local CPU validation path used `onnx/model_int8.onnx` from
`onnx-community/parakeet-ctc-0.6b-ONNX`. That int8/QDQ graph is not the server
performance target: use the default full-float `onnx/model.onnx` for CUDA or
TensorRT. `onnx/model_q4f16.onnx` and `onnx/model_fp16.onnx` produced all-NaN
logits in validation, and TensorRT FP16 builder mode also produced all-NaN
logits for this graph/runtime.

Validate Cohere timestamps against Parakeet/TDT:

```bash
ASR_ONNX_RUNTIME_LIB=/opt/homebrew/lib/libonnxruntime.dylib \
ASR_COHERE_FORCE_CPU=true \
ASR_ONNX_FORCE_CPU=true \
ASR_COHERE_TIMESTAMP_BACKEND=parakeet-ctc \
ASR_CTC_ALIGN_MODEL_DIR=models/parakeet-ctc-0.6b-onnx \
ASR_CTC_ALIGN_ONNX_FILE=onnx/model_int8.onnx \
ASR_CTC_ALIGN_FORCE_CPU=true \
cargo run --no-default-features \
  --features cohere-backend,parakeet-backend,audio-decoder \
  --bin timestamp-validate -- \
  --cohere-model-dir models/cohere-transcribe-03-2026 \
  --parakeet-model-dir models/parakeet-tdt-0.6b-v3 \
  --audio-path ../whisper.cpp/samples/jfk.wav \
  --device-ids ''
```

CoreML / Apple ONNX:

- `ASR_COHERE_COREML`: request CoreML EP
- `ASR_COHERE_EXECUTION_PROVIDER`: `coreml`, `metal`, or `apple` also requests
  CoreML EP
- `ASR_COHERE_COREML_COMPUTE_UNITS`: `cpu-and-gpu`, `all`,
  `cpu-and-neural-engine`, or `cpu-only`
- `ASR_COHERE_COREML_CACHE_DIR`: CoreML model cache path
- `ASR_COHERE_COREML_LOW_PRECISION_ACCUMULATION_ON_GPU`: enable CoreML low
  precision accumulation

TensorRT:

- `ASR_COHERE_TRT_COMPONENTS`: comma-separated `encoder`, `decoder_prefill`,
  `decoder_cached_step`, `all`, or `none`
- `ASR_COHERE_TRT_CACHE_DIR`: TensorRT engine cache path, default
  `$ASR_MODEL_DIR/.trt_cache`
- `ASR_COHERE_TRT_PROFILE_MIN_S`
- `ASR_COHERE_TRT_PROFILE_OPT_S`
- `ASR_COHERE_TRT_PROFILE_MAX_S` or `ASR_COHERE_TRT_PROFILE_SECONDS`
- `ASR_COHERE_TRT_PROFILE_MIN_FRAMES`
- `ASR_COHERE_TRT_PROFILE_OPT_FRAMES`
- `ASR_COHERE_TRT_PROFILE_MAX_FRAMES`
- `ASR_COHERE_TRT_WORKSPACE_BYTES`: default `4 GiB`
- `ASR_COHERE_TRT_BUILDER_OPT_LEVEL`: default `5`
- `ASR_COHERE_TRT_FP16`: default enabled
- `ASR_COHERE_TRT_DETAILED_BUILD_LOG`: verbose TensorRT build logging

`upload-response`:

- `UPLOAD_RESPONSE_NUM_STREAMS`: in-memory stream slots, default `16`
- `UPLOAD_RESPONSE_SLOT_SIZE_KB`: per-slot cache size, default `32`
- `UPLOAD_RESPONSE_SLOTS_PER_STREAM`: slots per stream, default `1024`
- `UPLOAD_RESPONSE_TIMEOUT_MS`: listen timeout, default `30000`
- `UPLOAD_RESPONSE_WATCH_POLL_MS`: ingress response watcher poll interval,
  default `1`
- `UPLOAD_RESPONSE_WORKER_POLL_MS`: worker poll interval, default `2`
- `UPLOAD_RESPONSE_MAX_INFLIGHT`: max simultaneously claimed streams per
  worker process, default `2`
- `UPLOAD_RESPONSE_WORKER_ID`: worker identity for cache claims
- `UPLOAD_RESPONSE_WORKER_ID_PREFIX`: local-orchestrator worker id prefix
- `UPLOAD_RESPONSE_INGRESS_URLS`: comma-separated ingress origins for worker
  mode
- `UPLOAD_RESPONSE_DISCOVERY_DNS`: optional `host:port` to resolve into ingress
  origins
- `UPLOAD_RESPONSE_DISCOVERY_INTERVAL_MS`: discovery refresh interval, default
  `2000`
- `UPLOAD_RESPONSE_INSECURE_TLS`: allow internal/self-signed TLS for worker
  mode

## Local Runs

Build a release debug binary for local Cohere benchmarking:

```bash
cargo build --release --no-default-features \
  --features cohere-backend,audio-decoder \
  --bin cohere-debug
```

CPU ONNX compare run:

```bash
ASR_ONNX_RUNTIME_LIB=/opt/homebrew/lib/libonnxruntime.dylib \
ASR_COHERE_FORCE_CPU=true \
ASR_COHERE_TIMINGS=true \
  target/release/cohere-debug \
  --model-dir models/cohere-transcribe-03-2026 \
  --audio-path ../whisper.cpp/samples/jfk.wav \
  --device-ids '' \
  --max-new-tokens 64 \
  --warmup 1 \
  --repeat 5
```

Apple ONNX/CoreML path:

```bash
ASR_COHERE_BACKEND=onnx \
ASR_COHERE_EXECUTION_PROVIDER=metal \
ASR_COHERE_COREML_COMPUTE_UNITS=cpu-and-gpu \
ASR_COHERE_COREML_CACHE_DIR=models/cohere-transcribe-03-2026/.coreml-cache-static \
  target/release/cohere-debug \
  --model-dir models/cohere-transcribe-03-2026 \
  --audio-path ../whisper.cpp/samples/jfk.wav \
  --device-ids '' \
  --max-new-tokens 64
```

Apple MLX runtime bundle check:

```bash
cd apple
swift build -c release
.build/release/asr-mlx-transcribe \
  --check \
  --model-dir ../models/cohere-transcribe-03-2026
```

Minimal Apple MLX Cohere ASR run:

```bash
cd apple && swift build -c release && cd ..

MACOSX_DEPLOYMENT_TARGET=14.0 \
cargo build --release --no-default-features \
  --features cohere-mlx,audio-decoder \
  --bin cohere-debug

ASR_COHERE_BACKEND=mlx \
ASR_COHERE_TIMINGS=true \
  target/release/cohere-debug \
  --model-provider cohere \
  --model-dir models/cohere-transcribe-03-2026 \
  --audio-path ../whisper.cpp/samples/jfk.wav \
  --device-ids '' \
  --max-new-tokens 64 \
  --warmup 1 \
  --repeat 3
```

Local three-role stack:

```bash
cargo build --release --no-default-features \
  --features cohere-backend,audio-decoder \
  --bin asr-api --bin local-orchestrator

ASR_COHERE_BACKEND=onnx \
ASR_COHERE_EXECUTION_PROVIDER=metal \
ASR_COHERE_COREML_COMPUTE_UNITS=cpu-and-gpu \
ASR_COHERE_COREML_CACHE_DIR=models/cohere-transcribe-03-2026/.coreml-cache-static \
  target/release/local-orchestrator \
  --model-provider cohere \
  --model-dir models/cohere-transcribe-03-2026 \
  --device-ids ''
```

For MLX on a local 16 GB Apple Silicon host, use one worker. Additional MLX
workers are separate processes with separate model copies, and the measured
behavior was worse due to Metal/unified-memory contention.

## Request Shapes

Buffered upload:

```bash
curl --http2 -k \
  -H 'Content-Type: audio/wav' \
  --data-binary @sample.wav \
  'https://localhost:8443/v1/listen?utterances=true&paragraphs=true&timestamps=true'
```

Buffered responses use the Deepgram JSON shape:

- `metadata`
- `results.channels[0].alternatives[0].transcript`
- `results.channels[0].alternatives[0].words`
- optional `results.utterances`
- optional `results.channels[0].alternatives[0].paragraphs`

Streaming is also supported on `/v1/listen`:

- request-body streaming returns newline-delimited JSON;
- WebSocket clients send binary audio frames and may send JSON control messages
  with `type` set to `KeepAlive`, `Finalize`, or `CloseStream`;
- `interim_results=true` enables interim `Results` events.

Correlation IDs use Wavey's snowflake generator and are propagated internally
in `x-wavey-request-id`. If the client supplies a numeric `x-request-id`,
`asr-api` reuses it; otherwise ingress mints one.

## TensorRT Cache Workflow

TensorRT engine caches should be built and validated on a compatible GPU host.
Cache ids should encode the compatibility dimensions: GPU family, CUDA,
TensorRT, ONNX Runtime, precision, components, and profile window.

Pull the Ada 35s Cohere cache:

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

Example Ada worker environment:

```bash
ASR_COHERE_BACKEND=onnx \
ASR_COHERE_TRT_COMPONENTS=all \
ASR_COHERE_TRT_CACHE_DIR=/var/lib/asr-api/models/cohere-transcribe-03-2026/.trt_cache_all_35s_frames \
ASR_COHERE_TRT_PROFILE_MAX_S=35 \
ASR_COHERE_TRT_FP16=true
```

## Benchmark Interpretation

`Stage RTFx` is the measured load window after warmup. `Whole RTFx` includes
prewarm, warmup, client/report overhead, and server orchestration effects.
`Response mean` is the benchmark client's part-response mean for `asr-api`
rows.

These numbers are useful because they answer different operational questions:

- single-session RTFx tells you whether a backend is viable for one request
  stream;
- multi-session RTFx tells you whether a GPU can turn memory into useful
  throughput;
- response mean tells you whether the service topology is hiding or exposing
  model latency;
- VRAM tells you whether the topology is a production candidate or just a
  benchmark artifact.

## macOS MLX And ONNX Findings

Hardware reported `8` physical and logical CPUs split across two macOS perf
levels of `4` cores each.

`../whisper.cpp/samples/jfk.wav`, 11.0s audio, Cohere ONNX, Homebrew ONNX
Runtime `1.25.1`:

| Mode | Thread config | Mean RTFx | Notes |
| --- | --- | ---: | --- |
| CPU | old single intra-op thread | 1.30x | 5-repeat run before thread changes |
| CoreML/Metal | old single intra-op thread | 1.04x | `cpu-and-gpu` compute units |
| CPU | `ASR_COHERE_INTRA_THREADS=2` | 1.60x | single sample |
| CPU | `ASR_COHERE_INTRA_THREADS=4` | 2.21x | single sample |
| CPU | `ASR_COHERE_INTRA_THREADS=6` | 2.03x | single sample |
| CPU | `ASR_COHERE_INTRA_THREADS=8` | 1.60x | single sample |
| CPU | ORT default thread pool | 3.20x | warmup 1, repeat 5 |
| CoreML/Metal | ORT default, `cpu-and-gpu` | 2.17x | single sample |
| CoreML/Metal | ORT default, `all` | 2.11x | single sample |

For the warm CPU run, the mean decode time was `3432.75ms` for 11.0s audio.
Stage timings put most time in `encoder_run_ms`.

`/tmp/asr-cohere-bench-30.wav`, 30.0s audio, CPU ORT default thread pool,
warmup 1, repeat 3:

- `mean_decode_ms=8196.49`
- `mean_rtfx=3.66`
- representative `encoder_run_ms` was `6414ms` to `7590ms`

The ONNX/CoreML path did not meet a `10x` real-time local target. Removing the
one-thread cap helped substantially, but this export remained encoder-bound.

`../whisper.cpp/samples/jfk.wav`, 11.0s audio, Cohere MLX, release build,
`ASR_COHERE_BACKEND=mlx`, `ASR_COHERE_TIMINGS=true`, `--max-new-tokens 128`:

| Runtime | Shape | Timing | RTFx | Notes |
| --- | --- | ---: | ---: | --- |
| `asr-api` MLX | init | `14513.94ms` | - | model load and backend setup |
| `asr-api` MLX | warmup 1 | `9409.93ms` | `1.17x` | includes cold Metal compilation |
| `asr-api` MLX | repeat 1 | `2191.12ms` | `5.02x` | still warming |
| `asr-api` MLX | repeats 2-5 | `1006-1021ms` | `10.77-10.93x` | steady warm path |

`/tmp/asr-cohere-bench-30.wav`, 30.0s audio, Cohere MLX, release build,
`--max-new-tokens 128`:

| Runtime | Shape | Timing | RTFx | Notes |
| --- | --- | ---: | ---: | --- |
| `asr-api` MLX | init | `8260.89ms` | - | model load and backend setup |
| `asr-api` MLX | warmup 1 | `6535.53ms` | `4.59x` | cold request |
| `asr-api` MLX | repeats 1-3 | `1986-1989ms` | `15.08-15.10x` | steady warm path |

Local service-stack MLX results:

| Runtime | Topology | Measured load | Stage RTFx | Whole RTFx | Response mean | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Mac MLX `asr-api` | `1` worker | `2` x 30s WAV, c=`1` | `14.57` | `3.99` | `4.98s` | Good local-dev path; use one worker. |
| Mac MLX `asr-api` | `2` workers | `2` x 30s WAV, c=`2` | `1.66` | `0.94` | `40.02s` | Two model copies fought for the same MLX/Metal resources. |

The slow MLX numbers are cold-start and first-request Metal compilation
effects. For this repo, MLX is the Apple Silicon development backend; Ada
TensorRT is the throughput path.

## Cohere Ada Benchmarks

These are point-in-time Cohere Transcribe measurements from the Linode NVIDIA
RTX 4000 Ada Generation host (`20475 MiB` VRAM) on `2026-05-14`. The
`asr-api` rows used release binaries, the split `ingress` / `decoder` /
`worker` upload-response path, the Cohere ONNX bundle synced from the bucket,
and Harvard `*.s16le` PCM files submitted over HTTP/2 after warmup.

| Runtime | Topology | Measured load | OK / fail | Stage RTFx | Whole RTFx | Mean TTFB | Response mean | GPU VRAM | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `asr-api` ONNX + TensorRT | `4` workers x `1` ONNX session, `max_inflight=1` each | `100`, c=`4` | `100 / 0` | `32.90` | `21.11` | `146.12 ms` | `298.80 ms` | `18681 MiB` | Best measured stage RTFx, with `27` internal stale response-claim warnings while clients still saw `100%` OK. |
| `asr-api` ONNX + TensorRT | `2` workers x `2` ONNX sessions, `max_inflight=2` each | `100`, c=`4` | `100 / 0` | `32.80` | `21.01` | `142.15 ms` | `314.26 ms` | `18347 MiB` | Similar throughput to `4x1`, slightly lower VRAM, `7` stale response-claim warnings. |
| `asr-api` ONNX + TensorRT | `1` worker x `4` ONNX sessions, `max_inflight=4` | `100`, c=`4` | `100 / 0` | `29.81` | `21.05` | `124.99 ms` | `284.68 ms` | `18182 MiB` | Fits, but did not improve whole-run throughput over split workers. |
| `asr-api` ONNX + TensorRT | `1` worker x `1` ONNX session | `100`, c=`1` | `100 / 0` | `16.58-16.77` | `13.00` | `53-60 ms` | `157-161 ms` | `~5100 MiB` | Hot-cache 35s TensorRT profile, no server errors. |
| `asr-api` ONNX + CUDA EP | `1` worker x `1` ONNX session, TensorRT disabled | `100`, c=`1` | `100 / 0` | `11.33` | `8.81` | `52.23 ms` | `237.84 ms` | `10514 MiB` | `ASR_COHERE_TRT_COMPONENTS=none`; clean CUDA-only baseline. |
| `asr-api` ONNX + CUDA EP | earlier CUDA baseline, `max_inflight=2` | `200`, c=`2` | `200 / 0` | `-` | `17.82` | `53.12 ms` | `255.94 ms` | `~10500 MiB` | User-run baseline before TensorRT tuning; load output did not include stage-window RTFx. |
| `asr-api` ONNX + decoder-only TensorRT | TensorRT only on decoder components | `100`, c=`1` | `100 / 0` | `8.22` | `-` | `-` | `~332 ms` | `~9400 MiB` | Slower than CUDA-only and full TensorRT; not a useful target. |

Capacity observations from the same host:

| Runtime | Topology | Result |
| --- | --- | --- |
| ONNX + TensorRT | `4` total sessions | Fits in all tested shapes (`1x4`, `2x2`, `4x1`), using about `18.2-18.7 GiB` steady-state VRAM. |
| ONNX + TensorRT | `5` total sessions | Not tested as a target; expected to be too tight on a `20 GiB` Ada card without reducing per-session memory. |
| ONNX + CUDA EP | `1` worker x `2` ONNX sessions | A single worker with two CUDA sessions consumed about `20012 MiB`; effectively full-card. |
| ONNX + CUDA EP | `2` workers x `2` ONNX sessions | Failed startup. One worker survived at about `20012 MiB`; the other failed initializing `decoder_cached_step` with a CUDA BFCArena allocation error for `67108864` bytes. |

Memory efficiency matters more than the single-session footprint. Full
TensorRT used about `~5.1 GiB` for one hot 35s session and `18.2-18.7 GiB` for
four total sessions, which is the tested path that kept the `20 GiB` Ada card
below capacity while increasing useful concurrency. Plain ONNX CUDA EP filled
the card at two sessions and failed at the four-session topology.

The practical deployment conclusion from these measurements is that full
TensorRT is both faster and more memory efficient on Ada. It is what makes four
total Cohere ONNX sessions fit on the `20 GiB` RTX 4000 Ada host.

## Example Workload

The [Bitneedle scratch tutorial sweep](examples/bitneedle-scratch-tutorial-sweep/README.md)
is the worked example in this repo. It resolves public DJ scratching tutorials
through `av-ingest`, segments the audio to mono `16 kHz` WAV chunks, submits
those chunks to `/v1/listen`, and stores non-verbatim research artifacts:
technique tags, timestamped summaries, aggregate term counts, and UI notes.

Measured portion from the `2026-05-17` run:

- `78` uploaded chunks across `11` videos
- `4,378.0s` aggregate media duration
- `294.4s` aggregate summed wall-clock time
- `14.87x` aggregate observed RTFx

## Internal Cache API

Ingress serves the `upload-response` cache API for inspection and worker
handoff:

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

`upload-response` cache sizing matters because `ChunkCache` eagerly allocates
ring buffers. The baseline config uses `16` streams, `32 KiB` slots, and `1024`
slots per stream for the request ring, decoded stage lane, and response ring.

Ingress stores raw upload bytes in the request ring, so request pressure depends
on the input codec and bitrate. The decoder writes canonical PCM into the
`decoded` stage lane. With mono `16 kHz` `f32`, that lane stores about
`62.5 KiB/s`. At `1024 * 32 KiB`, each stream has about `32 MiB`, or roughly
`8.7 minutes`, of decoded PCM capacity before wrapping. Use `2048` slots for
about `17.5 minutes`, or `4096` for about `35 minutes`.

## Verification

The current cleanup was verified with:

```bash
cargo check --no-default-features \
  --features cohere-backend,audio-decoder \
  --bin asr-api --bin local-orchestrator --bin cohere-debug

MACOSX_DEPLOYMENT_TARGET=14.0 \
  cargo check --no-default-features \
  --features cohere-mlx,audio-decoder \
  --bin asr-api --bin local-orchestrator

cargo check --no-default-features \
  --features parakeet-backend,audio-decoder \
  --bin asr-api --bin local-orchestrator

(cd apple && swift build -c debug)

cargo test --no-default-features --features cohere-backend,audio-decoder --lib
```
