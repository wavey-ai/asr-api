# asr-api

`asr-api` provides a Deepgram-compatible `/v1/listen` speech recognition API.
It supports Cohere Transcribe and NVIDIA Parakeet backends.

The service accepts buffered and streaming audio. It returns
Deepgram-compatible transcripts, words, utterances, and paragraphs.

## Supported Transports

The service supports these `/v1/listen` transports:

- buffered HTTP uploads over HTTP/2;
- streaming HTTP request bodies;
- WebSocket audio;
- WebTransport over HTTP/3.

The service converts input audio to mono `16 kHz` `f32` PCM before inference.

## Architecture

The service separates transport, audio decoding, and model inference.

```text
Client
  |
  |  /v1/listen
  v
Ingress -- raw request --> Decoder -- mono 16 kHz f32 --> Worker
  ^                                                       |
  |                                                       |
  +----------- Deepgram-compatible response -------------+
```

### Runtime Roles

| Role | Responsibility |
| --- | --- |
| `ingress` | Accept requests and store request streams. Return worker responses to clients. |
| `decoder` | Claim request streams. Decode, downmix, and resample audio. |
| `worker` | Claim decoded PCM. Run the selected ASR backend. Build the API response. |

The roles exchange data through
[`upload-response`](https://github.com/wavey-ai/web-services/tree/main/upload-response).
Each ingress process owns an in-memory stream cache.

Decoder and worker processes discover ingress origins through URLs or DNS.
They claim work from the cache and write results to stage lanes.

The service does not require Redis or an external message queue.

### Request Flow

1. Ingress stores the request headers and body.
2. A decoder claims the raw request stream.
3. The decoder converts the audio to canonical PCM.
4. The decoder writes PCM to the `decoded` stage lane.
5. A worker claims the decoded stream.
6. The worker divides PCM into overlapping windows.
7. The selected backend transcribes each window.
8. Rust combines stable words across window boundaries.
9. The worker writes the final response.
10. Ingress returns the response to the client.

The default window is `30` seconds. The default overlap is `2` seconds.
The worker does not commit words from the unstable overlap region.

### Runtime Ownership

Rust owns the service and model control flow.
Model tensor operations use ONNX Runtime or MLX.

| Layer | Owner |
| --- | --- |
| HTTP, WebSocket, and WebTransport | Rust `web-service` integration |
| Request and response handoff | Rust `upload-response` integration |
| Codec decode and resampling | Rust SoundKit integration |
| Windowing and overlap removal | Rust |
| Cohere and Parakeet mel features | Rust |
| ONNX session scheduling | Rust |
| Token generation loops | Rust, except the Cohere MLX loop |
| Word timestamp mapping | Rust |
| Parakeet and Cohere ONNX graphs | ONNX Runtime execution providers |
| Cohere Apple GPU graph | Swift and MLX, controlled by Rust |

Python is required for model export and validation only.
Python is not part of the serving process.

## Backend Selection

`ASR_MODEL_PROVIDER` selects the model family.
`ASR_COHERE_BACKEND` selects the Cohere runtime.

| Provider | Runtime | Cargo feature | Execution engine |
| --- | --- | --- | --- |
| Cohere | ONNX | `cohere-backend` | CPU, CUDA, TensorRT, or CoreML |
| Cohere | MLX | `cohere-mlx` | Apple MLX and Metal |
| Parakeet TDT | ONNX | `parakeet-backend` | CPU, CUDA, or TensorRT |

`ASR_MODEL_PROVIDER=auto` selects Cohere.
Cohere ONNX is the default when the build contains `cohere-backend`.

## Parakeet TDT Runtime

The serving model is NVIDIA Parakeet TDT 0.6B v3.
The export process loads the NeMo model and writes five ONNX graphs.

| File | Function |
| --- | --- |
| `encoder.onnx` | Convert a complete mel window into acoustic frames. |
| `decoder.onnx` | Run the recurrent prediction network. Update its hidden states. |
| `joint.enc.onnx` | Project encoder output into the joint-network space. |
| `joint.pred.onnx` | Project decoder output into the joint-network space. |
| `joint.joint_net.onnx` | Produce token and duration logits. |

The exporter also writes `tokens.txt`, `vocab.txt`, and `export.json`.
See [`../asr-onnx/export/export_parakeet_tdt.py`](../asr-onnx/export/export_parakeet_tdt.py).

### Parakeet Frontend

Rust computes the NeMo-compatible frontend with `mel-spec`.
The frontend includes these operations:

- pre-emphasis;
- centered FFT frames;
- Slaney mel filters;
- log energy;
- per-feature normalization;
- sparse filterbank projection;
- reusable FFT scratch storage.

The default shape uses `128` mel features.
The default FFT size is `512` samples.
The window is `400` samples, and the hop is `160` samples.

### Parakeet Decode

Rust runs the TDT decode policy around the ONNX graphs:

1. Run the encoder for the complete feature window.
2. Run the joint encoder projection once.
3. Initialize the recurrent decoder states.
4. Run the decoder for the current token.
5. Run the joint predictor projection.
6. Combine the encoder and predictor projections.
7. Select the token and duration with argmax.
8. Advance acoustic time by the predicted duration.
9. Repeat until the encoded window is complete.
10. Convert token emission positions to word timestamps.

`asr-onnx` creates one session pool for each device.
It can create multiple sessions for each device.
The pool distributes jobs across devices and sessions.

TensorRT selection is available for each split component.
The default `asr-onnx` TensorRT set is `encoder,joint_enc`.

See [`../asr-onnx/src/lib.rs`](../asr-onnx/src/lib.rs) for the decode loop.

## Parakeet CTC Runtime

Parakeet CTC is a separate ONNX path.
It can transcribe audio or align an existing Cohere transcript.

```text
Rust mel frontend
  -> Parakeet CTC ONNX
  -> frame token logits
  -> Rust greedy decode or forced alignment
  -> timed words
```

The Cohere side-channel uses forced alignment.
Rust tokenizes the Cohere transcript and aligns it to CTC frames.

The direct CTC path uses greedy token collapse.
It then converts token spans to word spans.

The server target is the full-float `onnx/model.onnx` artifact.
Use CUDA or validated TensorRT settings on NVIDIA hosts.

Do not enable CTC TensorRT FP16 without model-specific validation.
Tested FP16 paths produced non-finite logits on the current runtime.

See [`src/ctc_align.rs`](src/ctc_align.rs).

## Cohere ONNX Runtime

The serving model is Cohere Transcribe 03-2026.
The exporter separates the model into three runtime graphs.

| File | Function |
| --- | --- |
| `encoder.onnx` | Convert mel features into acoustic hidden states. |
| `decoder_prefill.onnx` | Process the control prompt. Create self-attention and cross-attention caches. |
| `decoder_cached_step.onnx` | Generate one token from the current caches. |

The exporter also writes tokenizer, generation, processor, and model metadata.
See [`../asr-onnx/export/export_cohere_transcribe.py`](../asr-onnx/export/export_cohere_transcribe.py).

### Cohere Frontend

Rust reads `preprocessor_config.json` and computes Cohere-compatible features.
The frontend uses a Hann window, pre-emphasis, Slaney filters, and normalization.

Rust uses the same frontend for Cohere ONNX and Cohere MLX.
This rule prevents frontend differences between the two runtimes.

### Cohere Decode

Rust controls the ONNX generation sequence:

1. Run `encoder.onnx` for the feature window.
2. Build the language and punctuation prompt.
3. Run `decoder_prefill.onnx` for the complete prompt.
4. Extract the first token and all decoder caches.
5. Run `decoder_cached_step.onnx` for one token.
6. Replace each self-attention cache with its new value.
7. Reuse the cross-attention caches.
8. Stop at EOS or the configured token limit.
9. Decode token identifiers with `tokenizer.json`.

Each ONNX component can use a different execution provider.
CUDA remains the fallback when TensorRT does not own a component.

See [`src/cohere.rs`](src/cohere.rs).

### Cohere Timestamps

Cohere does not expose a duration or CTC head in this serving path.
The default mode estimates word times from generated token positions.

Set `ASR_COHERE_TIMESTAMP_BACKEND=parakeet-ctc` for model-derived boundaries.
This mode runs the Parakeet CTC side-channel after Cohere decode.

The token-frequency mode is monotonic but approximate.
The Parakeet CTC mode gives more accurate word boundaries.

## Cohere MLX Runtime

The Apple path uses an owned Swift and MLX implementation.
It does not use `cohere-transcribe-rs` or a vendored `mlx-c` runtime.

Rust remains the service owner:

1. Rust computes the Cohere mel features.
2. Rust writes one contiguous little-endian `f32` feature tensor.
3. Rust sends a JSON request to the persistent Swift process.
4. Swift loads the feature tensor into MLX.
5. MLX runs the Conformer encoder.
6. MLX precomputes decoder cross-attention caches.
7. MLX runs cached autoregressive decoding.
8. Swift returns token identifiers and text.
9. Rust creates the final timestamps and API response.

The Rust worker starts one persistent Swift process.
The Swift process loads `model.safetensors` once.
Requests use newline-delimited JSON over standard input and output.

The default path uses BF16-oriented model execution.
Optional weight-only quantization uses MLX `quantizedMM`.

Use one MLX worker on a small Apple Silicon system.
Each additional worker loads another complete model copy.

See [`src/cohere_mlx.rs`](src/cohere_mlx.rs) and
[`apple/Sources/AsrMLXRuntime/CohereGraph.swift`](apple/Sources/AsrMLXRuntime/CohereGraph.swift).

Parakeet MLX is not implemented.

## Model Artifacts

Set `ASR_MODEL_DIR` only for the `worker` role.

### Cohere ONNX Bundle

The runtime requires these files:

- `encoder.onnx` and `encoder.onnx.data`;
- `decoder_prefill.onnx` and `decoder_prefill.onnx.data`;
- `decoder_cached_step.onnx` and `decoder_cached_step.onnx.data`;
- `tokenizer.json` and `tokenizer.model`;
- `config.json`;
- `generation_config.json`;
- `preprocessor_config.json`.

### Cohere MLX Bundle

The runtime requires these files:

- `model.safetensors`;
- `config.json`;
- `preprocessor_config.json`;
- `vocab.json`;
- `tokenizer_config.json`.

`tokenizer.model` is the source for generated `vocab.json` data.
Keep it when the host must rebuild the MLX bundle.

Prepare the MLX model directory with this command:

```bash
scripts/setup-cohere-mlx-model.sh --login
```

Omit `--login` when Hugging Face authentication is already available.

### Parakeet TDT Bundle

The runtime requires these files:

- `encoder.onnx`;
- `decoder.onnx`;
- `joint.enc.onnx`;
- `joint.pred.onnx`;
- `joint.joint_net.onnx`;
- `tokens.txt`.

The ONNX files can refer to external `.onnx.data` files.
Keep those data files beside their ONNX files.

### Parakeet CTC Bundle

The side-channel requires these files:

- `config.json`;
- `preprocessor_config.json`;
- `tokenizer.json`;
- the selected ONNX graph.

The default graph path is `onnx/model.onnx`.

### Bucket Synchronization

Model payloads are not stored in Git.
Use the model synchronization script on hosts that use the Wavey mirror:

```bash
scripts/sync-model-from-bucket.sh \
  --model cohere-transcribe-03-2026 \
  --dest /var/lib/asr-api/models/cohere-transcribe-03-2026
```

The script records remote metadata in the destination.
It skips the transfer when the local bundle is current.

## Build

### Cohere ONNX

```bash
cargo build --release --no-default-features \
  --features cohere-backend,audio-decoder \
  --bin asr-api --bin local-orchestrator --bin cohere-debug
```

### Parakeet TDT

```bash
cargo build --release --no-default-features \
  --features parakeet-backend,audio-decoder \
  --bin asr-api --bin local-orchestrator
```

### Cohere MLX

```bash
cd apple
swift build -c release
cd ..

MACOSX_DEPLOYMENT_TARGET=14.0 \
  cargo build --release --no-default-features \
  --features cohere-mlx,audio-decoder \
  --bin asr-api --bin local-orchestrator --bin cohere-debug
```

Backends can exist in the same build.
Use runtime variables to select one backend.

## Local Operation

### Three-Role Orchestrator

Use `local-orchestrator` to start ingress, decoder, and worker processes.

```bash
target/release/local-orchestrator \
  --model-provider cohere \
  --model-dir models/cohere-transcribe-03-2026 \
  --device-ids 0
```

For a CPU comparison, clear the device list and force CPU execution:

```bash
ASR_COHERE_FORCE_CPU=true \
ASR_ONNX_RUNTIME_LIB=/opt/homebrew/lib/libonnxruntime.dylib \
  target/release/local-orchestrator \
  --model-provider cohere \
  --model-dir models/cohere-transcribe-03-2026 \
  --device-ids ''
```

### Cohere MLX Check

Check the model bundle before you start the Rust service:

```bash
apple/.build/release/asr-mlx-transcribe \
  --check \
  --model-dir models/cohere-transcribe-03-2026
```

Run the local MLX backend:

```bash
ASR_COHERE_BACKEND=mlx \
  target/release/local-orchestrator \
  --model-provider cohere \
  --model-dir models/cohere-transcribe-03-2026 \
  --device-ids ''
```

### Separate Hosts

Set one role on each process with `ASR_API_ROLE`.

Decoder and worker roles require one discovery setting:

- `UPLOAD_RESPONSE_INGRESS_URLS` for an explicit origin list;
- `UPLOAD_RESPONSE_DISCOVERY_DNS` for DNS discovery.

Use unique `UPLOAD_RESPONSE_WORKER_ID` values for concurrent workers.

## API Use

### Buffered Upload

```bash
curl --http2 -k \
  -H 'Content-Type: audio/wav' \
  --data-binary @sample.wav \
  'https://localhost:8443/v1/listen?utterances=true&paragraphs=true&timestamps=true'
```

The response can contain these Deepgram-compatible fields:

- `metadata`;
- `results.channels[0].alternatives[0].transcript`;
- `results.channels[0].alternatives[0].words`;
- `results.utterances`;
- `results.channels[0].alternatives[0].paragraphs`.

### Streaming

Streaming request bodies return newline-delimited JSON events.

WebSocket clients send audio in binary frames.
They can send these JSON control messages:

- `KeepAlive`;
- `Finalize`;
- `CloseStream`.

Set `interim_results=true` to receive interim `Results` events.

### Request Identifiers

The service propagates request identifiers in `x-wavey-request-id`.
It reuses a numeric client `x-request-id` when one is present.

The service creates a Wavey snowflake identifier when the client omits one.

## Configuration

Run `asr-api --help` for command-line options.
The tables below show the principal environment variables.

### Service

| Variable | Default | Purpose |
| --- | --- | --- |
| `ASR_API_ROLE` | `ingress` | Select `ingress`, `decoder`, or `worker`. |
| `PORT` | `8443` | Set the TLS port. |
| `ENABLE_H3` | `false` | Enable HTTP/3. |
| `TLS_CERT_PATH` | local default | Set the TLS certificate path. |
| `TLS_KEY_PATH` | local default | Set the TLS key path. |
| `RUST_LOG` | `info` | Set the tracing filter. |
| `ASR_LOG_FORMAT` | `json` | Select `json`, `pretty`, or `compact`. |

Set both TLS path variables or set neither variable.

### Model and Windows

| Variable | Default | Purpose |
| --- | --- | --- |
| `ASR_MODEL_PROVIDER` | `auto` | Select `auto`, `cohere`, or `parakeet`. |
| `ASR_COHERE_BACKEND` | `onnx` | Select `onnx` or `mlx`. |
| `ASR_MODEL_DIR` | none | Set the worker model directory. |
| `ASR_DEVICE_IDS` | `0` | Set comma-separated GPU identifiers. |
| `ASR_ONNX_SESSIONS` | `1` | Set sessions for each device. |
| `ASR_COHERE_MAX_NEW_TOKENS` | `384` | Limit generated Cohere tokens. |
| `CHUNK_SECONDS` | `30` | Set the transcription window. |
| `OVERLAP_SECONDS` | `2` | Set adjacent window overlap. |
| `FINAL_MIN_SECONDS` | `0.5` | Set the minimum final tail. |
| `UTT_SPLIT_SECONDS` | `0.8` | Set the utterance pause threshold. |

### Ingress Cache and Discovery

| Variable | Default | Purpose |
| --- | --- | --- |
| `UPLOAD_RESPONSE_NUM_STREAMS` | `16` | Set concurrent stream slots. |
| `UPLOAD_RESPONSE_SLOT_SIZE_KB` | `32` | Set bytes for each ring slot. |
| `UPLOAD_RESPONSE_SLOTS_PER_STREAM` | `1024` | Set ring slots for each stream. |
| `UPLOAD_RESPONSE_TIMEOUT_MS` | `30000` | Set the response timeout. |
| `UPLOAD_RESPONSE_MAX_INFLIGHT` | `2` | Set claims for each worker process. |
| `UPLOAD_RESPONSE_WORKER_ID` | `asr-api-worker` | Set the worker identity. |
| `UPLOAD_RESPONSE_INGRESS_URLS` | none | Set explicit ingress origins. |
| `UPLOAD_RESPONSE_DISCOVERY_DNS` | none | Set the discovery DNS name and port. |
| `UPLOAD_RESPONSE_DISCOVERY_INTERVAL_MS` | `2000` | Set the discovery refresh interval. |
| `UPLOAD_RESPONSE_INSECURE_TLS` | `false` | Allow internal self-signed TLS. |

### Cohere ONNX

`ASR_COHERE_MAX_NEW_TOKENS` limits generated output for each ASR window.
Use the default value of `384` for production and representative benchmarks.
A smaller value can truncate dense speech before the window ends.

Use these variables to select an execution provider:

- `ASR_COHERE_FORCE_CPU`;
- `ASR_COHERE_COREML`;
- `ASR_COHERE_EXECUTION_PROVIDER`;
- `ASR_COHERE_TRT_COMPONENTS`.

CoreML settings include these variables:

- `ASR_COHERE_COREML_COMPUTE_UNITS`;
- `ASR_COHERE_COREML_CACHE_DIR`;
- `ASR_COHERE_COREML_LOW_PRECISION_ACCUMULATION_ON_GPU`.

TensorRT settings include these variables:

- `ASR_COHERE_TRT_CACHE_DIR`;
- `ASR_COHERE_TRT_PROFILE_MIN_S`;
- `ASR_COHERE_TRT_PROFILE_OPT_S`;
- `ASR_COHERE_TRT_PROFILE_MAX_S`;
- `ASR_COHERE_TRT_WORKSPACE_BYTES`;
- `ASR_COHERE_TRT_BUILDER_OPT_LEVEL`;
- `ASR_COHERE_TRT_FP16`.

`ASR_COHERE_TRT_COMPONENTS` accepts these values:

- `encoder`;
- `decoder_prefill`;
- `decoder_cached_step`;
- `all`;
- `none`.

### Parakeet TDT

The main Parakeet settings are:

- `ASR_ONNX_FORCE_CPU`;
- `ASR_ONNX_TRT_COMPONENTS`;
- `ASR_ONNX_TRT_CACHE_DIR`;
- `ASR_ONNX_PROFILE_MIN_S`;
- `ASR_ONNX_PROFILE_MAX_S`;
- `ASR_PARAKEET_N_MELS`;
- `ASR_PARAKEET_N_FFT`;
- `ASR_PARAKEET_WIN_LENGTH`;
- `ASR_PARAKEET_HOP_LENGTH`;
- `ASR_PARAKEET_PAD_TO`.

### Parakeet CTC Alignment

The main CTC settings are:

- `ASR_COHERE_TIMESTAMP_BACKEND`;
- `ASR_CTC_ALIGN_MODEL_DIR`;
- `ASR_CTC_ALIGN_ONNX_FILE`;
- `ASR_CTC_ALIGN_EXECUTION_PROVIDER`;
- `ASR_CTC_ALIGN_FORCE_CPU`;
- `ASR_CTC_ALIGN_TRT_CACHE_DIR`;
- `ASR_CTC_ALIGN_TRT_MIN_DURATION_S`;
- `ASR_CTC_ALIGN_TRT_OPT_DURATION_S`;
- `ASR_CTC_ALIGN_TRT_MAX_DURATION_S`;
- `ASR_CTC_ALIGN_TRT_WORKSPACE_BYTES`;
- `ASR_CTC_ALIGN_TRT_FP16`.

## TensorRT Cache Workflow

Build each TensorRT cache on a compatible NVIDIA host.
Include these values in each cache identifier:

- GPU family;
- CUDA version;
- TensorRT version;
- ONNX Runtime version;
- precision;
- selected components;
- profile window.

Pull a cache before worker startup:

```bash
scripts/sync-trt-cache.sh pull \
  --model cohere-transcribe-03-2026 \
  --cache-id rtx4000-ada-ort1.23.2-trt10-fp16-all-35s \
  --dir /var/lib/asr-api/models/cohere-transcribe-03-2026/.trt_cache_all_35s_frames
```

Publish a cache only from the host that built and validated it:

```bash
scripts/sync-trt-cache.sh push \
  --model cohere-transcribe-03-2026 \
  --cache-id rtx4000-ada-ort1.23.2-trt10-fp16-all-35s \
  --dir /var/lib/asr-api/models/cohere-transcribe-03-2026/.trt_cache_all_35s_frames
```

TensorRT does not run on macOS.
Use ONNX Runtime CPU, CoreML, or Cohere MLX on Apple systems.

## Cache Capacity

`upload-response` uses eager ring-buffer allocation.
Cache size changes affect memory use immediately.

The default configuration uses these values:

- `16` streams;
- `32 KiB` slots;
- `1024` slots for each stream lane.

Canonical mono `16 kHz` `f32` PCM uses approximately `62.5 KiB/s`.
One default `32 MiB` lane holds approximately `8.7` minutes of PCM.

Use `2048` slots for approximately `17.5` minutes.
Use `4096` slots for approximately `35` minutes.

Ingress stores encoded request bytes in the request lane.
The decoder stores canonical PCM in the `decoded` lane.

## Performance Direction

Point-in-time measurements support these deployment choices:

- Cohere TensorRT is the NVIDIA throughput path.
- Cohere MLX is the Apple Silicon path.
- Use one MLX worker on a shared Apple GPU.
- Use full-float CUDA for the current Parakeet CTC server artifact.
- Keep Cohere ONNX CPU and CoreML paths for comparison and validation.

On July 28, 2026, one RTX 4000 Ada host processed a metadata-defined PCM corpus.
The corpus contained `249` sources and `42.442` hours of audio.
All sources completed without a failure.

One worker used two TensorRT sessions and three concurrent requests.
The run reached `78.370x` effective realtime and `75.409x` lifecycle-inclusive realtime.
Median GPU utilization was `82%`, and maximum GPU memory was `9,662 MiB`.

See [NVIDIA Cohere PCM Benchmark](docs/nvidia-cohere-pcm-benchmark-2026-07-28.md)
for the complete configuration, distributions, and bottleneck assessment.

Warm Cohere MLX reached approximately `10.8x` realtime on an 11-second sample.
It reached approximately `15.1x` realtime on a 30-second sample.

Four Cohere TensorRT sessions reached `32.9x` stage realtime on RTX 4000 Ada.
Those sessions used approximately `18.2` to `18.7 GiB` of GPU memory.

Treat these values as hardware-specific measurements.
Rebuild caches and repeat tests after runtime or model changes.

See [macOS Native Inference Spike](docs/macos-native-inference-spike.md) and
[ASR Capability Inventory](docs/asr-capability-inventory.md).

## Internal Cache API

Ingress exposes internal `upload-response` endpoints under `/_upload_response`.
These endpoints support inspection, claims, stage writes, and response writes.

The principal endpoint groups are:

- `/streams` for stream inspection;
- `/streams/{stream_id}/request` for request data;
- `/streams/{stream_id}/stages/{stage}` for stage data;
- `/streams/{stream_id}/readers/{worker_id}` for reader registration;
- `/streams/{stream_id}/response` for response data.

These endpoints are service-internal interfaces.
Client applications must use `/v1/listen`.

## Example Workload

The [Bitneedle scratch tutorial sweep](examples/bitneedle-scratch-tutorial-sweep/README.md)
shows one complete research pipeline.

It resolves media, creates audio chunks, calls `/v1/listen`, and stores derived
research artifacts.

## Verification

Check the Cohere ONNX build:

```bash
cargo check --no-default-features \
  --features cohere-backend,audio-decoder \
  --bin asr-api --bin local-orchestrator --bin cohere-debug
```

Check the Cohere MLX build:

```bash
MACOSX_DEPLOYMENT_TARGET=14.0 \
  cargo check --no-default-features \
  --features cohere-mlx,audio-decoder \
  --bin asr-api --bin local-orchestrator

(cd apple && swift build -c debug)
```

Check the Parakeet TDT build:

```bash
cargo check --no-default-features \
  --features parakeet-backend,audio-decoder \
  --bin asr-api --bin local-orchestrator
```

Run the Rust library tests:

```bash
cargo test --no-default-features \
  --features cohere-backend,audio-decoder \
  --lib
```

## Further Reading

- [ASR Capability Inventory](docs/asr-capability-inventory.md)
- [ASR MLX Bring-Up Findings](docs/asr-mlx-bringup-findings.md)
- [macOS Native Inference Spike](docs/macos-native-inference-spike.md)
- [`asr-onnx` export and runtime guide](../asr-onnx/README.md)
