# ASR Capability Inventory

Status as of 2026-06-02, based on the current local crates and `asr-api`
integration points.

## Serving Surface

`asr-api` is currently a Deepgram-compatible `/v1/listen` service with three
roles:

- `ingress`: accepts buffered HTTP uploads, streaming HTTP request bodies,
  WebSocket audio, and WebTransport over HTTP/3.
- `decoder`: decodes/resamples/downmixes request audio to mono 16 kHz `f32`
  PCM and writes the canonical PCM stage.
- `worker`: runs the configured ASR backend and writes the final
  Deepgram-compatible response.

The service-level model provider enum supports `auto`, `cohere`, and
`parakeet`. `auto` still resolves to Cohere.

## Capability Matrix

| Area | Status | Notes |
| --- | --- | --- |
| Cohere ONNX | Implemented in `asr-api` | Default backend via `cohere-backend`; expects `encoder.onnx`, `decoder_prefill.onnx`, `decoder_cached_step.onnx`, tokenizer, generation, and preprocessor metadata. |
| Cohere ONNX + CUDA | Implemented in `asr-api` | Uses ONNX Runtime CUDA EP when GPU device ids are configured and TensorRT is not selected for a component. |
| Cohere ONNX + TensorRT | Implemented in `asr-api` | Controlled by `ASR_COHERE_TRT_COMPONENTS`; supports `encoder`, `decoder_prefill`, `decoder_cached_step`, `all`, and `none`, plus engine/timing cache and profile envs. |
| Cohere ONNX + CoreML/Metal | Implemented in `asr-api` | Apple ONNX path selected by `ASR_COHERE_COREML=true` or `ASR_COHERE_EXECUTION_PROVIDER=coreml|metal|apple`. |
| Cohere MLX | Implemented; JFK smoke matches ONNX tokens | Optional `cohere-mlx` feature, selected by `ASR_COHERE_BACKEND=mlx`; Rust invokes the owned Swift/MLX runtime in `apple/`. The service no longer depends on `cohere-transcribe-rs` or vendored `mlx-c`. See [ASR MLX Bring-Up Findings](asr-mlx-bringup-findings.md). |
| Cohere word timestamps | Implemented; two modes | Default mode returns monotonic `TimedWord` vectors from transcript token frequency. Cohere ONNX can optionally use the Parakeet CTC ONNX forced aligner with `ASR_COHERE_TIMESTAMP_BACKEND=parakeet-ctc`. Cohere MLX still uses token-frequency timestamps. |
| Cohere ONNX + Parakeet CTC timestamps | Implemented for local ONNX CPU validation and TensorRT/CUDA deployment | Uses `onnx-community/parakeet-ctc-0.6b-ONNX` as a side-channel aligner over the Cohere transcript. Server-side CUDA/TensorRT should use `onnx/model.onnx`; `onnx/model_int8.onnx` was valid but much slower under CUDA and OOMed TensorRT build, while `onnx/model_fp16.onnx` and TensorRT FP16 produced all-NaN logits. |
| Parakeet/TDT ONNX | Wired into `asr-api` behind `parakeet-backend` | Select with `ASR_MODEL_PROVIDER=parakeet`; consumes split TDT exports: `encoder.onnx`, `decoder.onnx`, `joint.enc.onnx`, `joint.pred.onnx`, `joint.joint_net.onnx`, and `tokens.txt`. |
| Parakeet/TDT ONNX + TensorRT | Wired via `asr-onnx` behind `parakeet-backend` | TensorRT configuration is inherited from `asr-onnx` through `ASR_ONNX_TRT_*` envs and component selection. |
| Parakeet pure Rust featurization | Implemented in `asr-api` using `mel-spec` mel frontend | The `parakeet-backend` frontend computes NeMo-style log mel features in Rust with sparse filterbank projection, reusable FFT scratch, and per-feature normalization. This is the only Parakeet featurization path used by `asr-api`. |
| Parakeet TDT timestamps | Exposed through `asr-api` behind `parakeet-backend` | The TDT decoder consumes duration predictions and `asr-api` maps `(word, start_ms, end_ms)` results to `TimedWord`. |
| Parakeet MLX | Paused / not implemented | No local Parakeet MLX backend was found in the inspected crates. Work is paused per current direction. |
| `asr-load` | Tester only | Load/regression client for ASR endpoints; not a serving backend. |

## Gap Summary

The repo now has a buildable Parakeet ONNX/TDT serving path:

- `asr-api` exposes `ASR_MODEL_PROVIDER=parakeet` behind `parakeet-backend`.
- `asr-api` computes pure Rust mel features using the `mel-spec` mel frontend.
- `asr-onnx` runs the split Parakeet/TDT ONNX bundle with CPU, CUDA, or
  TensorRT providers.
- `asr-onnx` produces word timestamp spans from TDT duration predictions.

Parakeet featurization now has one serving route:

- `mel-spec` featurization is the current `asr-api` serving path. It is pure
  Rust, in-process, simple to deploy, and avoids Python/CUDA frontend runtime
  dependencies. After the frame-count fix and sparse/reusable frontend
  optimization it matches the reference frontend shape on JFK (`128x1101`)
  with very small aggregate feature error, and benchmarks close to the previous
  traced CPU comparison on the M1 Mac.

`asr-api` no longer plans to wire a separate traced frontend runtime for
Parakeet. The bucket sync script now fetches only the ONNX/TDT bundle and token
files needed by the service.

Remaining Parakeet gaps:

- resume Parakeet MLX only if Apple-native Parakeet is required;
- broader load-test the split Parakeet ONNX model artifacts.

Separately, Cohere has a production ONNX/TensorRT path and an owned Swift/MLX
runtime. Cohere ONNX now has optional model-derived CTC word timestamps through
the Parakeet CTC side-channel. Cohere MLX remains on token-frequency estimated
timestamps until the Rust wrapper also wires an ONNX/CTC aligner around MLX
decode output.

## Cohere Timestamp Validation

Cohere does not expose a CTC or duration head in the current serving path.
`asr-api` therefore has two timestamp modes:

- Cohere ONNX uses the generated token IDs and decoded token text spans.
- Cohere MLX receives generated token IDs from the Swift runtime and uses the
  same Rust estimator.
- Cohere ONNX can run the Parakeet CTC ONNX side-channel and forced-align the
  Cohere transcript against the original audio.
- Parakeet/TDT remains the validation reference because its decoder emits
  duration-derived word spans.

### Token-Frequency Baseline

Local JFK validation against Parakeet/TDT on 2026-06-02:

```bash
ASR_ONNX_RUNTIME_LIB=/opt/homebrew/lib/libonnxruntime.dylib \
ASR_COHERE_FORCE_CPU=true \
ASR_ONNX_FORCE_CPU=true \
ASR_DEVICE_IDS='' \
cargo run --no-default-features \
  --features cohere-backend,parakeet-backend,audio-decoder \
  --bin timestamp-validate -- \
  --cohere-model-dir models/cohere-transcribe-03-2026 \
  --parakeet-model-dir models/parakeet-tdt-0.6b-v3 \
  --audio-path ../whisper.cpp/samples/jfk.wav \
  --device-ids ''
```

Result:

- Cohere and Parakeet transcript text matched exactly.
- Cohere emitted 22 estimated words; Parakeet emitted 22 TDT words.
- 22/22 normalized words aligned.
- Mean absolute midpoint delta: `728.27 ms`.
- p50 absolute midpoint delta: `845 ms`.
- p95 absolute midpoint delta: `1374 ms`.

Conclusion: token-frequency Cohere timestamps are API-useful approximate
timings, not CTC-quality model alignments.

### Parakeet CTC Side-Channel

Local JFK validation against Parakeet/TDT on 2026-06-02, using
`onnx-community/parakeet-ctc-0.6b-ONNX` with `onnx/model_int8.onnx`:

```bash
ASR_ONNX_RUNTIME_LIB=/opt/homebrew/lib/libonnxruntime.dylib \
ASR_COHERE_FORCE_CPU=true \
ASR_ONNX_FORCE_CPU=true \
ASR_COHERE_TIMESTAMP_BACKEND=parakeet-ctc \
ASR_CTC_ALIGN_MODEL_DIR=models/parakeet-ctc-0.6b-onnx \
ASR_CTC_ALIGN_ONNX_FILE=onnx/model_int8.onnx \
ASR_CTC_ALIGN_FORCE_CPU=true \
ASR_CTC_ALIGN_TIMINGS=true \
ASR_DEVICE_IDS='' \
cargo run --no-default-features \
  --features cohere-backend,parakeet-backend,audio-decoder \
  --bin timestamp-validate -- \
  --cohere-model-dir models/cohere-transcribe-03-2026 \
  --parakeet-model-dir models/parakeet-tdt-0.6b-v3 \
  --audio-path ../whisper.cpp/samples/jfk.wav \
  --device-ids '' \
  --show-words
```

Result:

- Cohere and Parakeet transcript text matched exactly.
- Cohere emitted 22 CTC-aligned words; Parakeet emitted 22 TDT words.
- 22/22 normalized words aligned.
- CTC aligner timing on M1 CPU: `4754.02 ms` for `11.0 s` audio.
- Mean absolute start delta: `90.91 ms`.
- Mean absolute end delta: `89.59 ms`.
- Mean absolute midpoint delta: `59.86 ms`.
- p50 absolute midpoint delta: `55 ms`.
- p95 absolute midpoint delta: `175 ms`.

The first q4f16 ONNX validation attempt loaded and accepted inputs on M1 CPU,
but returned `141450` NaN logits for JFK (`138 x 1025`). `onnx/model_int8.onnx`
was usable for CPU validation, but is not the performance target for CUDA or
TensorRT. The server-side default is now the full-float `onnx/model.onnx`.

TensorRT is still the production target for this aligner on NVIDIA hosts. It is
not an OSX/macOS path: NVIDIA's TensorRT support matrix lists Linux, Windows,
SBSA, and JetPack targets, but no macOS target:
https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html

### Parakeet CTC Sidecar Benchmark

Linode benchmark on 2026-06-02:

- Instance: `g2-gpu-rtx4000a1-m` / RTX 4000 Ada / 32 GiB system RAM.
- Driver: `610.43.02`.
- Runtime for ORT: CUDA 12.8 user-space libraries in
  `/opt/cuda-12.8-runtime`; ONNX Runtime TensorRT/CUDA provider linkage had no
  missing `ldd` dependencies after installing CUDA 12.8 cuBLAS/cuFFT/cuDNN.
- Model: `onnx-community/parakeet-ctc-0.6b-ONNX`.
- Corpus: CohereX fixture transcripts and audio files; transcript text was
  used only as sidecar alignment input, with no Cohere ONNX decode in-process.
- Binary: `ctc-align-debug`.

Incorrect artifact/runtime findings:

- `onnx/model_int8.onnx` is a poor TensorRT target here: isolated TensorRT
  builds OOM-killed around `31.6 GiB` RSS before writing an engine cache.
- `onnx/model_int8.onnx` also ran much slower under CUDA EP than the full-float
  graph.
- `onnx/model_fp16.onnx` loaded and TensorRT could build it, but CUDA and
  TensorRT both returned all-NaN logits for the tested fixture.
- `onnx/model.onnx` is the usable server artifact. CUDA EP returns finite logits
  and TensorRT returns finite logits when `ASR_CTC_ALIGN_TRT_FP16=false`.
- TensorRT FP16 builder mode returned all-NaN logits for this graph/runtime even
  when starting from the full-float ONNX graph, so CTC TensorRT FP16 is now
  opt-in rather than the default.

Full-float CUDA EP sidecar-only runtime, using one warmup and five measured
repeats:

| File | Audio | Words | Mean Align | Realtime |
| --- | ---: | ---: | ---: | ---: |
| `amelia_earhart_noisy.c431d09f.wav` | `31.800 s` | `74` | `46.44 ms` | `684.82x` |
| `david-gooding-noisy.mp3` | `30.154 s` | `86` | `38.74 ms` | `778.32x` |
| `hur.mp3` | `165.295 s` | `411` | `479.73 ms` | `344.56x` |

Measured sidecar memory footprint:

| Mode | File | Peak Host RSS | Peak GPU Memory |
| --- | --- | ---: | ---: |
| CUDA EP, full-float ONNX | `amelia_earhart_noisy.c431d09f.wav` | `1081.9 MiB` | `3256 MiB` |
| CUDA EP, full-float ONNX | `hur.mp3` | `1187.1 MiB` | `6332 MiB` |
| TensorRT cached engine, full-float ONNX, FP16 disabled | `amelia_earhart_noisy.c431d09f.wav` | `8408.8 MiB` | `3270 MiB` |
| TensorRT fresh build, full-float ONNX, FP16 disabled | `amelia_earhart_noisy.c431d09f.wav` | `8948.9 MiB` | `3270 MiB` |

Full-float TensorRT sidecar runtime with `ASR_CTC_ALIGN_TRT_FP16=false`,
35-second profile, one warmup, and five measured repeats:

| File | Audio | Words | Mean Align | Realtime |
| --- | ---: | ---: | ---: | ---: |
| `amelia_earhart_noisy.c431d09f.wav` | `31.800 s` | `74` | `78.61 ms` | `404.53x` |
| `david-gooding-noisy.mp3` | `30.154 s` | `86` | `68.18 ms` | `442.24x` |

The cached TensorRT engine for the 35-second profile was `2.4 GiB` on disk.
The full-float CUDA EP was faster than this constrained TensorRT engine on the
short fixtures, but both are fast enough for tiny-instance sidecar deployment.

### Parakeet Sidecar vs CohereX

Comparison target: CohereX default wav2vec2 alignment, loaded on CUDA
(`metadata_type=torchaudio`, model device `cuda:0`), using the same CohereX
fixture transcript segments and audio. This compares sidecar agreement with
CohereX's stored wav2vec2 regression alignment, not absolute timestamp accuracy
against a manually annotated corpus.

CohereX wav2vec2 alignment speed:

| File | Audio | Words | Mean Align | Realtime |
| --- | ---: | ---: | ---: | ---: |
| `amelia_earhart_noisy.c431d09f.wav` | `31.800 s` | `74` | `258.50 ms` | `123.02x` |
| `david-gooding-noisy.mp3` | `30.096 s` | `86` | `201.36 ms` | `149.46x` |
| `hur.mp3` | `165.326 s` | `411` | `1102.25 ms` | `149.99x` |

CohereX alignment memory across the same run: peak host RSS `1512.1 MiB`, peak
GPU memory `1848 MiB`.

Speed ratio versus CohereX wav2vec2:

| File | Parakeet CUDA | Parakeet TensorRT |
| --- | ---: | ---: |
| `amelia_earhart_noisy.c431d09f.wav` | `5.57x` faster | `3.33x` faster |
| `david-gooding-noisy.mp3` | `5.20x` faster | `3.00x` faster |
| `hur.mp3` | `2.30x` faster | Not run with the 35s TensorRT profile |

Parakeet sidecar word-boundary deltas versus CohereX's stored wav2vec2
alignment:

| File | Matched | Mean Start Delta | Mean End Delta | Mean Midpoint Delta | p50 Mid | p95 Mid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `amelia_earhart_noisy.c431d09f.wav` | `74/74` | `38.76 ms` | `134.38 ms` | `74.70 ms` | `41 ms` | `290 ms` |
| `david-gooding-noisy.mp3` | `86/86` | `80.40 ms` | `195.35 ms` | `136.37 ms` | `110 ms` | `360 ms` |
| `hur.mp3` | `411/411` | `75.26 ms` | `176.86 ms` | `114.94 ms` | `70 ms` | `416 ms` |

CohereX reproduces its own regression alignment with zero deltas by definition
for this comparison. The Parakeet sidecar therefore trades some disagreement
with CohereX's wav2vec2 boundaries for substantially higher speed.

This benchmark is a performance-only sidecar measurement. It intentionally does
not compare word-boundary accuracy against CohereX's alignment JSON.

### CohereX Comparison

[CohereX](https://github.com/Diffio-AI/CohereX) is also a side-channel
alignment design: it runs Cohere ASR, then aligns the transcript with wav2vec2
by default, or optional Qwen3 / NeMo Conformer CTC backends. Their README
reports a 48m28s RTX 6000 Ada full-pipeline benchmark at `260.36x` realtime
for CohereX + Cohere + FireRedVAD, and TIMIT forced-alignment measurements of
about `29 ms` clean boundary MAE for Qwen3, `45 ms` for wav2vec2, and `68 ms`
for NeMo Conformer CTC.

Our current JFK number is not the same benchmark: it is one 11s utterance on M1
CPU, measured as Cohere/Parakeet midpoint deltas rather than reference boundary
MAE. It is still directionally useful: the Parakeet CTC side-channel brings the
local midpoint error down from `728.27 ms` to `59.86 ms`. The remaining
comparison gap is a Linux/NVIDIA TensorRT benchmark using the same audio corpus
and a boundary-MAE metric.
