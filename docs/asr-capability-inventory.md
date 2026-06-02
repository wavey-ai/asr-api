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
| Cohere ONNX + Parakeet CTC timestamps | Implemented for local ONNX CPU validation and TensorRT/CUDA deployment | Uses `onnx-community/parakeet-ctc-0.6b-ONNX` as a side-channel aligner over the Cohere transcript. Validated local model is `onnx/model_int8.onnx`; `onnx/model_q4f16.onnx` produced all-NaN logits on M1 CPU. |
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
but returned `141450` NaN logits for JFK (`138 x 1025`), so the default CTC
aligner file is now `onnx/model_int8.onnx`.

TensorRT is still the production target for this aligner on NVIDIA hosts. It is
not an OSX/macOS path: NVIDIA's TensorRT support matrix lists Linux, Windows,
SBSA, and JetPack targets, but no macOS target:
https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html

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
