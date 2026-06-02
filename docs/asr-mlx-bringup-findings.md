# ASR MLX Bring-Up Findings

Status as of 2026-06-02.

This note captures the model sourcing, MLX runtime, performance, quantization,
and parity findings from the ASR API bring-up. It is intentionally operational:
the goal is to preserve what was learned while implementing and testing, not to
describe a final architecture.

## Model Artifacts

Parakeet ONNX artifacts were found in the Linode object bucket:

- Endpoint: `https://us-iad-1.linodeobjects.com`
- Bucket: `wavey.ai`
- Prefix: `models/parakeet-tdt-0.6b-v3/`
- Archive also present: `models/parakeet-tdt-0.6b-v3.tar.gz`

The prefix contains the split ONNX/TDT bundle:

- `encoder.onnx`
- `decoder.onnx`
- `joint.enc.onnx`
- `joint.pred.onnx`
- `joint.joint_net.onnx`
- external `.onnx.data` files
- `tokens.txt`
- `vocab.txt`
- `export.json`
- `SHA256SUMS`

The local copy was synced to:

- `asr-api/models/parakeet-tdt-0.6b-v3`

Checksums passed locally.

Hugging Face token access was checked from this folder. No Wavey-hosted split
Parakeet ONNX bundle was found there. Public HF repos do exist for NVIDIA
Parakeet and MLX community Parakeet, but the exact split ONNX bundle used here
came from Linode.

## Implemented ASR API Surface

The ASR API now has these backend entry points:

- Cohere ONNX, including ONNX Runtime CPU/CUDA/TensorRT/CoreML paths.
- Cohere MLX via an owned Swift/MLX runtime, without `cohere-transcribe-rs`.
- Parakeet ONNX/TDT behind `parakeet-backend`.
- Parakeet pure Rust featurization using `mel-spec` filterbanks.
- Parakeet TDT word timestamps mapped into `TimedWord`.

The Cohere MLX path computes log-mel features in Rust with the shared Cohere
frontend and passes f32 feature tensors to the Swift runtime. This keeps
featurization out of Swift and avoids duplicating frontend behavior.

## Parakeet Featurization Decision

The Parakeet serving path now uses only the pure Rust `mel-spec` frontend.
`asr-api` no longer needs a separate traced frontend runtime, and the bucket
sync script intentionally fetches only the split ONNX/TDT graph files plus
tokens needed for serving.

The current frontend uses sparse filterbank projection and reusable FFT scratch
in `mel-spec::mel`, while preserving the same parity metrics gathered against
the original NeMo frontend.

JFK featurizer benchmark on the M1 Mac after the frame-count fix and sparse
frontend optimization:

| Featurizer | Shape | Mean | p50 | p95 | RTFX |
| --- | --- | ---: | ---: | ---: | ---: |
| `mel-spec` Rust | `128x1101` | `2.341 ms` | `2.334 ms` | `2.406 ms` | `4699.62x` |

Feature comparison over the full `128x1101` tensor:

- MAE: `0.001183`
- RMSE: `0.023699`
- Max absolute error: `3.965733`
- Correlation: `0.999719`

The initial benchmark surfaced a one-frame mismatch: Rust emitted `128x1100`
while the traced NeMo frontend emitted `128x1101`. The Rust Parakeet frontend
now includes the final centered/padded frame, matching the traced frontend
shape. A one-shot Parakeet ONNX CPU smoke after this change still returned the
correct JFK transcript with 22 words at `3948.86 ms`, about `2.79x` realtime.

## Metal And MLX Runtime

SwiftPM did not automatically stage MLX's `default.metallib` for command-line
execution. The MLX loader first checks for a colocated `mlx.metallib`, then
SwiftPM bundle resources.

After the Metal toolchain was downloaded, `xcrun` could find:

- `metal`
- `metallib`

The core MLX Metal library was built from:

- `apple/.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal/*.metal`

and copied to:

- `apple/.build/debug/mlx.metallib`

With that in place, the Swift runtime can initialize MLX Metal and run graph
execution from the command line.

## CPU/Local Smoke Results

These are single-machine smoke results on `../whisper.cpp/samples/jfk.wav`.
They are useful for direction, not final benchmarking.

| Backend | Result | Notes |
| --- | --- | --- |
| Cohere ONNX CPU | Correct transcript | Three-repeat CPU check averaged `18974 ms`, about `0.58x` realtime, with `34411 ms` init. |
| Parakeet ONNX CPU | Correct transcript plus 22 words | Three-repeat CPU check averaged `4662 ms`, about `2.36x` realtime, with `6521 ms` init. |
| Cohere MLX BF16-oriented path | Correct JFK transcript; token sequence matches ONNX on this smoke | Debug Swift executable decoded in `5616 ms`, about `1.96x` realtime. |
| Cohere MLX 4-bit weight quant smoke | Correct JFK transcript | `ASR_COHERE_MLX_QUANT_BITS=4`, group size `64`, affine mode decoded in `5876 ms`, about `1.87x` realtime in the debug Swift executable. Keep quant opt-in pending release/warm-process benchmarking. |
| Cohere MLX float32 weight experiment | Correct after rel-shift fix, but slower | Casting BF16 safetensors to float32 is not needed for parity and reduced performance in earlier tests. |

## Quantization Position

Float-to-int quantization is acceptable for MLX performance work, but it should
remain weights-only and opt-in until broader quality and release-build
benchmarks are complete.

Recommended policy:

- Keep log-mel features and attention activations floating.
- Quantize large linear/matmul weights first.
- Validate transcript quality before enabling by default.
- Keep the default MLX path BF16-oriented for speed, and make fp32 and
  int-quantized paths explicit runtime/debug modes.

Current MLX quant knobs:

- `ASR_COHERE_MLX_QUANT_BITS`
- `ASR_COHERE_MLX_QUANT_GROUP_SIZE`
- `ASR_COHERE_MLX_QUANT_MODE`

The implemented path uses MLX `quantized` plus `quantizedMM` for eligible rank-2
linear weights. The JFK smoke transcript stayed correct with 4-bit affine
weights, but the debug executable was slightly slower than the BF16-oriented
path. Keep this as an explicit tuning mode for now.

One separate performance issue remains: the Rust wrapper currently launches the
Swift executable per transcription window. That is acceptable for bring-up but
not for serving performance. A long-lived Swift runtime process or in-process
FFI boundary should come after graph parity.

## Cohere MLX Parity Findings

The Cohere MLX path runs end to end and now matches the ONNX token sequence on
the JFK smoke sample.

ONNX full token sequence for JFK sample:

```text
[714, 650, 13784, 784, 8444, 13650, 13784, 1881, 676, 761, 821, 2500, 720, 633, 614, 573, 13784, 1881, 761, 573, 720, 633, 614, 821, 2500, 13785]
```

MLX token sequence after the relative-shift fix:

```text
[714, 650, 13784, 784, 8444, 13650, 13784, 1881, 676, 761, 821, 2500, 720, 633, 614, 573, 13784, 1881, 761, 573, 720, 633, 614, 821, 2500, 13785]
```

Before the fix, MLX matched through token 8 and then skipped ONNX token `676`
(`not`). That produced the missing clause:

```text
ask not what your country can do for you
```

What was ruled out during debugging:

- Tokenizer and prompt construction. Prompt IDs match the ONNX backend:
  `[7, 4, 16, 62, 62, 5, 9, 11, 13]`.
- Basic decoder prompt/cache math. Layer-0 self key matches ONNX very closely:
  mean absolute error about `2.3e-7`, max absolute error about `2.5e-6`,
  correlation effectively `1.0`.
- Cached decoder position offsets. Testing offsets `-1` and `+1` did not fix
  the transcript.
- BF16 input features. Running the encoder with f32 features did not fix the
  transcript.
- BF16 safetensors precision alone. Casting loaded BF16 weights to f32 did not
  fix the transcript and reduced performance.
- BF16 layer-0 precision as the primary source. On zero features, forcing f32
  encoder input and f32 weights did not materially improve the layer-0 encoder
  mismatch.
- Relative-position sign/order. ONNX uses a precomputed table with shape
  `[1, 9999, 1280]` and slices axis 1 as `[5000 - steps, 5000 + steps - 1)`.
  That matches the Swift-generated relative position order:
  `steps - 1, ..., 0, -1, ..., -(steps - 1)`.

The root cause was the relative-position attention `relShift` implementation in
the Swift Conformer encoder. ONNX performs:

```text
pad-left -> reshape [B,H,2T,T] -> drop first row -> reshape [B,H,T,2T-1] -> slice last axis to T
```

The Swift port had sliced the reshaped intermediate directly and skipped the
reshape-back step. After matching the ONNX sequence, layer-0 attention parity
improved substantially:

- Relative-position matrix after shift: mean absolute error about `0.0205`,
  correlation about `0.999997`.
- Attention scores: mean absolute error about `0.00815`, correlation about
  `0.999949`.
- Attention output projection: mean absolute error about `0.0101`, correlation
  about `0.999970`.
- Encoder layer-0 final output on zero features: mean absolute error about
  `0.00820`, correlation about `0.999973`.

Earlier evidence that led to the fix:

- ONNX encoder output dump shape: `[1, 138, 1280]`.
- Swift post-projection encoder/cross-attention input shape: `[1, 138, 1024]`.
- `encoder_decoder_proj.weight` shape: `[1024, 1280]`.
- Decoder layer-0 cross-attention weights expect 1024 input:
  `second_sub_layer.key_net.weight` shape `[1024, 1024]`.
- Layer-0 cross key differed substantially before the fix:
  mean absolute error about `0.639`, max absolute error about `5.22`,
  correlation about `0.694`.
- Pre-encoder/subsampling output on zero features is close:
  mean absolute error about `0.00387`, max absolute error about `0.0706`,
  correlation about `0.99987`.
- Encoder layer-0 final output on zero features had matching shape
  `[1, 138, 1280]`, but the default MLX path differs from ONNX by mean absolute
  error about `0.110`, max absolute error about `1.24`, RMS error about
  `0.152`, correlation about `0.995`.
- Re-running that layer-0 dump with `ASR_COHERE_MLX_F32_ENCODER=1` and
  `ASR_COHERE_MLX_F32_WEIGHTS=1` stayed effectively the same, with mean
  absolute error about `0.111` and correlation about `0.995`.

Next engineering steps:

- Replace the per-window Swift executable launch with a long-lived process or
  in-process boundary before serving performance work.
- Run broader transcript parity tests beyond JFK, including longer windows and
  noisy inputs.
- Keep Parakeet MLX as a separate backend gap; the current Parakeet serving path
  is ONNX/TDT with Rust `mel-spec` featurization.

## Diagnostic Env Vars Added

The following opt-in diagnostics were added during bring-up:

- `ASR_COHERE_DEBUG_TOKENS`
- `ASR_COHERE_DUMP_ENCODER`
- `ASR_COHERE_DUMP_SELF_KEY0`
- `ASR_COHERE_DUMP_CROSS_KEY0`
- `ASR_COHERE_MLX_DEBUG_TOKENS`
- `ASR_COHERE_MLX_DEBUG_STDERR`
- `ASR_COHERE_MLX_DUMP_ENCODER`
- `ASR_COHERE_MLX_DUMP_PREENCODE`
- `ASR_COHERE_MLX_DUMP_LAYER0`
- `ASR_COHERE_MLX_DUMP_LAYER0_PREFIX`
- `ASR_COHERE_MLX_DUMP_SELF_KEY0`
- `ASR_COHERE_MLX_DUMP_CROSS_KEY0`
- `ASR_COHERE_MLX_F32_ENCODER`
- `ASR_COHERE_MLX_F32_WEIGHTS`
- `ASR_COHERE_MLX_POSITION_OFFSET`

These should remain debug-only and should not be enabled in production.
