# macOS Native Inference Spike

## Question

Should local macOS use the existing Cohere ONNX bundle through ONNX Runtime
CoreML / Metal, or switch to a native MLX-style backend for the decoder/LLM
portion?

## Local Findings

- The original Cohere model bundle was an ONNX export, not native MLX/Hugging
  Face weights. It contained `encoder.onnx`, decoder ONNX graphs, and external
  `.onnx.data` files. For MLX testing, `model.safetensors` was synced from
  Hugging Face and `vocab.json` was generated from `tokenizer.model`.
- `second-state/cohere_transcribe_rs` has the shape we want: keep ONNX separate,
  but offer a `--features mlx` path that loads `model.safetensors`, implements
  the Cohere Conformer encoder and transformer decoder in Rust, and runs them
  through MLX C / Metal on Apple Silicon.
- Homebrew ONNX Runtime at `/opt/homebrew/lib/libonnxruntime.dylib` is linked
  against `CoreML.framework` and includes the CoreML execution provider.
- `cohere-debug` successfully transcribed `../whisper.cpp/samples/jfk.wav`
  through the Cohere ONNX path with:
  - `ASR_COHERE_EXECUTION_PROVIDER=metal`
  - `ASR_COHERE_COREML_COMPUTE_UNITS=cpu-and-gpu`
  - `ASR_ONNX_RUNTIME_LIB=/opt/homebrew/lib/libonnxruntime.dylib`
- The output was correct:
  `And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.`
- The local Python environment does not currently have `mlx`, `coremltools`,
  `onnxruntime`, `transformers`, or `torch` installed.
- The existing CoreML cache is small and partitioned, so ONNX Runtime is likely
  offloading supported subgraphs rather than converting the entire Cohere model
  to one native Core ML artifact.
- The first-pass benchmark is CPU-bound in the encoder. The decoder/prefill and
  cached LLM step are not the main bottleneck for the local clips tested.

## Benchmark Notes

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
Stage timings still put most time in `encoder_run_ms`.

`/tmp/asr-cohere-bench-30.wav`, 30.0s audio, CPU ORT default thread pool,
warmup 1, repeat 3:

- `mean_decode_ms=8196.49`
- `mean_rtfx=3.66`
- representative `encoder_run_ms` was `6414ms` to `7590ms`

This does not meet a `10x` real-time target. Removing the one-thread cap helps
substantially, but the current ONNX/CoreML path is still encoder-bound and
CoreML/Metal does not materially beat the CPU path on this export.

`../whisper.cpp/samples/jfk.wav`, 11.0s audio, Cohere MLX, release build,
`ASR_COHERE_BACKEND=mlx`, `ASR_COHERE_TIMINGS=true`, `--max-new-tokens 128`:

| Runtime | Shape | Timing | RTFx | Notes |
| --- | --- | ---: | ---: | --- |
| `asr-api` MLX | init | `14513.94ms` | - | model load and backend setup |
| `asr-api` MLX | warmup 1 | `9409.93ms` | `1.17x` | includes cold Metal compilation |
| `asr-api` MLX | repeat 1 | `2191.12ms` | `5.02x` | still warming |
| `asr-api` MLX | repeats 2-5 | `1006-1021ms` | `10.77-10.93x` | steady warm path |
| upstream `transcribe` CLI | one direct run | `22.64s` | `0.49x` | includes model load each process |
| upstream `transcribe-server` | request 1 | `6.739s` | `1.63x` | model already loaded, cold Metal request |
| upstream `transcribe-server` | requests 3-5 | `0.989-1.003s` | `10.96-11.12x` | steady warm path |

The MLX integration is therefore in the same performance band as the upstream
`second-state/cohere_transcribe_rs` server on this sample. The apparent slow
numbers are cold-start and first-request Metal compilation effects; the steady
warm path reaches the `10x` target.

`/tmp/asr-cohere-bench-30.wav`, 30.0s audio, Cohere MLX, release build,
`--max-new-tokens 128`:

| Runtime | Shape | Timing | RTFx | Notes |
| --- | --- | ---: | ---: | --- |
| `asr-api` MLX | init | `8260.89ms` | - | model load and backend setup |
| `asr-api` MLX | warmup 1 | `6535.53ms` | `4.59x` | cold request |
| `asr-api` MLX | repeats 1-3 | `1986-1989ms` | `15.08-15.10x` | steady warm path |
| upstream `transcribe-server` | request 1 | `7.253s` | `4.14x` | model already loaded, cold request |
| upstream `transcribe-server` | request 3 | `1.981s` | `15.14x` | steady warm path |

## Recommendation

Keep the Cohere ONNX backend as the default production path, and add MLX as a
separate optional feature for Apple Silicon.

This gives us the right split:

- ONNX remains better than libtorch for the existing deployment path.
- MLX is opt-in via `cohere-mlx` and selected with `ASR_COHERE_BACKEND=mlx`.
- The Mac path uses the upstream Rust/MLX implementation instead of trying to
  make ONNX Runtime CoreML accelerate an export that is still encoder-bound.

The operational catch is model packaging: MLX requires the Hugging Face
`model.safetensors` bundle plus `vocab.json`, while ONNX needs the existing
export graph files. The same model directory can hold both sets of artifacts,
but deploy packaging should decide explicitly whether it is shipping ONNX,
MLX, or both.

## MLX Option

Implemented integration shape:

- Cargo feature: `cohere-mlx`
- Runtime selector: `ASR_COHERE_BACKEND=onnx|mlx`
- ONNX remains the default when both `cohere-backend` and `cohere-mlx` are
  compiled.
- `cohere-debug` now goes through `AsrBackend`, so the same debug binary can
  exercise either Cohere backend depending on `ASR_COHERE_BACKEND`.

Build check:

```bash
MACOSX_DEPLOYMENT_TARGET=14.0 \
  cargo check --no-default-features --features cohere-mlx,audio-decoder \
  --bin cohere-debug
```

## References

- ONNX Runtime CoreML Execution Provider:
  https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- MLX:
  https://github.com/ml-explore/mlx
- MLX-LM:
  https://github.com/ml-explore/mlx-lm
- Cohere Transcribe Rust / MLX implementation:
  https://github.com/second-state/cohere_transcribe_rs

## Next Useful Work

1. Run a >35s benchmark with `ASR_COHERE_BACKEND=mlx` to check the chunking path
   with overlap.
2. Run the local three-role service stack with `ASR_COHERE_BACKEND=mlx` and
   confirm end-to-end upload-response timings.
3. Decide model packaging for Mac dev machines: ONNX-only by default plus an
   opt-in safetensors/vocab sync, or a combined ONNX+MLX bundle.
