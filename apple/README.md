# asr-api Apple MLX Runtime

This package provides the Swift/MLX runtime used by `asr-api` on Apple Silicon.

Build it from this directory:

```bash
swift build -c release
```

`asr-api` uses `apple/.build/release/asr-mlx-transcribe` by default when
`ASR_COHERE_BACKEND=mlx` is selected. Set `ASR_MLX_TRANSCRIBE_BIN` to override
the executable path.

The Rust backend runs the executable with `--server`, keeping one model instance
loaded and sending feature-file requests over standard input. Direct one-shot
CLI transcription remains available for debugging.

The package follows the `encodec-rs/apple` pattern and uses `mlx-swift` for the
MLX graph runtime.
