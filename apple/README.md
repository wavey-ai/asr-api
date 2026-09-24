# asr-api Apple MLX Runtime

This package provides the Swift/MLX runtime used by `asr-api` on Apple Silicon.

Build it from this directory:

```bash
./build.sh
```

The build requires Xcode's `metal` and `metallib` tools.
Swift Package Manager builds the executable but can leave out the MLX Metal library.
`build.sh` compiles the generated MLX shaders when the library is missing or older than its sources.
It puts `mlx.metallib` beside `apple/.build/release/asr-mlx-transcribe`.
Run `./build.sh --force-metal` to rebuild the library after a toolchain change.

`asr-api` uses `apple/.build/release/asr-mlx-transcribe` by default when
`ASR_COHERE_BACKEND=mlx` is selected. Set `ASR_MLX_TRANSCRIBE_BIN` to override
the executable path.

The Rust backend runs the executable with `--server`, keeping one model instance
loaded and sending feature-file requests over standard input. Direct one-shot
CLI transcription remains available for debugging.

The package follows the `encodec-rs/apple` pattern and uses `mlx-swift` for the
MLX graph runtime.
