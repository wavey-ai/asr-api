# asr-api Apple MLX Runtime

This package is the owned Swift/MLX runtime boundary for `asr-api`.

Build it from this directory:

```bash
swift build -c release
```

`asr-api` uses `apple/.build/release/asr-mlx-transcribe` by default when
`ASR_COHERE_BACKEND=mlx` is selected. Set `ASR_MLX_TRANSCRIBE_BIN` to override
the executable path.

The package intentionally uses `mlx-swift`, matching the `encodec-rs/apple`
pattern, rather than linking Rust directly against `mlx-c`.
