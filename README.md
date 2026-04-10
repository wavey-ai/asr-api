# bag-of-beats

`bag-of-beats` serves NVIDIA Parakeet TDT ONNX models over Wavey's `web-service` stack and uses `upload-response` as the request/response transport.

It accepts raw file uploads on `POST /transcribe`, decodes supported audio formats through `soundkit-decoder`, chunks the decoded mono 16 kHz PCM into overlapped windows, runs `parakeet-rs` on each window, and streams newline-delimited JSON back to the client as transcript segments are committed.

## Environment

- `PARAKEET_MODEL_DIR` (required): directory containing `encoder-model.onnx`, `decoder_joint-model.onnx`, and `vocab.txt`
- `PORT`: TLS port, default `8443`
- `ENABLE_H3`: enable HTTP/3 in addition to HTTP/2
- `TLS_CERT_PATH` / `TLS_KEY_PATH`: optional PEM paths; if omitted the workspace's default local TLS material is used
- `MODEL_INSTANCES`: number of model replicas to load, default `1`
- `CHUNK_SECONDS`: transcription window length, default `30`
- `OVERLAP_SECONDS`: overlap between adjacent windows, default `2`

## Run

```bash
cargo run -- \
  --model-dir /path/to/parakeet-tdt
```

## Upload

Upload raw file bytes and read back NDJSON:

```bash
curl --http2 -k \
  --data-binary @sample.wav \
  https://localhost:8443/transcribe
```

Response events:

- `started`
- `segment`
- `error`
- `done`
