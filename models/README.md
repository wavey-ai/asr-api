# Model Assets

This directory stores checked-in model bundles for `asr-api`.

Current layout:

- `parakeet-tdt-0.6b-v3/`
  - split Parakeet TDT ONNX bundle used by `asr-onnx`
  - includes `featurizer_cuda0.pt` from `asr-torch`
- `cohere-transcribe-03-2026/`
  - Cohere Transcribe ONNX export bundle used by `asr-onnx`
  - includes `encoder.onnx` plus the decoder last-token, prefill, and cached-step graphs

Only large binary assets are tracked through Git LFS. Small text metadata such as `tokens.txt`, `vocab.txt`, and `export.json` stay as normal Git blobs.
