# Model Assets

`asr-api` no longer treats this repo as the source of truth for model payloads.

Model binaries live in Wavey's Linode object-storage bucket and should be synced
into a local or remote model directory before running the worker.

Bucket inventory is described in [bucket-manifest.json](./bucket-manifest.json).

Supported bundles:

- `parakeet-tdt-0.6b-v3`
  - provider: `nemo`
  - default worker dir: `parakeet-tdt`
  - source prefix: `s3://wavey.ai/models/parakeet-tdt-0.6b-v3/`
- `cohere-transcribe-03-2026`
  - provider: `cohere`
  - default worker dir: `cohere-transcribe-03-2026`
  - source prefix: `s3://wavey.ai/models/cohere-transcribe-03-2026/`

To stage a model locally, use:

```bash
AWS_ACCESS_KEY_ID=... \
AWS_SECRET_ACCESS_KEY=... \
scripts/sync-model-from-bucket.sh \
  --model cohere-transcribe-03-2026 \
  --dest /var/lib/asr-api/models/cohere-transcribe-03-2026
```

The repo ignores local payloads under `models/`, so downloaded bundles stay out
of Git.
