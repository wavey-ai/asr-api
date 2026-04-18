#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sync-model-from-bucket.sh --model <model-id> --dest <dir>

Environment:
  AWS_ACCESS_KEY_ID        Required S3 access key
  AWS_SECRET_ACCESS_KEY    Required S3 secret key
  AWS_SESSION_TOKEN        Optional session token
  ASR_MODEL_BUCKET_NAME    Defaults to wavey.ai
  ASR_MODEL_BUCKET_REGION  Defaults to us-iad
  ASR_MODEL_BUCKET_ENDPOINT Defaults to https://us-iad-1.linodeobjects.com

Supported models:
  parakeet-tdt-0.6b-v3
  cohere-transcribe-03-2026
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

MODEL_ID=""
DEST_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_ID="${2:-}"
      shift 2
      ;;
    --dest)
      DEST_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL_ID" || -z "$DEST_DIR" ]]; then
  usage >&2
  exit 1
fi

: "${AWS_ACCESS_KEY_ID:?set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?set AWS_SECRET_ACCESS_KEY}"

require_cmd aws

BUCKET_NAME="${ASR_MODEL_BUCKET_NAME:-wavey.ai}"
BUCKET_REGION="${ASR_MODEL_BUCKET_REGION:-us-iad}"
BUCKET_ENDPOINT="${ASR_MODEL_BUCKET_ENDPOINT:-https://us-iad-1.linodeobjects.com}"

declare -a FILES=()
SOURCE_PREFIX=""

case "$MODEL_ID" in
  parakeet-tdt-0.6b-v3)
    SOURCE_PREFIX="models/parakeet-tdt-0.6b-v3"
    FILES=(
      SHA256SUMS
      decoder.onnx
      decoder.onnx.data
      encoder.onnx
      encoder.onnx.data
      export.json
      featurizer_cuda0.pt
      joint.enc.onnx
      joint.enc.onnx.data
      joint.joint_net.onnx
      joint.joint_net.onnx.data
      joint.pred.onnx
      joint.pred.onnx.data
      tokens.txt
      vocab.txt
    )
    ;;
  cohere-transcribe-03-2026)
    SOURCE_PREFIX="models/cohere-transcribe-03-2026"
    FILES=(
      config.json
      decoder_cached_step.onnx
      decoder_cached_step.onnx.data
      decoder_last_token.onnx
      decoder_last_token.onnx.data
      decoder_prefill.onnx
      decoder_prefill.onnx.data
      encoder.onnx
      encoder.onnx.data
      export.json
      generation_config.json
      preprocessor_config.json
      processor_config.json
      special_tokens_map.json
      tokenizer.json
      tokenizer.model
      tokenizer_config.json
    )
    ;;
  *)
    echo "unsupported model: $MODEL_ID" >&2
    exit 1
    ;;
esac

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

mkdir -p "$tmpdir" "$DEST_DIR"

for name in "${FILES[@]}"; do
  aws \
    --region "$BUCKET_REGION" \
    --endpoint-url "$BUCKET_ENDPOINT" \
    s3 cp \
    "s3://${BUCKET_NAME}/${SOURCE_PREFIX}/${name}" \
    "${tmpdir}/${name}" \
    --no-progress
done

find "$DEST_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
cp -R "${tmpdir}/." "$DEST_DIR/"

echo "synced ${MODEL_ID} into ${DEST_DIR}"
