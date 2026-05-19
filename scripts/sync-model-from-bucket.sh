#!/usr/bin/env bash
set -euo pipefail

STATE_FILE_NAME=".bucket-sync-state.v1"

usage() {
  cat <<'EOF'
Usage:
  sync-model-from-bucket.sh --model <model-id> --dest <dir> [--force]

Environment:
  AWS_ACCESS_KEY_ID        Required S3 access key
  AWS_SECRET_ACCESS_KEY    Required S3 secret key
  AWS_SESSION_TOKEN        Optional session token
  ASR_MODEL_BUCKET_NAME    Defaults to wavey.ai
  ASR_MODEL_BUCKET_REGION  Defaults to us-iad
  ASR_MODEL_BUCKET_ENDPOINT Defaults to https://us-iad-1.linodeobjects.com

Supported models:
  cohere-transcribe-03-2026

Behavior:
  The script writes a state file into the destination directory and skips the
  download when the remote object metadata still matches the local model bundle.
  Temporary staging happens beside --dest, not in /tmp.
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
FORCE_SYNC="false"

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
    --force)
      FORCE_SYNC="true"
      shift
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
DEST_PARENT="$(dirname "$DEST_DIR")"
STATE_PATH="${DEST_DIR%/}/${STATE_FILE_NAME}"
STAGED_DIR=""
BACKUP_DIR=""

case "$MODEL_ID" in
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

mkdir -p "$DEST_PARENT"
STAGED_DIR="$(mktemp -d "${DEST_PARENT%/}/.asr-model-sync.XXXXXX")"

cleanup() {
  local exit_code=$?
  trap - EXIT
  if [[ -n "$STAGED_DIR" && -d "$STAGED_DIR" ]]; then
    rm -rf "$STAGED_DIR"
  fi
  if [[ $exit_code -ne 0 && -n "$BACKUP_DIR" && -d "$BACKUP_DIR" && ! -e "$DEST_DIR" ]]; then
    mv "$BACKUP_DIR" "$DEST_DIR"
    BACKUP_DIR=""
  fi
  if [[ -n "$BACKUP_DIR" && -d "$BACKUP_DIR" ]]; then
    rm -rf "$BACKUP_DIR"
  fi
  exit "$exit_code"
}
trap cleanup EXIT

remote_state_path="${STAGED_DIR}/${STATE_FILE_NAME}.remote"

write_remote_state() {
  local state_path="$1"
  {
    printf 'version\t1\n'
    printf 'model\t%s\n' "$MODEL_ID"
    printf 'bucket\t%s\n' "$BUCKET_NAME"
    printf 'region\t%s\n' "$BUCKET_REGION"
    printf 'endpoint\t%s\n' "$BUCKET_ENDPOINT"
    printf 'source_prefix\t%s\n' "$SOURCE_PREFIX"
    for name in "${FILES[@]}"; do
      read -r etag content_length last_modified <<<"$(aws \
        --region "$BUCKET_REGION" \
        --endpoint-url "$BUCKET_ENDPOINT" \
        s3api head-object \
        --bucket "$BUCKET_NAME" \
        --key "${SOURCE_PREFIX}/${name}" \
        --query '[ETag, ContentLength, LastModified]' \
        --output text)"
      etag="${etag#\"}"
      etag="${etag%\"}"
      printf 'file\t%s\t%s\t%s\t%s\n' "$name" "$content_length" "$etag" "$last_modified"
    done
  } >"$state_path"
}

verify_local_payload() {
  local state_path="$1"
  local kind=""
  local name=""
  local size=""
  local _etag=""
  local _modified=""
  while IFS=$'\t' read -r kind name size _etag _modified; do
    [[ "$kind" == "file" ]] || continue
    if [[ ! -f "${DEST_DIR}/${name}" ]]; then
      return 1
    fi
    local local_size
    local_size="$(wc -c < "${DEST_DIR}/${name}")"
    local_size="${local_size//[[:space:]]/}"
    if [[ "$local_size" != "$size" ]]; then
      return 1
    fi
  done <"$state_path"
}

write_remote_state "$remote_state_path"

if [[ "$FORCE_SYNC" != "true" && -f "$STATE_PATH" ]]; then
  if cmp -s "$remote_state_path" "$STATE_PATH" && verify_local_payload "$STATE_PATH"; then
    echo "model ${MODEL_ID} already current at ${DEST_DIR}"
    exit 0
  fi
fi

if [[ "$FORCE_SYNC" != "true" && ! -f "$STATE_PATH" && -d "$DEST_DIR" ]]; then
  if verify_local_payload "$remote_state_path"; then
    cp "$remote_state_path" "$STATE_PATH"
    echo "model ${MODEL_ID} already current at ${DEST_DIR} (state recorded)"
    exit 0
  fi
fi

for name in "${FILES[@]}"; do
  aws \
    --region "$BUCKET_REGION" \
    --endpoint-url "$BUCKET_ENDPOINT" \
    s3 cp \
    "s3://${BUCKET_NAME}/${SOURCE_PREFIX}/${name}" \
    "${STAGED_DIR}/${name}" \
    --no-progress
done

mv "$remote_state_path" "${STAGED_DIR}/${STATE_FILE_NAME}"

if [[ -d "$DEST_DIR" ]]; then
  BACKUP_DIR="${DEST_DIR}.bak.$$"
  rm -rf "$BACKUP_DIR"
  mv "$DEST_DIR" "$BACKUP_DIR"
fi
mv "$STAGED_DIR" "$DEST_DIR"
STAGED_DIR=""
if [[ -n "$BACKUP_DIR" && -d "$BACKUP_DIR" ]]; then
  rm -rf "$BACKUP_DIR"
  BACKUP_DIR=""
fi

echo "synced ${MODEL_ID} into ${DEST_DIR}"
