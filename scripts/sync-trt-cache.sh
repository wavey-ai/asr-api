#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sync-trt-cache.sh pull --model <model-id> --cache-id <id> --dir <cache-dir>
  sync-trt-cache.sh push --model <model-id> --cache-id <id> --dir <cache-dir>

Environment:
  AWS_ACCESS_KEY_ID         Required S3 access key
  AWS_SECRET_ACCESS_KEY     Required S3 secret key
  AWS_SESSION_TOKEN         Optional session token
  ASR_MODEL_BUCKET_NAME     Defaults to wavey.ai
  ASR_MODEL_BUCKET_REGION   Defaults to us-iad
  ASR_MODEL_BUCKET_ENDPOINT Defaults to https://us-iad-1.linodeobjects.com

Notes:
  TensorRT engines are not portable across arbitrary GPU, TensorRT, CUDA,
  ONNX Runtime, model, precision, and profile-shape changes. Use a cache id
  that names those compatibility dimensions.
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

ACTION="${1:-}"
if [[ "$ACTION" == "pull" || "$ACTION" == "push" ]]; then
  shift
else
  usage >&2
  exit 1
fi

MODEL_ID=""
CACHE_ID=""
CACHE_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_ID="${2:-}"
      shift 2
      ;;
    --cache-id)
      CACHE_ID="${2:-}"
      shift 2
      ;;
    --dir)
      CACHE_DIR="${2:-}"
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

if [[ -z "$MODEL_ID" || -z "$CACHE_ID" || -z "$CACHE_DIR" ]]; then
  usage >&2
  exit 1
fi

: "${AWS_ACCESS_KEY_ID:?set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?set AWS_SECRET_ACCESS_KEY}"

require_cmd aws

BUCKET_NAME="${ASR_MODEL_BUCKET_NAME:-wavey.ai}"
BUCKET_REGION="${ASR_MODEL_BUCKET_REGION:-us-iad}"
BUCKET_ENDPOINT="${ASR_MODEL_BUCKET_ENDPOINT:-https://us-iad-1.linodeobjects.com}"

case "$MODEL_ID" in
  cohere-transcribe-03-2026|parakeet-tdt-0.6b-v3)
    ;;
  *)
    echo "unsupported model: $MODEL_ID" >&2
    exit 1
    ;;
esac

SOURCE_PREFIX="models/${MODEL_ID}/trt-cache/${CACHE_ID}"
REMOTE_URI="s3://${BUCKET_NAME}/${SOURCE_PREFIX}/"

write_manifest() {
  local path="$1"
  {
    printf 'version\t1\n'
    printf 'model\t%s\n' "$MODEL_ID"
    printf 'cache_id\t%s\n' "$CACHE_ID"
    printf 'bucket\t%s\n' "$BUCKET_NAME"
    printf 'region\t%s\n' "$BUCKET_REGION"
    printf 'endpoint\t%s\n' "$BUCKET_ENDPOINT"
    printf 'source_prefix\t%s\n' "$SOURCE_PREFIX"
    printf 'created_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host\t%s\n' "$(hostname 2>/dev/null || true)"
    printf 'uname\t%s\n' "$(uname -a 2>/dev/null || true)"
    printf 'ASR_COHERE_TRT_COMPONENTS\t%s\n' "${ASR_COHERE_TRT_COMPONENTS:-}"
    printf 'ASR_COHERE_TRT_PROFILE_MIN_FRAMES\t%s\n' "${ASR_COHERE_TRT_PROFILE_MIN_FRAMES:-}"
    printf 'ASR_COHERE_TRT_PROFILE_OPT_FRAMES\t%s\n' "${ASR_COHERE_TRT_PROFILE_OPT_FRAMES:-}"
    printf 'ASR_COHERE_TRT_PROFILE_MAX_FRAMES\t%s\n' "${ASR_COHERE_TRT_PROFILE_MAX_FRAMES:-}"
    printf 'ASR_COHERE_TRT_FP16\t%s\n' "${ASR_COHERE_TRT_FP16:-}"
    printf 'ORT_DYLIB_PATH\t%s\n' "${ORT_DYLIB_PATH:-}"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader 2>/dev/null |
        while IFS= read -r line; do
          printf 'nvidia_smi\t%s\n' "$line"
        done
    fi
    find "$CACHE_DIR" -type f | sort | while IFS= read -r file; do
      rel="${file#"$CACHE_DIR"/}"
      size="$(wc -c < "$file")"
      size="${size//[[:space:]]/}"
      printf 'file\t%s\t%s\n' "$rel" "$size"
    done
  } >"$path"
}

case "$ACTION" in
  pull)
    cache_parent="$(dirname "$CACHE_DIR")"
    mkdir -p "$cache_parent"
    staged_dir="$(mktemp -d "${cache_parent%/}/.trt-cache-sync.XXXXXX")"
    backup_dir=""
    cleanup() {
      local exit_code=$?
      trap - EXIT
      if [[ -n "${staged_dir:-}" && -d "$staged_dir" ]]; then
        rm -rf "$staged_dir"
      fi
      if [[ $exit_code -ne 0 && -n "${backup_dir:-}" && -d "$backup_dir" && ! -e "$CACHE_DIR" ]]; then
        mv "$backup_dir" "$CACHE_DIR"
        backup_dir=""
      fi
      if [[ -n "${backup_dir:-}" && -d "$backup_dir" ]]; then
        rm -rf "$backup_dir"
      fi
      exit "$exit_code"
    }
    trap cleanup EXIT

    aws \
      --region "$BUCKET_REGION" \
      --endpoint-url "$BUCKET_ENDPOINT" \
      s3 sync \
      "$REMOTE_URI" \
      "$staged_dir/" \
      --no-progress

    if [[ -d "$CACHE_DIR" ]]; then
      backup_dir="${CACHE_DIR}.bak.$$"
      rm -rf "$backup_dir"
      mv "$CACHE_DIR" "$backup_dir"
    fi
    mv "$staged_dir" "$CACHE_DIR"
    staged_dir=""
    if [[ -n "$backup_dir" && -d "$backup_dir" ]]; then
      rm -rf "$backup_dir"
      backup_dir=""
    fi
    trap - EXIT
    echo "synced TensorRT cache ${CACHE_ID} into ${CACHE_DIR}"
    ;;
  push)
    if [[ ! -d "$CACHE_DIR" ]]; then
      echo "cache directory does not exist: $CACHE_DIR" >&2
      exit 1
    fi
    manifest_path="$(mktemp)"
    cleanup() {
      local exit_code=$?
      trap - EXIT
      rm -f "$manifest_path"
      exit "$exit_code"
    }
    trap cleanup EXIT
    write_manifest "$manifest_path"
    aws \
      --region "$BUCKET_REGION" \
      --endpoint-url "$BUCKET_ENDPOINT" \
      s3 sync \
      "$CACHE_DIR/" \
      "$REMOTE_URI" \
      --delete \
      --no-progress
    aws \
      --region "$BUCKET_REGION" \
      --endpoint-url "$BUCKET_ENDPOINT" \
      s3 cp \
      "$manifest_path" \
      "${REMOTE_URI}trt-cache-manifest.tsv" \
      --no-progress
    trap - EXIT
    rm -f "$manifest_path"
    echo "published TensorRT cache ${CACHE_ID} from ${CACHE_DIR} to ${REMOTE_URI}"
    ;;
esac
