#!/usr/bin/env bash
set -euo pipefail

DEFAULT_REPO_ID="CohereLabs/cohere-transcribe-03-2026"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ID="${DEFAULT_REPO_ID}"
DEST_DIR="${REPO_ROOT}/models/cohere-transcribe-03-2026"
RUN_LOGIN="false"
SKIP_DOWNLOAD="false"
REGENERATE_VOCAB="false"

usage() {
  cat <<'EOF'
Usage:
  setup-cohere-mlx-model.sh [--dest <dir>] [--repo <repo-id>] [--login] [--skip-download] [--regenerate-vocab]

Downloads and prepares the local Cohere Transcribe MLX bundle for asr-api.

Defaults:
  --repo CohereLabs/cohere-transcribe-03-2026
  --dest models/cohere-transcribe-03-2026, relative to the asr-api checkout

Requirements:
  - A Hugging Face login or HF_TOKEN with access to Cohere's gated model
  - huggingface-cli or hf
  - python3 with sentencepiece installed

The script downloads the Hugging Face model files, generates vocab.json from
tokenizer.model when needed, and verifies the files needed by the local MLX
runtime.
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      DEST_DIR="${2:-}"
      shift 2
      ;;
    --repo)
      REPO_ID="${2:-}"
      shift 2
      ;;
    --login)
      RUN_LOGIN="true"
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD="true"
      shift
      ;;
    --regenerate-vocab)
      REGENERATE_VOCAB="true"
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

if [[ -z "$DEST_DIR" || -z "$REPO_ID" ]]; then
  usage >&2
  exit 1
fi

require_cmd python3

mkdir -p "$DEST_DIR"

if [[ "$SKIP_DOWNLOAD" != "true" ]]; then
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI=(huggingface-cli)
    HF_DOWNLOAD=(huggingface-cli download "$REPO_ID" --local-dir "$DEST_DIR")
  elif command -v hf >/dev/null 2>&1; then
    HF_CLI=(hf)
    HF_DOWNLOAD=(hf download "$REPO_ID" --local-dir "$DEST_DIR")
  else
    cat >&2 <<'EOF'
missing Hugging Face CLI

Install it with:
  python3 -m pip install -U huggingface_hub
EOF
    exit 1
  fi

  if [[ "$RUN_LOGIN" == "true" ]]; then
    "${HF_CLI[@]}" login
  fi

  "${HF_DOWNLOAD[@]}"
fi

if [[ "$REGENERATE_VOCAB" == "true" || ! -f "${DEST_DIR%/}/vocab.json" ]]; then
  if ! python3 -c 'import sentencepiece' >/dev/null 2>&1; then
    cat >&2 <<'EOF'
missing Python package: sentencepiece

Install it with:
  python3 -m pip install -U sentencepiece
EOF
    exit 1
  fi
  python3 "$SCRIPT_DIR/cohere-extract-vocab.py" --model-dir "$DEST_DIR"
else
  echo "Using existing vocab.json in ${DEST_DIR}"
fi

missing=()
for file in model.safetensors config.json preprocessor_config.json tokenizer.model vocab.json; do
  if [[ ! -f "${DEST_DIR%/}/$file" ]]; then
    missing+=("$file")
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  printf 'Cohere MLX bundle is missing required files in %s:\n' "$DEST_DIR" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  exit 1
fi

cat <<EOF
Prepared Cohere MLX model bundle:
  ${DEST_DIR}

Use it with:
  ASR_MODEL_DIR=${DEST_DIR}
EOF
