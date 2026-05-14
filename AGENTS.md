Linode Ada Cohere ONNX runbook

Use this when bringing up a temporary NVIDIA GPU box for Cohere ONNX throughput
testing. Keep this scoped to `asr-api`; do not push model payloads from the local
Mac. Model files must come from the object-storage bucket.

Tested target

- Linode plan: `g2-gpu-rtx4000a1-m` (`RTX4000 Ada x1 Medium`, 1 GPU, 32 GB RAM)
- Tested region: `de-fra-2`
- Tested image: `linode/ubuntu22.04`
- Test instance created 2026-05-14:
  - label: `asr-ada-medium-test-20260514143238`
  - id: `97597470`
  - public IP: `172.238.115.254`
- Running tmux session on that host: `asr-cohere-api`
- Public ingress URL: `https://172.238.115.254:18443/v1/listen`

Provisioning

Create the Linode through the Linode API or UI. If using the API, use a local
Linode token from outside the repo, and pass SSH public keys only:

```bash
curl -fsS -X POST https://api.linode.com/v4/linode/instances \
  -H "Authorization: Bearer ${LINODE_TOKEN}" \
  -H 'Content-Type: application/json' \
  --data '{
    "region": "de-fra-2",
    "type": "g2-gpu-rtx4000a1-m",
    "image": "linode/ubuntu22.04",
    "label": "asr-ada-cohere-test",
    "authorized_keys": ["..."]
  }'
```

Base server setup

Fresh Ubuntu 22.04 images do not have the NVIDIA driver. Install build tools,
the newest server driver, Rust, Python CUDA runtime packages, and Opus 1.5.2:

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential pkg-config cmake clang libssl-dev git curl jq tmux \
  python3-venv python3-dev awscli ubuntu-drivers-common pciutils \
  ca-certificates openssl rsync "linux-headers-$(uname -r)" \
  libflac-dev libogg-dev libmp3lame-dev libfdk-aac-dev

drv="$(apt-cache search -n '^nvidia-driver-[0-9]+-server$' | awk '{print $1}' | sort -V | tail -1)"
apt-get install -y "$drv"
nvidia-smi

curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
. /root/.cargo/env
rustup default stable

python3 -m venv /root/bench/venv-ort-cu12
/root/bench/venv-ort-cu12/bin/pip install --upgrade pip
/root/bench/venv-ort-cu12/bin/pip install torch==2.7.0 onnxruntime-gpu==1.23.2
```

Ubuntu 22.04 ships Opus 1.3.1, but the decoder dependency requires
`opus >= 1.5.2`. Build it into `/usr/local`:

```bash
cd /root
curl -fL -o opus-1.5.2.tar.gz https://downloads.xiph.org/releases/opus/opus-1.5.2.tar.gz
tar xf opus-1.5.2.tar.gz
cd opus-1.5.2
./configure --prefix=/usr/local
make -j"$(nproc)"
make install
ldconfig
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --modversion opus
```

Deploy code only

Copy `asr-api` plus local sibling crates. Exclude model directories and build
artifacts:

```bash
rsync -a --delete --exclude .git --exclude target --exclude models \
  ~/wavey.ai/asr-api/ root@HOST:/root/wavey.ai/asr-api/
rsync -a --delete --exclude .git --exclude target \
  ~/wavey.ai/gpu-workers/ root@HOST:/root/wavey.ai/gpu-workers/
rsync -a --delete --exclude .git --exclude target \
  ~/wavey.ai/web-services/ root@HOST:/root/wavey.ai/web-services/
```

For private git dependencies, install a temporary GitHub token on the host, then
remove it after the build:

```bash
scp ~/.gh-token root@HOST:/root/.gh-token
ssh root@HOST 'chmod 600 /root/.gh-token && tok=$(cat /root/.gh-token) && \
  git config --global url."https://x-access-token:${tok}@github.com/wavey-ai/".insteadOf \
  "https://github.com/wavey-ai/"'
```

Model sync from bucket

Do not rsync `models/cohere-transcribe-03-2026` from the local machine. The
bundle is about 9 GB and should be fetched on the Linode:

```bash
cd /root/wavey.ai/asr-api
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export ASR_MODEL_BUCKET_NAME=wavey.ai
export ASR_MODEL_BUCKET_REGION=us-iad
export ASR_MODEL_BUCKET_ENDPOINT=https://us-iad-1.linodeobjects.com

scripts/sync-model-from-bucket.sh \
  --model cohere-transcribe-03-2026 \
  --dest /var/lib/asr-api/models/cohere-transcribe-03-2026

du -sh /var/lib/asr-api/models/cohere-transcribe-03-2026
```

If the default AWS profile returns `403 Forbidden`, create a temporary Linode
Object Storage key scoped read-only to bucket `wavey.ai` in cluster `us-iad-1`.
Delete that key after sync and remove any remote env file containing it.

Dynamic ONNX Runtime on Ubuntu 22.04

The static `ort` 1.24 CUDA artifact failed to link on Ubuntu 22.04 because it
references newer glibc C23 symbols such as `__isoc23_strtoll`. For the Ada
server, use dynamic ONNX Runtime from the `onnxruntime-gpu==1.23.2` Python wheel.
The `asr-api` Cohere path should be built with `ort/load-dynamic` and `api-23`.

Runtime env used for the working host:

```bash
cat >/root/asr-cohere-env.sh <<'EOF'
export ASR_MODEL_PROVIDER=cohere
export ASR_MODEL_DIR=/var/lib/asr-api/models/cohere-transcribe-03-2026
export ASR_DEVICE_IDS=0
export ASR_ONNX_SESSIONS=1
export ASR_TORCH_SESSIONS=1
export ASR_COHERE_MAX_NEW_TOKENS=384
export ASR_COHERE_TRT_COMPONENTS=none
export UPLOAD_RESPONSE_TIMEOUT_MS=180000
export RUST_LOG=info,asr_api=debug,ort=info
export ASR_LOG_FORMAT=compact
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH:-}
export ORT_DYLIB_PATH=/root/bench/venv-ort-cu12/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.2
EOF

python3 - <<'PY' >>/root/asr-cohere-env.sh
from pathlib import Path
base = Path("/root/bench/venv-ort-cu12/lib/python3.10/site-packages")
paths = [
    Path("/usr/local/lib"),
    base / "tensorrt_libs",
    base / "onnxruntime" / "capi",
    base / "torch" / "lib",
]
paths += sorted(base.glob("nvidia/*/lib"))
print('export LD_LIBRARY_PATH="' + ":".join(str(p) for p in paths) + ':${LD_LIBRARY_PATH:-}"')
PY
```

The `tensorrt_libs` entry is required for TensorRT runs; without it,
`libonnxruntime_providers_tensorrt.so` fails to load `libnvinfer.so.10`.

Build

```bash
. /root/.cargo/env
. /root/asr-cohere-env.sh
cd /root/wavey.ai/asr-api
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo build --release \
  --no-default-features \
  --features cohere-backend,audio-decoder \
  --bin asr-api \
  --bin local-orchestrator
```

Start in tmux

```bash
tmux kill-session -t asr-cohere-api 2>/dev/null || true
tmux new-session -d -s asr-cohere-api \
  'cd /root/wavey.ai/asr-api && . /root/.cargo/env && . /root/asr-cohere-env.sh && \
   exec target/release/local-orchestrator \
     --asr-api-bin target/release/asr-api \
     --model-dir /var/lib/asr-api/models/cohere-transcribe-03-2026 \
     --model-provider cohere \
     --device-ids 0 \
     --onnx-sessions 1 \
     --worker-count 1 \
     --ingress-port 18443 \
     --decoder-port 19443 \
     --worker-port 20443 \
     --upload-response-timeout-ms 180000 \
     --upload-response-worker-ttl-ms 30000 \
     --upload-response-worker-heartbeat-interval-ms 1000 \
     --upload-response-max-inflight 2'

tmux capture-pane -t asr-cohere-api -p -S -120
ss -ltnp | grep -E ':(18443|19443|20443)'
nvidia-smi
```

For a full Cohere TensorRT test that covers the normal 30s API window and the
model's 35s cap, add these overrides before `exec target/release/local-orchestrator`:

```bash
export ASR_COHERE_TRT_COMPONENTS=all
export ASR_COHERE_TRT_CACHE_DIR=/var/lib/asr-api/models/cohere-transcribe-03-2026/.trt_cache_all_35s_frames
export ASR_COHERE_TRT_PROFILE_MIN_FRAMES=64
export ASR_COHERE_TRT_PROFILE_OPT_FRAMES=3000
export ASR_COHERE_TRT_PROFILE_MAX_FRAMES=3500
export ASR_COHERE_TRT_FP16=true
```

For multi-worker TensorRT throughput tests, keep `--onnx-sessions 1`, set
`--worker-count 4`, and set `--upload-response-max-inflight 1`. The
orchestrator assigns worker ports starting at `--worker-port` and worker IDs
from `UPLOAD_RESPONSE_WORKER_ID_PREFIX`.

To seed the TensorRT cache from the bucket before startup, use the same cache
directory and profile:

```bash
scripts/sync-trt-cache.sh pull \
  --model cohere-transcribe-03-2026 \
  --cache-id rtx4000-ada-ort1.23.2-trt10-fp16-all-35s \
  --dir /var/lib/asr-api/models/cohere-transcribe-03-2026/.trt_cache_all_35s_frames
```

After a cache has been built and validated on the Ada host, publish it from that
host:

```bash
scripts/sync-trt-cache.sh push \
  --model cohere-transcribe-03-2026 \
  --cache-id rtx4000-ada-ort1.23.2-trt10-fp16-all-35s \
  --dir /var/lib/asr-api/models/cohere-transcribe-03-2026/.trt_cache_all_35s_frames
```

Do not publish TensorRT caches from a different machine. Engine files are tied
to the GPU family, TensorRT/CUDA/ONNX Runtime stack, model files, precision, and
profile shapes.

The Cohere preprocessor uses 100 feature frames/sec, so `3000` frames is 30s
and `3500` frames is 35s. The lower `64`-frame bound covers short final tails
and short HTTP test clips; a one-second min profile (`100` frames) rejected a
94-frame Harvard clip.

Expected readiness signals

- Ports `18443`, `19443`, and `20443` are listening.
- Worker logs include CUDA EP/cuDNN lines such as `cuDNN version: 90501`.
- `nvidia-smi` shows the `asr-api` worker using about 10.5 GB on the RTX 4000 Ada.

Smoke test

From the local machine:

```bash
curl -sk --http1.1 \
  -H 'content-type: application/octet-stream' \
  -H 'accept: application/x-ndjson' \
  --data-binary @~/wavey.ai/asr-load/harvard-lines-s16le/001_the_birch_canoe_slid_on_the_smooth_planks.s16le \
  'https://HOST:18443/v1/listen?encoding=linear16&sample_rate=16000&channels=1&language=en_US'
```

Expected transcript:

- `The birch canoe slid on the smooth planks.`

Throughput quick check from `~/wavey.ai/asr-load`:

```bash
./target/release/asr-load \
  --url 'https://HOST:18443/v1/listen' \
  --dir ./harvard-lines-s16le \
  --start 1 \
  --end 8 \
  --target 50 \
  --h2 \
  --content-type application/octet-stream \
  --accept application/x-ndjson \
  --quiet
```

Known result on 2026-05-14 with `--start 1 --end 1 --target 10`:

- `10 OK / 0 fail`
- mean TTFB: `56.6ms`
- mean part response: `244ms`
- whole-run RTFx: `9.34`

Secret cleanup

After setup/build, remove temporary secrets from the Linode:

```bash
ssh root@HOST 'set -e; \
  if [ -f /root/.gh-token ]; then \
    tok=$(cat /root/.gh-token); \
    git config --global --unset-all "url.https://x-access-token:${tok}@github.com/wavey-ai/.insteadOf" 2>/dev/null || true; \
    rm -f /root/.gh-token; \
  fi; \
  rm -f /root/object-storage-read.env'
```

Current operating context

- Public API shape is Deepgram-compatible `POST /v1/listen`.
- The runtime path is `audio -> soundkit-decoder -> asr-torch -> asr-onnx -> Deepgram-shaped JSON`.
- Do not route new work back through `parakeet-rs` for serving. This repo now fronts the split ONNX/Torch stack directly.

Repos / local dependencies

- Canonical repo: `/Users/jamieb/wavey.ai/asr-api`
- Sibling repos used by the current stack:
  - `/Users/jamieb/wavey.ai/asr-onnx`
  - `/Users/jamieb/wavey.ai/asr-torch`
  - `/Users/jamieb/wavey.ai/soundkit`
  - `/Users/jamieb/wavey.ai/web-services`

Model assets

- Current target model family: `nvidia/parakeet-tdt-0.6b-v3`
- Export bundle location in object storage:
  - `s3://wavey.ai/models/parakeet-tdt-0.6b-v3/`
  - `s3://wavey.ai/models/parakeet-tdt-0.6b-v3.tar.gz`
- Expected local model directory contents:
  - `encoder.onnx`
  - `decoder.onnx`
  - `joint.enc.onnx`
  - `joint.pred.onnx`
  - `joint.joint_net.onnx`
  - `tokens.txt`
  - `vocab.txt`
  - `export.json`
  - `featurizer_cuda0.pt`

Important runtime constraints

- `asr-torch` is on `tch 0.20.0`, which expects PyTorch/libtorch `2.7.0`.
- If you change the torch runtime, retrace `featurizer_cuda0.pt` with that same runtime before debugging anything else.
- The traced featurizer and the libtorch runtime must match. Mixed versions fail in non-obvious ways.
- The ONNX path is CUDA-backed and expects GPU execution for realistic performance.

Known-good GPU host

- Host: `scratch-fm-gpu-de-fra-2-1`
- Public IP: `172.238.102.200`
- Known-good Python env: `/root/bench/venv-torch27`
- Known-good model dir on host: `/root/asr-export/out/parakeet-tdt-0.6b-v3`

Known-good host env

- `LIBTORCH_USE_PYTORCH=1`
- `LIBTORCH_BYPASS_VERSION_CHECK=1`
- `LD_LIBRARY_PATH=/root/bench/venv-torch27/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}`
- `LD_PRELOAD=/root/bench/venv-torch27/lib/python3.12/site-packages/torch/lib/libc10_cuda.so:/root/bench/venv-torch27/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so:/root/bench/venv-torch27/lib/python3.12/site-packages/torch/lib/libtorch_cuda_linalg.so:${LD_PRELOAD:-}`
- `ASR_MODEL_DIR=/root/asr-export/out/parakeet-tdt-0.6b-v3`
- `ASR_DEVICE_IDS=0`
- `ASR_TORCH_SESSIONS=1`
- `ASR_ONNX_SESSIONS=1`
- `PORT=8443`

GPU smoke test

1. Start the server with the env above.
2. Health check:

```bash
curl --http2 -k https://172.238.102.200:8443/healthz
```

3. Listen test:

```bash
curl --http2 -k \
  -H 'Content-Type: audio/wav' \
  --data-binary @sample.wav \
  'https://172.238.102.200:8443/v1/listen?utterances=true&paragraphs=true'
```

Current expected result on the bundled sample:

- Transcript: `A tusk is used to make costly gifts.`

Operational reminders

- `web-service` expects local TLS assets when running the current dev server path. If startup fails in TLS initialization, check sibling `web-services/tls/local.wavey.ai`.
- `upload-response` is the ingress-to-worker handoff for `/v1/listen`; keep worker orchestration on `gpu-worker::upload_response` rather than duplicating cache claim loops in `asr-api`.
- If you need to regenerate model assets, do it from `asr-onnx/export/export_parakeet_tdt.py` and `asr-torch/trace_featurizer.py`, not by trying to split `parakeet-rs` artifacts after export.
