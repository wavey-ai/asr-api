# TODO: Complete the Audiomovers ASR benchmark

Updated: 2026-07-28

## Objective

Complete the paused Audiomovers transcription run with MPX.
Measure ASR throughput on the local Mac and one Linode GPU host.
Tune ONNX session, request concurrency, and worker settings on Linode.
Commit the scripts, measured results, and conclusions.

## Important constraints

- Use pure-Rust `libopus-rs`.
- Do not install or use native `libopus`.
- Use the `main` branch for Wavey Git dependencies.
- Do not put Git commit hashes in dependency declarations.
- Do not upload model files from the Mac.
- Make the Linode download the gated model from Hugging Face.
- Do not print or commit any credential.
- Delete the Linode after all remote results are retrieved.
- Follow `/Users/jamie/wavey.ai/AGENTS.md`.

## Credential and key paths

These paths contain the required credentials:

```text
/Users/jamie/wavey.ai/.linode-token
/Users/jamie/wavey.ai/.hf_token
```

The Linode SSH key is:

```text
/Users/jamie/wavey.ai/media-research-stack/target/linode/ssh_key
/Users/jamie/wavey.ai/media-research-stack/target/linode/ssh_key.pub
```

Read each token only in the command that needs it.
Never print, copy, document, or commit a token value.
The Hugging Face token has access to the gated Cohere model.

## Current Linode state

No benchmark Linode currently exists.
The Small instance was deleted and verified as absent.

Deleted instance details:

```text
ID: 101554099
Label: asr-mpx-rtx4000a-us-sea-20260727
Region: us-sea
Old address: 172.234.238.99
```

Do not recreate a Small instance.
Create one `RTX4000 Ada x1 Medium` instance in `us-sea`.

```text
Linode type: g2-gpu-rtx4000a1-m
Host memory: 32768 MiB
Disk: 524288 MiB
GPU: one RTX 4000 Ada with 20 GB
Price observed on 2026-07-28: USD 0.67 per hour
```

The Medium plan is authorized.
Create it only after the pending scripts are committed and pushed.

## Model and exporter

Use this gated source model:

```text
CohereLabs/cohere-transcribe-03-2026
```

The authoritative exporter is in the sibling `asr-onnx` repository:

```text
/Users/jamie/wavey.ai/asr-onnx/export/export_cohere_transcribe.py
/Users/jamie/wavey.ai/asr-onnx/export/export_cohere_bundle.py
/Users/jamie/wavey.ai/asr-onnx/python/setup-export-env.sh
```

The locked export environment uses CUDA 12.8 and PyTorch 2.10.
The single-model exporter creates these four ONNX graphs:

```text
encoder.onnx
decoder_last_token.onnx
decoder_prefill.onnx
decoder_cached_step.onnx
```

Each graph has an external `.onnx.data` file.
The complete runtime bundle is approximately 8.9 GB.
The gated Hugging Face checkpoint is approximately 4.1 GB.

The Linode export script validates all four graphs.
It creates `SHA256SUMS` before it publishes the model directory.
It uses a staging directory under `/opt/asr-bench/models`.

## Repository state

These supporting changes are committed and pushed:

```text
soundkit 2c42e12 Use one frame header source across SoundKit
soundkit 63f2319 Use the workspace SoundKit crate for FLAC
av-api   0c3ab4e Use pure Rust Opus through SoundKit main
asr-api  5d4473f Use pure Rust Opus dependencies from main
av-ingest 08567cb Allow low-bandwidth WebM audio caching
```

`soundkit` resolves Opus through `libopus-rs`.
The checked dependency tree did not contain `opus-sys`.

The `asr-api` repository has an unrelated modified `README.md`.
Preserve that change.
Do not include it in a handoff-only commit without review.

The `media-research-stack` repository contains uncommitted work.
That work includes the cache split, concurrency controls, metrics, and scripts.
Its `Cargo.lock` was refreshed immediately before this handoff.
A locked test build was stopped during compilation at the user's request.

Review this repository first:

```text
/Users/jamie/wavey.ai/media-research-stack
```

Important new files include:

```text
src/lib.rs
src/research_cache.rs
src/bin/cache-research-media.rs
scripts/watch-asr-throughput.py
scripts/run-asr-benchmark-matrix.py
scripts/prepare-asr-benchmark-dataset.py
scripts/compare-asr-benchmarks.py
scripts/bootstrap-ubuntu-nvidia-asr.sh
scripts/linode-asr-benchmark-instance.sh
scripts/sync-asr-benchmark-assets.sh
scripts/export-cohere-onnx-on-linux.sh
scripts/run-remote-cohere-export.sh
scripts/run-local-asr-baseline.sh
```

The Linode helper now defaults to `g2-gpu-rtx4000a1-m`.
The asset sync clones each repository from `main`.
It uploads only the 40 MiB benchmark data set.
It does not upload any model.

## Benchmark data

Use this fixed local data set for both hosts:

```text
/Users/jamie/wavey.ai/media-research-stack/target/audiomovers/benchmark-10
```

It contains 10 cached WebM/Opus sources.
It contains 4,311 seconds of audio.
Its total duration is 71.85 minutes.
Its size is approximately 40 MiB.

The manifest fingerprint is:

```text
84481374d3749f5a716d09190009e8b9802205ee5d5f41a434e9052c02feb21c
```

Use the same files on both hosts.
This keeps the architecture comparison repeatable.

## Immediate local checks

Run these checks before the first commit:

```sh
cd /Users/jamie/wavey.ai/media-research-stack

for script in scripts/*.sh; do
  bash -n "$script" || exit
done

PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/compare-asr-benchmarks.py \
  scripts/prepare-asr-benchmark-dataset.py \
  scripts/run-asr-benchmark-matrix.py \
  scripts/watch-asr-throughput.py

cargo fmt --check
CARGO_BUILD_JOBS=2 MACOSX_DEPLOYMENT_TARGET=14.0 \
  cargo test --locked --test mastering_videos

cargo tree --locked | rg 'libopus-rs|opus-sys'
git diff --check
git status --short
```

The expected tree contains `libopus-rs`.
The expected tree does not contain `opus-sys`.
Inspect all changes before staging them.
Do not stage anything under `target/`.

Update the README with the remote procedure.
Commit and push `media-research-stack` to `main`.
The Medium host must clone this pushed revision.

## Create and prepare the Medium Linode

Create the host from `media-research-stack`:

```sh
cd /Users/jamie/wavey.ai/media-research-stack

scripts/linode-asr-benchmark-instance.sh create \
  --token-file ../.linode-token \
  --ssh-public-key target/linode/ssh_key.pub \
  --type g2-gpu-rtx4000a1-m \
  --region us-sea \
  --label asr-mpx-rtx4000a-medium-us-sea-20260728
```

The helper stores non-secret state here:

```text
target/linode/instance.json
```

Wait until SSH is ready.
Read the address from the state file.
Do not paste the token into a shell argument.

Clone the source and transfer the benchmark data:

```sh
scripts/sync-asr-benchmark-assets.sh \
  --host root@LINODE_ADDRESS \
  --identity target/linode/ssh_key
```

This command transfers only the 40 MiB benchmark data set.
All source repositories clone from GitHub `main`.

Install the pinned GPU environment:

```sh
ssh -i target/linode/ssh_key root@LINODE_ADDRESS \
  'bash /opt/asr-bench/media-research-stack/scripts/bootstrap-ubuntu-nvidia-asr.sh --reboot'
```

Wait for the reboot.
Then verify `nvidia-smi`.

## Export the ONNX model on Linode

Run the wrapper from the Mac:

```sh
cd /Users/jamie/wavey.ai/media-research-stack

scripts/run-remote-cohere-export.sh \
  --host root@LINODE_ADDRESS \
  --identity target/linode/ssh_key \
  --token-file ../.hf_token
```

The wrapper sends the token through SSH standard input.
It does not create a remote token file.
The Linode downloads the checkpoint directly from Hugging Face.

The expected model directory is:

```text
/opt/asr-bench/models/cohere-transcribe-03-2026
```

Verify `SHA256SUMS` after export.
Verify all graph and data files exist.
Run an ONNX load or ASR smoke test before the full matrix.

## Linode benchmark

Build the benchmark from the remote `main` checkout.
Use the installed ONNX Runtime GPU library.

Start with a one-session CUDA smoke test.
Then run the TensorRT matrix.
Keep failed configurations in the results.
Some four-session settings can exceed 20 GB of GPU memory.

The current default matrix is:

```text
1:1:1
1:2:1
2:2:1
2:4:1
3:3:1
3:6:1
4:4:1
4:8:1
4:8:2
4:8:4
```

Each value means `sessions:concurrency:workers`.
The runner records service RTFx and effective RTFx.
It also records GPU use, memory, temperature, and power.

Use:

```text
/opt/asr-bench/media-research-stack/scripts/run-asr-benchmark-matrix.py
```

Do not select a winner from service RTFx alone.
Select the stable configuration with the best effective RTFx.
Reject configurations that fail, exhaust memory, or reduce throughput.

## Local benchmark

Run the matching fixed data set on Apple Silicon:

```sh
cd /Users/jamie/wavey.ai/media-research-stack
scripts/run-local-asr-baseline.sh
```

Compare the local MLX run with the best Linode run:

```text
scripts/compare-asr-benchmarks.py
```

Record model, host, architecture, data fingerprint, and configuration.
Record both service RTFx and effective RTFx.

## Audiomovers sweep

The complete cache contains 249 WebM sources.
Its audio duration is approximately 42.44 hours.
The cache process completed before this handoff.

An earlier local ASR run reached at least source 20.
It reported approximately 4.92 times real-time ASR service throughput.
That process used an older binary.

Check for an active process before starting another run.
Stop the old process at a source boundary if it still exists.
Build a new binary that uses pure-Rust Opus.
Resume from the existing report with:

```text
MEDIA_RESEARCH_STACK_RESUME=1
MEDIA_RESEARCH_STACK_REQUIRE_CACHE=1
```

Verify the final report has one successful row per source.
Verify the required transcript files exist.
Do not commit cached media or full transcripts.

## Results and shutdown

- Copy compact benchmark summaries from Linode to the Mac.
- Do not copy the exported model unless the user requests it.
- Add the measured matrix and conclusion to `README.md`.
- Keep documentation sentences short and direct.
- Commit and push scripts, summaries, and documentation.
- Do not commit tokens, model files, media, or transcripts.
- Delete the Medium Linode after all results are retrieved.
- Verify the instance API returns `404` after deletion.

Delete the host with:

```sh
scripts/linode-asr-benchmark-instance.sh delete \
  --token-file ../.linode-token \
  --confirm-delete
```

## Completion checklist

- [ ] Validate the pending `media-research-stack` changes.
- [ ] Commit and push the cache and benchmark scripts.
- [ ] Create one RTX4000 Ada Medium host in `us-sea`.
- [ ] Bootstrap the pinned NVIDIA environment.
- [ ] Export and checksum all four ONNX graphs on Linode.
- [ ] Verify the pure-Rust Opus decode path on Linux.
- [ ] Run the CUDA smoke test.
- [ ] Tune the TensorRT concurrency matrix.
- [ ] Run the matching local MLX benchmark.
- [ ] Compare both architectures on the fixed data set.
- [ ] Resume and complete the 249-source Audiomovers sweep.
- [ ] Add measured results and conclusions to the README.
- [ ] Commit and push all approved code and compact results.
- [ ] Retrieve the remote summaries.
- [ ] Delete the Medium Linode and verify deletion.
