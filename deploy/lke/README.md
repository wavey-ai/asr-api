# asr-api on wavey LKE

`asr-api` deploys into the shared Linode Kubernetes Engine cluster labeled `wavey-us-ord`.

Public naming:

- public API host: `asr.wavey.ai`

Internal naming:

- shared cluster: `wavey-us-ord`
- namespace: `asr-api`
- ingress image: `ghcr.io/wavey-ai/asr-api-ingress`
- decoder image: `ghcr.io/wavey-ai/asr-api-decoder`
- worker image: `ghcr.io/wavey-ai/asr-api-worker`
- ingress deployment: `asr-api-ingress`
- decoder deployment: `asr-api-decoder`
- worker deployment: `asr-api-worker`

## Files

- image build workflow: `.github/workflows/build-image.yml`
- deploy workflow: `.github/workflows/deploy-main.yml`
- Kubernetes manifests: `deploy/k8s/asr-api/`
- TensorRT overlay: `deploy/k8s/asr-api-trt/`
- Linode helper: `deploy/linode_api.py`
- ingress image build: `docker/asr-api-ingress.Dockerfile`
- decoder image build: `docker/asr-api-decoder.Dockerfile`
- worker image build: `docker/asr-api.Dockerfile`
- fallback source-build image: `docker/asr-api-source.Dockerfile`
- TensorRT worker image: `docker/asr-api-trt.Dockerfile`

## Required GitHub secrets

- `LINODE_TOKEN`: token that can read the shared LKE cluster
- `WAVEY_AI_GH_TOKEN`: classic PAT with package access that can clone private Wavey Git dependencies during Docker build and pull `ghcr.io/wavey-ai/*` images from the cluster

## Optional GitHub secrets

- `ASR_API_MODEL_TARBALL_URL`: HTTPS URL to a `.tar.gz` archive containing the split TDT ONNX files plus `tokens.txt` or `vocab.txt`

Current bucket-backed Parakeet archive:

- `https://us-iad-1.linodeobjects.com/wavey.ai/models/parakeet-tdt-0.6b-v3.tar.gz`

When `ASR_API_MODEL_TARBALL_URL` is set, each deploy run will sync that archive into the `asr-api-model` PVC before rolling the workload. If it is not set, you need to seed the PVC yourself before the pod can become healthy.

For split bundles such as Cohere, use the bucket manifest in `models/bucket-manifest.json`
plus `scripts/sync-model-from-bucket.sh` to seed the target host or PVC directly.

## Runtime layout

- namespace: `asr-api`
- public service: `asr-api-ingress`
- internal headless service: `asr-api-ingress-internal`
- public ingress host: `asr.wavey.ai`
- CPU ingress deployment: `asr-api-ingress`
- CPU decoder deployment: `asr-api-decoder`
- GPU worker deployment: `asr-api-worker`
- model PVC: `asr-api-model`
- mounted model dir: `/var/lib/asr-api/models/parakeet-tdt`

## Manual model seeding

If you do not want CI to sync the model archive, populate the PVC with the split TDT model files under:

```text
/var/lib/asr-api/models/parakeet-tdt
```

The deployment init container verifies:

- one encoder file: `encoder.fp16.onnx`, `encoder.onnx`, or `encoder.int8.onnx`
- one decoder file: `decoder.fp16.onnx`, `decoder.onnx`, or `decoder.int8.onnx`
- one joint encoder file: `joint.enc.fp16.onnx`, `joint.enc.onnx`, or `joint.enc.int8.onnx`
- one joint predictor file: `joint.pred.fp16.onnx`, `joint.pred.onnx`, or `joint.pred.int8.onnx`
- one joint net file: `joint.joint_net.fp16.onnx`, `joint.joint_net.onnx`, or `joint.joint_net.int8.onnx`
- `tokens.txt` or `vocab.txt`

## Notes

- All roles terminate TLS themselves on port `8443`.
- Public nginx ingress points to `asr-api-ingress` and disables request buffering so long uploads stream cleanly.
- `asr-api-ingress-internal` is headless so decoder and GPU worker pods can discover individual ingress pod IPs and use the internal `/_upload_response/...` cache API directly.
- Ingress stays HTTP-only and lands raw request bytes in `upload-response`.
- Audio decode / resample / downmix runs on CPU decoder pods.
- Featurization stays with ONNX decode on the GPU worker because `asr-torch` loads traced CUDA featurizer modules.
- The baseline manifests keep TensorRT disabled. Use `deploy/k8s/asr-api-trt/` with the TRT image once you want engine building on the GPU worker.
- The current LKE cluster has only one GPU node, so `asr-api-worker` competes directly with any other `nvidia.com/gpu: 1` deployment, especially `bitneedle-gpu-api`.
