# transcriber on wavey LKE

`transcriber` deploys into the shared Linode Kubernetes Engine cluster labeled `wavey`.

Public naming:

- public API host: `transcribe.wavey.ai`

Internal naming:

- shared cluster: `wavey`
- namespace: `transcriber`
- image: `ghcr.io/wavey-ai/transcriber`

## Files

- image build workflow: `.github/workflows/build-image.yml`
- deploy workflow: `.github/workflows/deploy-main.yml`
- Kubernetes manifests: `deploy/k8s/transcriber/`
- Linode helper: `deploy/linode_api.py`
- image build: `docker/transcriber.Dockerfile`

## Required GitHub secrets

- `LINODE_TOKEN`: token that can read the shared LKE cluster

## Optional GitHub secrets

- `TRANSCRIBER_MODEL_TARBALL_URL`: HTTPS URL to a `.tar.gz` archive containing the split TDT ONNX files plus `tokens.txt` or `vocab.txt`

When `TRANSCRIBER_MODEL_TARBALL_URL` is set, each deploy run will sync that archive into the `transcriber-model` PVC before rolling the workload. If it is not set, you need to seed the PVC yourself before the pod can become healthy.

## Runtime layout

- namespace: `transcriber`
- service: `transcriber`
- ingress host: `transcribe.wavey.ai`
- model PVC: `transcriber-model`
- mounted model dir: `/var/lib/transcriber/models/parakeet-tdt`

## Manual model seeding

If you do not want CI to sync the model archive, populate the PVC with the split TDT model files under:

```text
/var/lib/transcriber/models/parakeet-tdt
```

The deployment init container verifies:

- one encoder file: `encoder.fp16.onnx`, `encoder.onnx`, or `encoder.int8.onnx`
- one decoder file: `decoder.fp16.onnx`, `decoder.onnx`, or `decoder.int8.onnx`
- one joint encoder file: `joint.enc.fp16.onnx`, `joint.enc.onnx`, or `joint.enc.int8.onnx`
- one joint predictor file: `joint.pred.fp16.onnx`, `joint.pred.onnx`, or `joint.pred.int8.onnx`
- one joint net file: `joint.joint_net.fp16.onnx`, `joint.joint_net.onnx`, or `joint.joint_net.int8.onnx`
- `tokens.txt` or `vocab.txt`

## Notes

- The service terminates TLS itself on port `8443`.
- The ingress uses HTTPS to the backend and disables request buffering so long uploads can stream through cleanly.
- The service is GPU-oriented now. The repo's Kubernetes config matches the split ONNX model layout, but the container image still needs a CUDA/libtorch-aware runtime before LKE deploys will be usable end to end.
