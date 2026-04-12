#!/usr/bin/env bash
set -euo pipefail

: "${KUBECONFIG:?set KUBECONFIG to the target cluster kubeconfig}"
: "${LINODE_TOKEN:?set LINODE_TOKEN for Linode DNS updates}"
: "${REGISTRY_SERVER:=asr-registry.wavey.ai}"
: "${REGISTRY_USERNAME:=asr-api}"
: "${REGISTRY_PASSWORD:?set REGISTRY_PASSWORD for private image pulls}"

ASR_API_NAMESPACE="${ASR_API_NAMESPACE:-asr-api}"
ASR_API_DOMAIN="${ASR_API_DOMAIN:-asr.wavey.ai}"
ASR_API_KUSTOMIZE_PATH="${ASR_API_KUSTOMIZE_PATH:-deploy/k8s/asr-api}"
ASR_API_INGRESS_IMAGE="${ASR_API_INGRESS_IMAGE:-asr-registry.wavey.ai/asr-api-ingress:main}"
ASR_API_WORKER_IMAGE="${ASR_API_WORKER_IMAGE:-asr-registry.wavey.ai/asr-api-worker:main}"
ASR_API_MODEL_PVC="${ASR_API_MODEL_PVC:-asr-api-model}"
MODEL_TARBALL_URL="${MODEL_TARBALL_URL:-}"

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

openssl req -x509 -nodes -newkey rsa:2048 -sha256 \
  -keyout "$tmpdir/asr-api.key" \
  -out "$tmpdir/asr-api.crt" \
  -days 30 \
  -subj "/CN=asr-api" \
  -addext "subjectAltName=DNS:asr-api-ingress,DNS:asr-api-ingress.${ASR_API_NAMESPACE}.svc.cluster.local,DNS:asr-api-ingress-internal,DNS:asr-api-ingress-internal.${ASR_API_NAMESPACE}.svc.cluster.local" \
  >/dev/null 2>&1

openssl req -x509 -nodes -newkey rsa:2048 -sha256 \
  -keyout "$tmpdir/public.key" \
  -out "$tmpdir/public.crt" \
  -days 30 \
  -subj "/CN=${ASR_API_DOMAIN}" \
  -addext "subjectAltName=DNS:${ASR_API_DOMAIN}" \
  >/dev/null 2>&1

kubectl apply -f "${ASR_API_KUSTOMIZE_PATH}/namespace.yaml"

kubectl -n "$ASR_API_NAMESPACE" create secret docker-registry asr-registry \
  --docker-server="$REGISTRY_SERVER" \
  --docker-username="$REGISTRY_USERNAME" \
  --docker-password="$REGISTRY_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$ASR_API_NAMESPACE" create secret tls asr-api-tls \
  --cert="$tmpdir/asr-api.crt" \
  --key="$tmpdir/asr-api.key" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$ASR_API_NAMESPACE" create secret tls asr-wavey-ai-tls \
  --cert="$tmpdir/public.crt" \
  --key="$tmpdir/public.key" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f "${ASR_API_KUSTOMIZE_PATH}/pvc.yaml"

if [[ -n "$MODEL_TARBALL_URL" ]]; then
  if kubectl -n "$ASR_API_NAMESPACE" get deployment asr-api-worker >/dev/null 2>&1; then
    kubectl -n "$ASR_API_NAMESPACE" scale deployment/asr-api-worker --replicas=0
    kubectl -n "$ASR_API_NAMESPACE" wait \
      --for=delete pod \
      -l app.kubernetes.io/name=asr-api-worker \
      --timeout=10m || true
  fi

  kubectl -n "$ASR_API_NAMESPACE" delete job asr-api-model-sync --ignore-not-found=true --wait=true || true
  kubectl -n "$ASR_API_NAMESPACE" create secret generic asr-api-model-sync \
    --from-literal=model-url="$MODEL_TARBALL_URL" \
    --dry-run=client -o yaml | kubectl apply -f -

  cat <<'EOF' | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: asr-api-model-sync
  namespace: asr-api
spec:
  ttlSecondsAfterFinished: 300
  backoffLimit: 1
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: sync
          image: alpine:3.20
          command:
            - sh
            - -lc
            - |
              set -euo pipefail
              apk add --no-cache ca-certificates curl >/dev/null
              archive="/tmp/parakeet-model.tar.gz"
              scratch_dir="/tmp/parakeet-model"
              target_dir="/var/lib/asr-api/models/parakeet-tdt"
              rm -rf "$scratch_dir" "$target_dir"
              mkdir -p "$scratch_dir" "$target_dir"
              curl -fsSL "$MODEL_TARBALL_URL" -o "$archive"
              tar -xzf "$archive" -C "$scratch_dir"
              source_dir="$scratch_dir"
              if [ ! -f "$source_dir/encoder.onnx" ] && [ ! -f "$source_dir/encoder.fp16.onnx" ] && [ ! -f "$source_dir/encoder.int8.onnx" ]; then
                candidate="$(find "$scratch_dir" -mindepth 1 -maxdepth 4 -type f \( -name encoder.onnx -o -name encoder.fp16.onnx -o -name encoder.int8.onnx \) | head -n 1)"
                [ -n "$candidate" ] || exit 1
                source_dir="$(dirname "$candidate")"
              fi
              copy_first() {
                dest_dir="$1"
                shift
                for name in "$@"; do
                  if [ -f "$source_dir/$name" ]; then
                    cp "$source_dir/$name" "$dest_dir/"
                    if [ -f "$source_dir/$name.data" ]; then
                      cp "$source_dir/$name.data" "$dest_dir/"
                    fi
                    return 0
                  fi
                done
                return 1
              }
              copy_first "$target_dir" encoder.fp16.onnx encoder.onnx encoder.int8.onnx
              copy_first "$target_dir" decoder.fp16.onnx decoder.onnx decoder.int8.onnx
              copy_first "$target_dir" joint.enc.fp16.onnx joint.enc.onnx joint.enc.int8.onnx
              copy_first "$target_dir" joint.pred.fp16.onnx joint.pred.onnx joint.pred.int8.onnx
              copy_first "$target_dir" joint.joint_net.fp16.onnx joint.joint_net.onnx joint.joint_net.int8.onnx
              copy_first "$target_dir" tokens.txt vocab.txt
          env:
            - name: MODEL_TARBALL_URL
              valueFrom:
                secretKeyRef:
                  name: asr-api-model-sync
                  key: model-url
          volumeMounts:
            - name: model
              mountPath: /var/lib/asr-api/models
      volumes:
        - name: model
          persistentVolumeClaim:
            claimName: asr-api-model
EOF

  kubectl -n "$ASR_API_NAMESPACE" wait --for=condition=complete job/asr-api-model-sync --timeout=30m
  kubectl -n "$ASR_API_NAMESPACE" logs job/asr-api-model-sync
  kubectl -n "$ASR_API_NAMESPACE" delete secret asr-api-model-sync --ignore-not-found=true
fi

kubectl apply -k "$ASR_API_KUSTOMIZE_PATH"
kubectl -n "$ASR_API_NAMESPACE" set image deployment/asr-api-ingress ingress="$ASR_API_INGRESS_IMAGE"
kubectl -n "$ASR_API_NAMESPACE" set image deployment/asr-api-worker worker="$ASR_API_WORKER_IMAGE"
kubectl -n "$ASR_API_NAMESPACE" rollout restart deployment/asr-api-ingress
kubectl -n "$ASR_API_NAMESPACE" rollout restart deployment/asr-api-worker
kubectl -n "$ASR_API_NAMESPACE" rollout status deployment/asr-api-ingress --timeout=20m
kubectl -n "$ASR_API_NAMESPACE" rollout status deployment/asr-api-worker --timeout=20m

ingress_ip=""
for _ in $(seq 1 60); do
  ingress_ip="$(kubectl -n "$ASR_API_NAMESPACE" get ingress asr-api-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)"
  if [[ -z "$ingress_ip" ]]; then
    ingress_ip="$(kubectl -n ingress-nginx get svc ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)"
  fi
  if [[ -n "$ingress_ip" ]]; then
    break
  fi
  sleep 5
done

if [[ -z "$ingress_ip" ]]; then
  echo "failed to resolve ingress IP" >&2
  exit 1
fi

python3 deploy/linode_api.py upsert-domain-a-record \
  --domain wavey.ai \
  --name asr \
  --target "$ingress_ip" \
  --ttl-sec 30

kubectl -n "$ASR_API_NAMESPACE" get deploy,pods,svc,ingress
echo "asr-api deployed to ${ASR_API_DOMAIN} (${ingress_ip})"
