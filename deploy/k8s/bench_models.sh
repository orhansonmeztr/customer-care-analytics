#!/usr/bin/env bash
# Benchmark several open models for the analytics assistant on the single GPU node.
#
# One GPU means one model at a time: the script pauses the main deployment, then for each
# model starts a vLLM deployment, waits for it, runs the golden-set evaluation through a
# port-forward, removes it, and finally brings the main deployment back.
#
#   deploy/k8s/bench_models.sh            # run from the repository root, default models
#   deploy/k8s/bench_models.sh "name|hf-id|--extra-arg|null|2048"   # or the given entries
#
# Each entry: served-name | Hugging Face id | extra vLLM args | request extra_body | max tokens
set -uo pipefail

NS=llm
MAIN=vllm-qwen-coder-14b
PORT=18002
LOG_DIR=${CARE_DATA_DIR:-data}/bench
mkdir -p "$LOG_DIR"

NO_THINK='{"chat_template_kwargs":{"enable_thinking":false}}'
MODELS=(
  "arctic-text2sql-r1-7b|Snowflake/Arctic-Text2SQL-R1-7B|--quantization=fp8|null|4096"
  "qwen3-8b|Qwen/Qwen3-8B|--quantization=fp8|${NO_THINK}|1024"
  "distil-qwen3-4b-text2sql|distil-labs/distil-qwen3-4b-text2sql||${NO_THINK}|1024"
)
[ $# -gt 0 ] && MODELS=("$@")

manifest() {  # served model extra_args
  local extra=""
  [ -n "$3" ] && extra="            - $3"
  cat <<EOF
apiVersion: apps/v1
kind: Deployment
metadata: {name: vllm-bench, namespace: $NS, labels: {app: vllm-bench}}
spec:
  replicas: 1
  strategy: {type: Recreate}
  selector: {matchLabels: {app: vllm-bench}}
  template:
    metadata: {labels: {app: vllm-bench}}
    spec:
      runtimeClassName: nvidia
      nodeSelector: {nvidia.com/gpu.present: "true"}
      tolerations: [{key: nvidia.com/gpu, operator: Equal, value: "true", effect: NoSchedule}]
      enableServiceLinks: false
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.34.0
          args:
            - --model=$2
            - --served-model-name=$1
            - --max-model-len=8192
            - --gpu-memory-utilization=0.92
            - --max-num-seqs=8
$extra
          env: [{name: HF_HUB_OFFLINE, value: "0"}]
          ports: [{name: http, containerPort: 8000}]
          resources:
            requests: {cpu: "2", memory: 8Gi, nvidia.com/gpu: "1"}
            limits: {memory: 28Gi, nvidia.com/gpu: "1"}
          startupProbe: {httpGet: {path: /health, port: http}, periodSeconds: 10, failureThreshold: 240}
          readinessProbe: {httpGet: {path: /health, port: http}, periodSeconds: 10}
          volumeMounts:
            - {name: hf-cache, mountPath: /root/.cache/huggingface}
            - {name: shm, mountPath: /dev/shm}
      volumes:
        - {name: hf-cache, hostPath: {path: /home/orhan/.cache/huggingface, type: Directory}}
        - {name: shm, emptyDir: {medium: Memory, sizeLimit: 4Gi}}
EOF
}

stop_bench() {
  kubectl -n $NS delete deploy vllm-bench --ignore-not-found --wait=true >/dev/null
  kubectl -n $NS wait --for=delete pod -l app=vllm-bench --timeout=300s >/dev/null 2>&1 || true
}

echo "pausing $MAIN"
kubectl -n $NS scale deploy $MAIN --replicas=0
kubectl -n $NS wait --for=delete pod -l app=vllm-coder --timeout=300s >/dev/null 2>&1 || true

for entry in "${MODELS[@]}"; do
  IFS='|' read -r served model extra body maxtok <<<"$entry"
  echo "=== $served ($model) $(date -u +%H:%M:%S)"
  stop_bench
  manifest "$served" "$model" "$extra" | kubectl apply -f - >/dev/null
  start=$(date +%s)
  if ! kubectl -n $NS wait --for=condition=ready pod -l app=vllm-bench --timeout=2400s; then
    echo "!!! $served did not become ready"
    kubectl -n $NS logs deploy/vllm-bench --tail=60 > "$LOG_DIR/$served.server.log" 2>&1
    continue
  fi
  echo "ready after $(( $(date +%s) - start )) s"
  kubectl -n $NS logs deploy/vllm-bench > "$LOG_DIR/$served.server.log" 2>&1
  grep -E "Model loading took|GPU KV cache size" "$LOG_DIR/$served.server.log" | cut -c1-200

  pod=$(kubectl -n $NS get pod -l app=vllm-bench -o jsonpath='{.items[0].metadata.name}')
  kubectl -n $NS port-forward "pod/$pod" $PORT:8000 >/dev/null 2>&1 &
  pf=$!
  for _ in $(seq 1 20); do curl -sf localhost:$PORT/v1/models >/dev/null && break; sleep 1; done
  ASSISTANT_BASE_URL=http://localhost:$PORT/v1 ASSISTANT_MODEL=$served ASSISTANT_API_KEY=none \
    ASSISTANT_EXTRA_BODY="$body" ASSISTANT_MAX_TOKENS=$maxtok \
    .venv/bin/care eval-assistant --provider openai --data-dir "${CARE_DATA_DIR:-data}" \
    > "$LOG_DIR/$served.eval.log" 2>&1
  grep -E '"passed"|answerable|safety_pass|p50' "$LOG_DIR/$served.eval.log"
  kill $pf 2>/dev/null
done

stop_bench
echo "restoring $MAIN"
kubectl -n $NS scale deploy $MAIN --replicas=1
kubectl -n $NS wait --for=condition=ready pod -l app=vllm-coder --timeout=1200s
