#!/bin/bash
# Tuning window: stop prod, run tune_cmp.py Part A (Marlin) on GPU 0 and Part B (GEMM) on GPU 1 in parallel, restore prod.
set -u
B=/home/user/qwen3nextflash/batchinv; LOG=$B/logs/dt_window4.log; PROD=qwen38-flash-next-pp3-int8all
IMAGE=qwen38-flash-next:ple-fp8-pp3-detmoe
exec > >(tee -a $LOG) 2>&1
ts() { date -u +%H:%M:%S; }
echo "=== DT WINDOW 4 (tuning) START $(date -u) ==="
docker ps --format '{{.Names}}' | grep -qx $PROD || { echo "prod not running"; exit 1; }
docker inspect $PROD > $B/logs/w4_prod_inspect_before.json
POLICY=$(docker inspect $PROD --format '{{.HostConfig.RestartPolicy.Name}}')
STOPPED=0; RESTORED=0
restore() {
  [ "$RESTORED" = 1 ] && return; RESTORED=1
  docker rm -f tune-a tune-b tune-sms >/dev/null 2>&1
  [ "$STOPPED" = 1 ] || return
  echo "[$(ts)] restore prod"; local since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  docker start $PROD >/dev/null; docker update --restart=$POLICY $PROD >/dev/null
  for i in $(seq 1 150); do docker logs --since $since $PROD 2>&1 | grep -q "Starting vLLM server" && break; sleep 10; done
  for i in $(seq 1 60); do curl -sf -o /dev/null localhost:8001/v1/models && break; sleep 5; done
  local s=$(curl -s localhost:8001/v1/models | python3 -c "import json,sys; d=json.load(sys.stdin)['data'][0]; print(d['id'], d.get('max_model_len'))")
  echo "[$(ts)] served: $s"
  if [ "$s" != "qwen38-flash-next-awq 524288" ]; then since=$(date -u +%Y-%m-%dT%H:%M:%SZ); docker restart $PROD >/dev/null
    for i in $(seq 1 150); do docker logs --since $since $PROD 2>&1 | grep -q "Starting vLLM server" && break; sleep 10; done
    for i in $(seq 1 60); do curl -sf -o /dev/null localhost:8001/v1/models && break; sleep 5; done
    echo "[$(ts)] served after warm restart: $(curl -s localhost:8001/v1/models | head -c 200)"; fi
  curl -s localhost:8001/v1/completions -H 'Content-Type: application/json' -d '{"model":"qwen38-flash-next-awq","prompt":"The capital of France is","max_tokens":4,"temperature":0}' | python3 -c "import json,sys; print('smoke:', json.load(sys.stdin)['choices'][0]['text'])"
  echo "=== DT WINDOW 4 prod restored $(date -u) ==="
}
trap restore EXIT
for i in $(seq 1 60); do busy=$(curl -s localhost:8001/metrics | awk '/^vllm:num_requests_(running|waiting)\{/ {s+=$2} END {print s+0}'); [ "$busy" = "0" ] && break; sleep 10; done
echo "[$(ts)] stopping prod"; docker update --restart=no $PROD >/dev/null; STOPPED=1; docker stop $PROD >/dev/null; sleep 5
docker run --rm --name tune-sms --runtime nvidia --gpus '"device=3"' --entrypoint python3 $IMAGE -c "import torch; p=torch.cuda.get_device_properties(0); print('GPU3', p.name, p.multi_processor_count, 'SMs')" 2>&1 | tail -1
docker run --rm --name tune-a --runtime nvidia --gpus '"device=0"' -e PART=A -v $B:/w -w /w/hook3 --entrypoint python3 $IMAGE /w/tune_cmp.py /w/results/tune_cmp_A.json > $B/logs/tune_cmp_A.log 2>&1 &
docker run --rm --name tune-b --runtime nvidia --gpus '"device=1"' -e PART=B -v $B:/w -w /w/hook3 --entrypoint python3 $IMAGE /w/tune_cmp.py /w/results/tune_cmp_B.json > $B/logs/tune_cmp_B.log 2>&1 &
wait
echo "[$(ts)] tuning done"; grep -E "^A (stock|best)" $B/logs/tune_cmp_A.log | cut -c1-300; grep -c "^B " $B/logs/tune_cmp_B.log
