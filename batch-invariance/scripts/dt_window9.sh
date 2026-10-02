#!/bin/bash
# Window 9: deploy the batch- AND instance-invariant config as prod (supersedes window 6's rev-2 deploy).
#   new image  qwen38-flash-next:ple-fp8-pp3-detmoe-inv  (= prod image + batchinv/hook3 baked in, flags in ENV)
#   new prod   container qwen38-flash-next-pp3-int8all, same serve args/env/GPUs/volumes/port 8001/restart always,
#              torch.compile cache on volume vllm-cache-inv (the patched GEMM changes the traced graph)
#   rollback   the old container is kept, stopped (restart=no), as qwen38-flash-next-pp3-int8all-stock; on any failure
#              the new container is removed and the old one is renamed back and started.
# Validation: context (warm restart once if short; YaRN factor adjusted if still short), hook banners, smoke, probe
# bit-identical to the validated E3 run (20 windows x 2048, K=20), decode reproducibility under concurrency, bench, prefill.
set -u
B=/home/user/qwen3nextflash/batchinv
LOG=$B/logs/dt_window9.log
PROD=qwen38-flash-next-pp3-int8all; OLD=qwen38-flash-next-pp3-int8all-rev2
NEWIMG=qwen38-flash-next:ple-fp8-pp3-detmoe-inv-det
CC='{"inductor_compile_config":{"combo_kernels":true,"benchmark_combo_kernel":false}}'
VOL=vllm-cache-prod-det
INSPECT=$B/logs/w6_prod_inspect_before.json
URL=http://127.0.0.1:8001
exec > >(tee -a $LOG) 2>&1
ts() { date -u +%H:%M:%S; }
echo "=== DT WINDOW 9 (deploy) START $(date -u) ==="
wait_ready() { # $1 container, $2 max s, $3 since
  for i in $(seq 1 $(($2 / 10))); do
    if docker logs --since $3 $1 2>&1 | grep -q "Starting vLLM server"; then echo "[$(ts)] $1 ready after $((i * 10))s"; sleep 5; return 0; fi
    if ! docker ps -q -f name=^$1$ | grep -q .; then echo "[$(ts)] $1 exited"; docker logs --since $3 $1 2>&1 | grep -E "Error|Traceback|hook3" | tail -40; return 1; fi
    sleep 10
  done
  echo "[$(ts)] $1 not ready after $2 s"; docker logs --since $3 $1 2>&1 | tail -30; return 1
}
served() { curl -s localhost:8001/v1/models | python3 -c "import json,sys; d=json.load(sys.stdin)['data'][0]; print(d['id'], d.get('max_model_len'))" 2>/dev/null; }
smoke() { curl -s localhost:8001/v1/completions -H 'Content-Type: application/json' -d '{"model":"qwen38-flash-next-awq","prompt":"The capital of France is","max_tokens":4,"temperature":0}' | python3 -c "import json,sys; print('smoke:', repr(json.load(sys.stdin)['choices'][0]['text']))"; }
DONE=0; SWAPPED=0
rollback() {
  [ "$DONE" = 1 ] && return 0
  [ "$SWAPPED" = 1 ] || { echo "[$(ts)] nothing swapped; prod untouched"; docker start $PROD >/dev/null 2>&1; return 0; }
  echo "[$(ts)] --- ROLLBACK to the stock container ---"
  docker rm -f $PROD >/dev/null 2>&1
  docker rename $OLD $PROD; docker update --restart=always $PROD >/dev/null
  local since=$(date -u +%Y-%m-%dT%H:%M:%SZ); docker start $PROD >/dev/null; wait_ready $PROD 1500 $since
  for i in $(seq 1 60); do curl -sf -o /dev/null localhost:8001/v1/models && break; sleep 5; done
  local s=$(served); echo "[$(ts)] served: $s"
  if [ "$s" != "qwen38-flash-next-awq 524288" ]; then since=$(date -u +%Y-%m-%dT%H:%M:%SZ); docker restart $PROD >/dev/null; wait_ready $PROD 1500 $since
    for i in $(seq 1 60); do curl -sf -o /dev/null localhost:8001/v1/models && break; sleep 5; done; echo "[$(ts)] served: $(served)"; fi
  smoke; echo "=== DT WINDOW 9: ROLLED BACK $(date -u) ==="
}
trap rollback EXIT
trap 'echo "[$(ts)] trapped signal"; exit 1' INT TERM
abort() { echo "[$(ts)] ABORT: $1"; exit 1; }
new_prod() { # extra dt5_run_cmd args
  python3 $B/dt5_run_cmd.py $INSPECT $PROD 8001 --image-hook --image $NEWIMG --cache-volume $VOL --restart always --arg=--compilation-config --arg="$CC" "$@" > $B/logs/run9.txt 2>&1 || { cut -c1-600 $B/logs/run9.txt; return 1; }
  head -1 $B/logs/run9.txt | cut -c1-2500
}
boot_wait() { # $1 since
  wait_ready $PROD 2700 $1 || return 1
  for i in $(seq 1 60); do curl -sf -o /dev/null localhost:8001/v1/models && break; sleep 5; done
  docker logs --since $1 $PROD 2>&1 | grep -h "hook3\|Auto-fit max_model_len\|GPU KV cache size" | sed 's/^.*INFO //; s/^([^)]*) //' | cut -c1-200 | sort | uniq -c | head
  echo "[$(ts)] served: $(served)"; smoke
}

# ---------------- Step 0: image + preconditions (prod still up) ----------------
echo "[$(ts)] --- Step 0: build image, preconditions ---"
docker ps --format '{{.Names}}' | grep -qx $PROD || abort "prod not running"
docker ps -a --format '{{.Names}}' | grep -qx $OLD && abort "$OLD already exists"
docker inspect $PROD > $B/logs/w9_prod_inspect_before.json
[ "$(docker inspect $PROD --format '{{.Config.Image}}')" = "qwen38-flash-next:ple-fp8-pp3-detmoe-inv" ] || abort "prod is not the rev-2 image"
docker build -q -t $NEWIMG $B/deploy_det || abort "image build"
docker run --rm --entrypoint python3 $NEWIMG -c "import os, sitecustomize; print('hook baked:', sitecustomize.__file__, {k: v for k, v in os.environ.items() if k.startswith(('PLEFP8_', 'TORCHINDUCTOR_', 'VLLM_TRITON'))})" 2>&1 | grep "hook baked" || abort "image check"
python3 $B/dt5_run_cmd.py $INSPECT $PROD 8001 --image-hook --image $NEWIMG --cache-volume $VOL --restart always --arg=--compilation-config --arg="$CC" --dry-run > /dev/null || abort "dry run"
docker volume rm $VOL >/dev/null 2>&1; docker volume create $VOL >/dev/null
for i in $(seq 1 60); do
  busy=$(curl -s localhost:8001/metrics | awk '/^vllm:num_requests_(running|waiting)\{/ {s+=$2} END {print s+0}')
  [ "$busy" = "0" ] && break; echo "prod busy ($busy), waiting"; sleep 10
done

# ---------------- Step 1: swap ----------------
echo "[$(ts)] --- Step 1: stop rev-2 prod, keep it as $OLD, start the deterministic prod ---"
docker update --restart=no $PROD >/dev/null; docker stop $PROD >/dev/null; docker rename $PROD $OLD; SWAPPED=1
sleep 5
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
new_prod || abort "docker run"
boot_wait $since || abort "boot"
CTX=$(served | awk '{print $2}')
if [ "$CTX" != "524288" ]; then
  echo "[$(ts)] context $CTX after first boot -> warm restart"
  since=$(date -u +%Y-%m-%dT%H:%M:%SZ); docker restart $PROD >/dev/null; boot_wait $since || abort "warm restart"
  CTX=$(served | awk '{print $2}')
fi
if [ "$CTX" != "524288" ]; then
  # owner: a lower context is fine if YaRN is adjusted: factor = context / original_max_position_embeddings (262144)
  CTX=$(( CTX / 1600 * 1600 )); F=$(python3 -c "print(round($CTX / 262144, 6))")
  HFO=$(python3 -c "import json; print(json.dumps({'text_config': {'rope_parameters': {'mrope_interleaved': True, 'mrope_section': [11, 11, 10], 'rope_type': 'yarn', 'rope_theta': 10000000, 'partial_rotary_factor': 0.25, 'factor': $F, 'original_max_position_embeddings': 262144}}}, separators=(',', ':')))")
  echo "[$(ts)] context still short: re-create with --max-model-len $CTX and YaRN factor $F"
  docker rm -f $PROD >/dev/null; since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  new_prod --set-arg --max-model-len $CTX --set-arg --hf-overrides "$HFO" || abort "docker run (yarn)"
  boot_wait $since || abort "boot (yarn)"
fi
docker logs $PROD 2>&1 | grep -q "FAILED" && abort "hook patch FAILED in prod"
for pat in "invariant Marlin MoE active" "invariant Triton GEMM in use" "QSA pinned split profile in use" "non-final prefill chunks aligned"; do
  docker logs $PROD 2>&1 | grep -q "$pat" && echo "[$(ts)] prod: '$pat' seen" || abort "prod: '$pat' not seen"
done

# ---------------- Step 2: validation ----------------
echo "[$(ts)] --- Step 2: validation ---"
python3 $B/dt5_prefill.py $URL PROD_det
rm -f $B/results/s_P2_L*.npz $B/results/prefix_P2.json $B/results/decode_P2.json
for L in 2048 2047 2040 2001 1985 1984 1983 1600 1100; do python3 $B/dt3_score.py $URL P2_L$L $L 20 || abort "probe scoring L=$L"; done
for pat in "mamba split aligned" "QSA deterministic token selection in use" "triton-autotune-disabled"; do
  docker logs $PROD 2>&1 | grep -q "$pat" && echo "[$(ts)] prod: '$pat' seen" || abort "prod: '$pat' not seen"
done
python3 - <<EOF
import numpy as np
a, b = np.load("$B/results/s_D3_L2048.npz"), np.load("$B/results/s_P2_L2048.npz")
print("deployed prod vs D3 (window 8; separate container, separate compile; 20 windows x 2048, K=20) bit-identical:",
      bool(np.array_equal(a["logprobs"], b["logprobs"]) and np.array_equal(a["ids"], b["ids"])))
EOF
python3 $B/dt5_analyze.py P2 | head -20
python3 $B/dt5_prefix.py $URL P2 || echo "[$(ts)] prefix test failed"
python3 $B/dt3_decode.py $URL P2 || echo "[$(ts)] decode test failed"
python3 $B/dt5_analyze.py P2 | tail -6
python3 $B/dt3_bench.py P2_a $URL; python3 $B/dt3_bench.py P2_b $URL
TOKS=$(python3 -c "import json; print(max(json.load(open('$B/results/bench_P2_%s.json' % t))['res']['tok_per_s'] for t in 'ab'))" 2>/dev/null || echo 0)
python3 -c "import sys; sys.exit(0 if float('$TOKS') >= 0.9 * 72.5 else 1)" || abort "decode speed $TOKS tok/s is more than 10% below stock (72.5)"
smoke
docker update --restart=always $PROD >/dev/null
echo "[$(ts)] prod: $(docker inspect $PROD --format '{{.Config.Image}} restart={{.HostConfig.RestartPolicy.Name}}') served: $(served)"
echo "[$(ts)] rollback container: $(docker inspect $OLD --format '{{.Name}} {{.State.Status}} restart={{.HostConfig.RestartPolicy.Name}}')"
DONE=1
echo "=== DT WINDOW 9: deterministic invariant prod deployed $(date -u) ==="
