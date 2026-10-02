#!/bin/bash
# Window 7: compile-instance invariance. Two separately compiled containers of the same config differed (deployed prod vs
# E3: 99% of positions, 3.3% top-1 flips) while two boots sharing one compile cache were bit-identical -> torch.compile
# (Inductor) picks kernel configs by on-device benchmarking (reduction configs, combo kernels). Inductor's deterministic
# mode (TORCHINDUCTOR_DETERMINISTIC=1) skips numerics-affecting benchmarking; vLLM's benchmark_combo_kernel is turned off
# explicitly (combo kernels kept).
#   D1, D2  INT8+BF16 PLE (prod checkpoint), hook rev 2, deterministic Inductor, each with its OWN fresh compile cache
#           -> KLD corpus + 20x2048 probe must be bit-identical between D1 and D2 (compile-instance invariance)
#   Q4d     INT4 + BF16 PLE, same mode, fresh compile -> KLD vs D1
#   FP8d    INT8 + FP8 PLE, same mode, fresh compile -> KLD vs D1
# prod (rev 2, image -inv) is restored at the end in all cases (trap).
set -u
B=/home/user/qwen3nextflash/batchinv
KLD=/home/user/qwen3nextflash/kld
LOG=$B/logs/dt_window7.log
PROD=qwen38-flash-next-pp3-int8all
IMG=qwen38-flash-next:ple-fp8-pp3-detmoe-inv
NAME=plefp8-det; URL=http://127.0.0.1:8002
BASE_INSPECT=$B/logs/w6_prod_inspect_before.json     # stock prod's config (serve args, env, GPUs, volumes)
HUB_INT4=hub/models--cyankiwi--Qwen3.8-Flash-Next-AWQ-INT4/snapshots/d39638a0e740fccb3e24ae0ea5cab34c15371ae6
CC='{"inductor_compile_config":{"combo_kernels":true,"benchmark_combo_kernel":false}}'
CC_FALLBACK='{"inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false}}'
mkdir -p $B/logs $B/results/kld5
exec > >(tee -a $LOG) 2>&1
ts() { date -u +%H:%M:%S; }
echo "=== DT WINDOW 7 START $(date -u) ==="
wait_ready() { # $1 container, $2 max s, $3 since
  for i in $(seq 1 $(($2 / 10))); do
    if docker logs --since $3 $1 2>&1 | grep -q "Starting vLLM server"; then echo "[$(ts)] $1 ready after $((i * 10))s"; sleep 5; return 0; fi
    if ! docker ps -q -f name=^$1$ | grep -q .; then echo "[$(ts)] $1 exited"; docker logs --since $3 $1 2>&1 | grep -E "Error|Traceback|hook3|deterministic" | tail -40; return 1; fi
    sleep 10
  done
  echo "[$(ts)] $1 not ready after $2 s"; docker logs --since $3 $1 2>&1 | tail -30; return 1
}
served() { curl -s localhost:$1/v1/models | python3 -c "import json,sys; d=json.load(sys.stdin)['data'][0]; print(d['id'], d.get('max_model_len'))" 2>/dev/null; }
smoke() { curl -s localhost:$1/v1/completions -H 'Content-Type: application/json' -d '{"model":"qwen38-flash-next-awq","prompt":"The capital of France is","max_tokens":4,"temperature":0}' | python3 -c "import json,sys; print('smoke:', repr(json.load(sys.stdin)['choices'][0]['text']))"; }
RESTORED=0; STOPPED=0
restore_prod() {
  [ "$RESTORED" = 1 ] && return 0; RESTORED=1
  docker rm -f $NAME >/dev/null 2>&1
  [ "$STOPPED" = 1 ] || { echo "[$(ts)] prod was never stopped"; return 0; }
  echo "[$(ts)] --- restore prod ($PROD, rev 2) ---"
  local since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  docker start $PROD >/dev/null; docker update --restart=always $PROD >/dev/null
  wait_ready $PROD 1500 $since
  for i in $(seq 1 60); do curl -sf -o /dev/null localhost:8001/v1/models && break; sleep 5; done
  local s=$(served 8001); echo "[$(ts)] served: $s"
  if [ "$s" != "qwen38-flash-next-awq 524288" ]; then
    since=$(date -u +%Y-%m-%dT%H:%M:%SZ); docker restart $PROD >/dev/null; wait_ready $PROD 1500 $since
    for i in $(seq 1 60); do curl -sf -o /dev/null localhost:8001/v1/models && break; sleep 5; done
    echo "[$(ts)] served: $(served 8001)"
  fi
  smoke 8001
  echo "=== DT WINDOW 7: prod restored $(date -u) ==="
}
trap 'restore_prod' EXIT
trap 'echo "[$(ts)] trapped signal"; exit 1' INT TERM
abort() { echo "[$(ts)] ABORT: $1"; exit 1; }
stop_requested() { [ -e $B/STOP ] && { echo "[$(ts)] STOP file found"; return 0; }; return 1; }
boot() { # $1 tag, $2 cache volume, $3 compilation-config json, $4.. extra dt5_run_cmd flags
  local tag=$1 vol=$2 cc=$3; shift 3
  docker rm -f $NAME >/dev/null 2>&1
  echo "[$(ts)] host RAM available: $(free -g | awk '/Mem:/ {print $7}') GB"
  local since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[$(ts)] === boot $tag (volume $vol, $cc, $*) ==="
  python3 $B/dt5_run_cmd.py $BASE_INSPECT $NAME 8002 --image-hook --image $IMG --cache-volume $vol --env TORCHINDUCTOR_DETERMINISTIC=1 \
    --arg=--compilation-config --arg="$cc" --arg=--max-logprobs --arg=200 "$@" > $B/logs/run7_$tag.txt 2>&1 || { cut -c1-600 $B/logs/run7_$tag.txt; return 1; }
  head -1 $B/logs/run7_$tag.txt | cut -c1-400
  wait_ready $NAME 2700 $since || return 1
  docker logs --since $since $NAME > $B/logs/boot7_$tag.log 2>&1
  grep -h "hook3\|Auto-fit max_model_len\|GPU KV cache size\|inductor_compile_config" $B/logs/boot7_$tag.log | sed 's/^.*INFO //; s/^([^)]*) //' | grep -o "hook3.*\|Auto-fit.*\|GPU KV.*\|'inductor_compile_config': {[^}]*}" | cut -c1-200 | sort | uniq -c | head -20
  for i in $(seq 1 30); do curl -sf -o /dev/null $URL/v1/models && break; sleep 2; done
  echo "[$(ts)] $tag served: $(served 8002)"; smoke 8002
}
fresh_volume() { docker volume rm $1 >/dev/null 2>&1; docker volume create $1 >/dev/null; }
kld_score() { # $1 out name
  echo "[$(ts)] KLD scoring $1 (77 windows, K=200)"
  python3 $KLD/score.py collect --url $URL/v1 --model qwen38-flash-next-awq --k 200 --out $B/results/kld5/$1.npz --windows $KLD/corpus/windows.jsonl 2>&1 | grep -E "77/77|wrote|retry|Error|Traceback" | tail -5
  [ -s $B/results/kld5/$1.npz ] && echo "[$(ts)] KLD $1 done" || { echo "[$(ts)] KLD $1 FAILED"; return 1; }
}
same() { # $1 $2 npz paths -> prints bit-identical
  python3 -c "
import numpy as np, sys
a, b = np.load('$1'), np.load('$2')
print('bit-identical:', bool(np.array_equal(a['logprobs'], b['logprobs']) and np.array_equal(a['ids'], b['ids'])), '| positions differing:', round(float((a['logprobs'] != b['logprobs']).any(-1).mean()), 4))"
}

# ---------------- Step 0 ----------------
echo "[$(ts)] --- Step 0: preconditions ---"
docker ps --format '{{.Names}}' | grep -qx $PROD || abort "prod not running"
[ "$(docker inspect $PROD --format '{{.Config.Image}}')" = "$IMG" ] || abort "prod is not the rev-2 image"
[ -s $BASE_INSPECT ] || abort "base inspect missing"
python3 $B/dt5_run_cmd.py $BASE_INSPECT $NAME 8002 --image-hook --image $IMG --cache-volume x --env TORCHINDUCTOR_DETERMINISTIC=1 --arg=--compilation-config --arg="$CC" --dry-run >/dev/null || abort "dry run"
rm -f $B/results/kld5/REF_d1.npz $B/results/kld5/REF_d2.npz $B/results/kld5/Q4_d.npz $B/results/kld5/FP8_d.npz $B/results/s_D1_* $B/results/s_D2_*
fresh_volume vllm-cache-det-a; fresh_volume vllm-cache-det-b; fresh_volume vllm-cache-det-c
for i in $(seq 1 60); do
  busy=$(curl -s localhost:8001/metrics | awk '/^vllm:num_requests_(running|waiting)\{/ {s+=$2} END {print s+0}')
  [ "$busy" = "0" ] && break; echo "prod busy ($busy), waiting"; sleep 10
done
echo "[$(ts)] --- stopping prod (downtime starts) ---"
docker update --restart=no $PROD >/dev/null; STOPPED=1; docker stop $PROD >/dev/null; sleep 5

# ---------------- Step 1: D1 ----------------
USE_CC="$CC"
if ! boot D1 vllm-cache-det-a "$USE_CC"; then
  echo "[$(ts)] D1 failed with combo kernels on; retrying with combo kernels off"
  fresh_volume vllm-cache-det-a; USE_CC="$CC_FALLBACK"
  boot D1 vllm-cache-det-a "$USE_CC" || abort "D1 boot failed"
fi
python3 $B/dt5_prefill.py $URL D1
python3 $B/dt3_score.py $URL D1_L2048 2048 20 || abort "probe D1"
kld_score REF_d1 || abort "REF_d1"
python3 $B/dt3_bench.py D1_a $URL; python3 $B/dt3_bench.py D1_b $URL

# ---------------- Step 2: D2 (fresh compile, new process) ----------------
stop_requested && exit 0
boot D2 vllm-cache-det-b "$USE_CC" || abort "D2 boot failed"
python3 $B/dt3_score.py $URL D2_L2048 2048 20 || abort "probe D2"
echo -n "[$(ts)] probe D1 vs D2 (separate compiles): "; same $B/results/s_D1_L2048.npz $B/results/s_D2_L2048.npz
kld_score REF_d2
echo -n "[$(ts)] KLD corpus D1 vs D2 (separate compiles): "; same $B/results/kld5/REF_d1.npz $B/results/kld5/REF_d2.npz
echo -n "[$(ts)] probe deployed rev-2 prod (non-deterministic compile) vs D1: "; same $B/results/s_P_L2048.npz $B/results/s_D1_L2048.npz

# ---------------- Step 3: INT4 ----------------
stop_requested && exit 0
if boot Q4d vllm-cache-det-c "$USE_CC" --ckpt $HUB_INT4; then
  kld_score Q4_d; python3 $B/dt3_bench.py Q4d $URL
else echo "[$(ts)] Q4d boot failed, skipping"; fi
docker rm -f $NAME >/dev/null

# ---------------- Step 4: FP8 PLE ----------------
stop_requested && exit 0
fresh_volume vllm-cache-det-c
if boot FP8d vllm-cache-det-c "$USE_CC" --ckpt ple-fp8-int8all-38f --env VLLM_PLE_FP8_TABLE=1; then
  kld_score FP8_d; python3 $B/dt3_bench.py FP8d $URL
else echo "[$(ts)] FP8d boot failed, skipping"; fi
docker rm -f $NAME >/dev/null
echo "[$(ts)] compilation config used: $USE_CC"
echo "[$(ts)] --- all steps done, restoring prod ---"
