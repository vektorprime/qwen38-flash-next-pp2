#!/bin/bash
# Window 5 (owner-approved downtime, 2026-10-02): validate the complete batch-invariance fix on prod's stack, then KLD.
#   0  preconditions; stop prod
#   1  CMP 170HX micro-steps (parallel): split-K tile tuning (GPU 0/1) -> hook3/tuned_gemm.json; INT4 Marlin config check (GPU 3)
#   2  E3 = prod (INT8 experts + BF16 PLE) + all fixes (MoE/GEMM/QSA + chunk alignment 64), fresh compile cache volume,
#      --max-logprobs 200; warm restart once (context)
#   3  validation on E3: prompt lengths (ALL positions), prefix-cache/chunk repro, decode repro under concurrency, bench, prefill
#   4  KLD REF_a (77 windows, K=200) on E3; 5: restart E3 (new process) -> REF_b + cross-boot probe
#   6  Q4  = INT4 experts (original AWQ hub checkpoint, BF16 PLE) + fixes -> KLD Q4
#   7  FP8 = INT8 experts + FP8 PLE table + fixes -> KLD FP8
#   8  restore prod exactly (trap does this on any exit); offline analysis
# touch batchinv/STOP to skip remaining steps and restore early.
set -u
B=/home/user/qwen3nextflash/batchinv
KLD=/home/user/qwen3nextflash/kld
LOG=$B/logs/dt_window5.log
PROD=${PROD:-qwen38-flash-next-pp3-int8all}
IMAGE=qwen38-flash-next:ple-fp8-pp3-detmoe
NAME=plefp8-e3; URL=http://127.0.0.1:8002
VOL=vllm-cache-inv
INSPECT=$B/logs/w5_prod_inspect_before.json
HUB_INT4=hub/models--cyankiwi--Qwen3.8-Flash-Next-AWQ-INT4/snapshots/d39638a0e740fccb3e24ae0ea5cab34c15371ae6
mkdir -p $B/logs $B/results/kld5
exec > >(tee -a $LOG) 2>&1
ts() { date -u +%H:%M:%S; }
echo "=== DT WINDOW 5 START $(date -u) ==="

wait_ready() { # $1 container, $2 max s, $3 since
  for i in $(seq 1 $(($2 / 10))); do
    if docker logs --since $3 $1 2>&1 | grep -q "Starting vLLM server"; then echo "[$(ts)] $1 ready after $((i * 10))s"; sleep 5; return 0; fi
    if ! docker ps -q -f name=^$1$ | grep -q .; then echo "[$(ts)] $1 exited"; docker logs --since $3 $1 2>&1 | grep -E "Error|Traceback|hook3" | tail -40; return 1; fi
    sleep 10
  done
  echo "[$(ts)] $1 not ready after $2 s"; docker logs --since $3 $1 2>&1 | tail -30; return 1
}
served() { curl -s localhost:$1/v1/models | python3 -c "import json,sys; d=json.load(sys.stdin)['data'][0]; print(d['id'], d.get('max_model_len'))" 2>/dev/null; }
smoke() { curl -s localhost:$1/v1/completions -H 'Content-Type: application/json' -d '{"model":"qwen38-flash-next-awq","prompt":"The capital of France is","max_tokens":4,"temperature":0}' | python3 -c "import json,sys; print('smoke:', repr(json.load(sys.stdin)['choices'][0]['text']))"; }
RESTORED=0; STOPPED=0
restore_prod() {
  [ "$RESTORED" = 1 ] && return 0; RESTORED=1
  docker rm -f $NAME tune-a tune-b tune-i4 >/dev/null 2>&1
  [ "$STOPPED" = 1 ] || { echo "[$(ts)] prod was never stopped"; return 0; }
  echo "[$(ts)] --- restore prod ($PROD, restart policy ${POLICY:-always}) ---"
  local since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  docker start $PROD >/dev/null; docker update --restart=${POLICY:-always} $PROD >/dev/null
  wait_ready $PROD 1500 $since
  for i in $(seq 1 60); do curl -sf -o /dev/null localhost:8001/v1/models && break; sleep 5; done
  local s=$(served 8001); echo "[$(ts)] served: $s"
  if [ "$s" != "qwen38-flash-next-awq 524288" ]; then
    echo "[$(ts)] context short after restore -> one warm restart"
    since=$(date -u +%Y-%m-%dT%H:%M:%SZ); docker restart $PROD >/dev/null; wait_ready $PROD 1500 $since
    for i in $(seq 1 60); do curl -sf -o /dev/null localhost:8001/v1/models && break; sleep 5; done
    echo "[$(ts)] served: $(served 8001)"
  fi
  smoke 8001
  echo "=== DT WINDOW 5: prod restored $(date -u) ==="
  python3 $B/dt5_analyze.py E3 E3b 2>&1 | tail -40
  python3 $B/dt5_kld_compare.py 2>&1 | tail -20
}
trap 'restore_prod' EXIT
trap 'echo "[$(ts)] trapped signal"; exit 1' INT TERM
abort() { echo "[$(ts)] ABORT: $1"; exit 1; }
stop_requested() { [ -e $B/STOP ] && { echo "[$(ts)] STOP file found"; return 0; }; return 1; }
boot() { # $1 tag, $2.. flags for dt5_run_cmd.py
  local tag=$1; shift
  docker rm -f $NAME >/dev/null 2>&1
  echo "[$(ts)] host RAM available: $(free -g | awk '/Mem:/ {print $7}') GB"
  local since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[$(ts)] === boot $tag ($*) ==="
  python3 $B/dt5_run_cmd.py $INSPECT $NAME 8002 "$@" > $B/logs/run5_$tag.txt 2>&1 || { cat $B/logs/run5_$tag.txt | cut -c1-500; return 1; }
  head -1 $B/logs/run5_$tag.txt | cut -c1-2500
  docker ps -q -f name=^$NAME$ | grep -q . || return 1
  wait_ready $NAME 2700 $since || return 1
  docker logs --since $since $NAME > $B/logs/boot5_$tag.log 2>&1
  grep -h "hook3\|Auto-fit max_model_len\|GPU KV cache size" $B/logs/boot5_$tag.log | sed 's/^.*INFO //; s/^([^)]*) //' | cut -c1-200 | sort | uniq -c | head -20
  for i in $(seq 1 30); do curl -sf -o /dev/null $URL/v1/models && break; sleep 2; done
  echo "[$(ts)] $tag served: $(served 8002)"; smoke 8002
}
warm_restart() { # $1 tag
  local since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[$(ts)] --- warm restart of $NAME ($1) ---"
  docker restart $NAME >/dev/null; wait_ready $NAME 1800 $since || return 1
  docker logs --since $since $NAME > $B/logs/boot5_$1.log 2>&1
  grep -h "hook3\|Auto-fit max_model_len\|GPU KV cache size" $B/logs/boot5_$1.log | sed 's/^.*INFO //; s/^([^)]*) //' | cut -c1-200 | sort | uniq -c | head -20
  for i in $(seq 1 30); do curl -sf -o /dev/null $URL/v1/models && break; sleep 2; done
  echo "[$(ts)] $1 served: $(served 8002)"; smoke 8002
}
check_hooks() { # $1 tag: every fix must report active
  local ok=1
  for pat in "invariant Marlin MoE active" "invariant Triton GEMM in use" "QSA pinned split profile in use" "non-final prefill chunks aligned" "prefill chunk aligned:"; do
    docker logs $NAME 2>&1 | grep -q "$pat" && echo "[$(ts)] $1: '$pat' seen" || { echo "[$(ts)] $1: '$pat' NOT seen"; ok=0; }
  done
  docker logs $NAME 2>&1 | grep -q "FAILED" && { echo "[$(ts)] $1: a hook3 patch FAILED:"; docker logs $NAME 2>&1 | grep FAILED | head -3; ok=0; }
  [ $ok = 1 ]
}
kld_score() { # $1 out name
  echo "[$(ts)] KLD scoring $1 (77 windows, K=200)"
  python3 $KLD/score.py collect --url $URL/v1 --model qwen38-flash-next-awq --k 200 --out $B/results/kld5/$1.npz --windows $KLD/corpus/windows.jsonl 2>&1 | grep -E "77/77|wrote|retry|Error|Traceback" | tail -5
  [ -s $B/results/kld5/$1.npz ] && echo "[$(ts)] KLD $1 done" || { echo "[$(ts)] KLD $1 FAILED"; return 1; }
}

# ---------------- Step 0 ----------------
echo "[$(ts)] --- Step 0: preconditions ---"
docker ps --format '{{.Names}}' | grep -qx $PROD || abort "prod container $PROD is not running"
for f in hook3/sitecustomize.py hook3/inv_gemm.py hook3/inv_marlin.py dt5_run_cmd.py dt5_prefix.py dt5_prefill.py dt5_analyze.py dt5_kld_compare.py dt3_score.py dt3_decode.py dt3_bench.py tune_splitk.py build_tuned_table.py test_inv_marlin_int4.py; do [ -s $B/$f ] || abort "missing $f"; done
python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" $B/hook3/*.py $B/dt5_*.py $B/tune_splitk.py $B/build_tuned_table.py || abort syntax
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); [ "$FREE_GB" -ge 20 ] || abort "only ${FREE_GB} GB free"
docker run --rm -v vllm-hf-cache:/hf:ro --entrypoint ls $IMAGE /hf/ple-bf16-int8all-38f/config.json /hf/ple-fp8-int8all-38f/config.json /hf/$HUB_INT4/config.json >/dev/null || abort "checkpoint missing"
docker inspect $PROD > $INSPECT
POLICY=$(docker inspect $PROD --format '{{.HostConfig.RestartPolicy.Name}}')
docker volume rm $VOL >/dev/null 2>&1; docker volume create $VOL >/dev/null || abort "volume"
python3 $B/dt5_run_cmd.py $INSPECT $NAME 8002 --inv --cache-volume $VOL --arg=--max-logprobs --arg=200 --dry-run >/dev/null || abort "run cmd dry run"
rm -f $B/results/s_E3_* $B/results/s_E3b_* $B/results/kld5/*.npz $B/results/prefix_E3.json $B/results/decode_E3.json $B/results/bench_E3* $B/results/bench_Q4* $B/results/bench_FP8*
for i in $(seq 1 60); do
  busy=$(curl -s localhost:8001/metrics | awk '/^vllm:num_requests_(running|waiting)\{/ {s+=$2} END {print s+0}')
  [ "$busy" = "0" ] && break; echo "prod busy ($busy), waiting"; sleep 10
done
echo "[$(ts)] --- stopping prod (downtime starts) ---"
docker update --restart=no $PROD >/dev/null; STOPPED=1; docker stop $PROD >/dev/null
sleep 5; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo

# ---------------- Step 1: CMP micro-steps ----------------
echo "[$(ts)] --- Step 1: CMP split-K tile tuning (GPU 0, 1) + INT4 Marlin config check (GPU 3) ---"
docker run --rm --name tune-a --runtime nvidia --gpus '"device=0"' -e "KEYS=336,10240;96,2560" -v $B:/w --entrypoint python3 $IMAGE /w/tune_splitk.py /w/results/tune_splitk_cmp_a.json > $B/logs/tune_splitk_cmp_a.log 2>&1 &
PA=$!
docker run --rm --name tune-b --runtime nvidia --gpus '"device=1"' -e "KEYS=512,2560;1,2560;640,2560" -v $B:/w --entrypoint python3 $IMAGE /w/tune_splitk.py /w/results/tune_splitk_cmp_b.json > $B/logs/tune_splitk_cmp_b.log 2>&1 &
PB=$!
INT4CFG=64,128,2,128,1
for c in 64,128,2,128,2 64,128,2,128,1; do
  r=$(docker run --rm --name tune-i4 --runtime nvidia --gpus '"device=3"' -e E=128 -e CFG=$c -v $B:/w --entrypoint python3 $IMAGE /w/test_inv_marlin_int4.py 2>&1 | grep "INT4 g32" | tail -1)
  echo "[$(ts)] INT4 Marlin CFG=$c: ${r:-invalid/failed}"
  if echo "$r" | grep -q "invariant rows differing: 0 "; then INT4CFG=$c; break; fi
done
echo "[$(ts)] INT4 Marlin config: $INT4CFG"
wait $PA $PB
grep -h "done; mismatches" $B/logs/tune_splitk_cmp_a.log $B/logs/tune_splitk_cmp_b.log
if [ -s $B/results/tune_splitk_cmp_a.json ] && [ -s $B/results/tune_splitk_cmp_b.json ] && ! grep -q MISMATCH $B/logs/tune_splitk_cmp_*.log; then
  cp $B/hook3/tuned_gemm.json $B/results/tuned_gemm_before_w5.json
  python3 $B/build_tuned_table.py $B/results/tune_splitk_cmp_a.json $B/results/tune_splitk_cmp_b.json | tail -1
else
  echo "[$(ts)] split-K tuning incomplete or mismatched -> keeping existing table"
fi

# ---------------- Step 2: E3 ----------------
stop_requested && exit 0
boot E3 --inv --cache-volume $VOL --arg=--max-logprobs --arg=200 || abort "E3 boot failed"
check_hooks E3 || echo "[$(ts)] WARNING: E3 hook check incomplete (chunk-align log appears on the first chunked prompt)"
warm_restart E3w || abort "E3 warm restart failed"
E3CTX=$(served 8002 | awk '{print $2}'); echo "[$(ts)] E3 context after warm restart: $E3CTX"

# ---------------- Step 3: validation ----------------
stop_requested && exit 0
echo "[$(ts)] --- Step 3: validation on E3 ---"
python3 $B/dt5_prefill.py $URL E3
for L in 2048 2047 2040 2001 1985 1984 1983 1600 1100; do python3 $B/dt3_score.py $URL E3_L$L $L 20 || abort "scoring E3_L$L"; done
check_hooks E3
python3 $B/dt5_prefix.py $URL E3 || echo "[$(ts)] prefix test failed"
python3 $B/dt3_decode.py $URL E3 || echo "[$(ts)] decode test failed"
python3 $B/dt3_bench.py E3_a $URL; python3 $B/dt3_bench.py E3_b $URL
python3 $B/dt5_analyze.py E3 | tail -25

# ---------------- Step 4/5: KLD reference, two boots ----------------
stop_requested && exit 0
kld_score REF_a || abort "REF_a"
warm_restart E3b || abort "E3 second boot failed"
python3 $B/dt3_score.py $URL E3b_L2048 2048 20
python3 - <<EOF
import numpy as np
a, b = np.load("$B/results/s_E3_L2048.npz"), np.load("$B/results/s_E3b_L2048.npz")
print("cross-boot probe (20 windows x 2048, K=20): bit-identical:", bool(np.array_equal(a["logprobs"], b["logprobs"]) and np.array_equal(a["ids"], b["ids"])))
EOF
kld_score REF_b
docker rm -f $NAME >/dev/null

# ---------------- Step 6: INT4 ----------------
stop_requested && exit 0
if boot Q4 --inv --marlin-cfg $INT4CFG --ckpt $HUB_INT4 --cache-volume $VOL --arg=--max-logprobs --arg=200; then
  check_hooks Q4 || true
  kld_score Q4
  python3 $B/dt5_prefill.py $URL Q4; python3 $B/dt3_bench.py Q4 $URL
else
  echo "[$(ts)] Q4 boot failed, skipping"
fi
docker rm -f $NAME >/dev/null

# ---------------- Step 7: FP8 PLE ----------------
stop_requested && exit 0
if boot FP8 --inv --ckpt ple-fp8-int8all-38f --env VLLM_PLE_FP8_TABLE=1 --cache-volume $VOL --arg=--max-logprobs --arg=200; then
  check_hooks FP8 || true
  kld_score FP8
  python3 $B/dt5_prefill.py $URL FP8; python3 $B/dt3_bench.py FP8 $URL
else
  echo "[$(ts)] FP8 boot failed, skipping"
fi
docker rm -f $NAME >/dev/null
echo "[$(ts)] --- all steps done, restoring prod ---"
