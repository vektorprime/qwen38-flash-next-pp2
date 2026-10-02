#!/bin/bash
# Batch-invariance downtime window (owner approved downtime 2026-10-02). Prod stack, eval containers on port 8002.
#   0  preconditions; stop prod
#   1  cost/invariance bench on a CMP 170HX (GPU 0): Marlin MoE stock vs invariant, invariant GEMM vs cuBLAS
#   2  E1 = prod + invariant Marlin MoE (hook3, PLEFP8_INV_MOE=1) + control hooks
#        prompt-length invariance, mechanism: 1-ulp injections (dose-response), routing capture, routing replay,
#        decode reproducibility under concurrency, decode bench
#   3  E2 = prod + invariant MoE + invariant dense GEMM + pinned QSA split (fresh compile cache)
#        prompt-length invariance (all positions), decode reproducibility, decode bench x2
#   4  restore prod exactly (trap does this on any exit), verify config + context + smoke; offline analysis
# touch batchinv/STOP to skip remaining steps and restore early.
set -u
B=/home/user/qwen3nextflash/batchinv
E2E=/home/user/qwen3nextflash/plefp8/e2e
LOG=$B/logs/dt_window3.log
PROD=${PROD:-qwen38-flash-next-pp3-int8all}
IMAGE=qwen38-flash-next:ple-fp8-pp3-detmoe
NAME=plefp8-dt; URL=http://127.0.0.1:8002
CTRL=$B/ctrl
mkdir -p $B/logs $B/results $CTRL/route $CTRL/replay
exec > >(tee -a $LOG) 2>&1
ts() { date -u +%H:%M:%S; }
echo "=== DT WINDOW 3 START $(date -u) ==="

wait_ready() { # $1 container, $2 max s, $3 since
  for i in $(seq 1 $(($2 / 10))); do
    if docker logs --since $3 $1 2>&1 | grep -q "Starting vLLM server"; then echo "[$(ts)] $1 ready after $((i * 10))s"; sleep 5; return 0; fi
    if ! docker ps -q -f name=^$1$ | grep -q .; then echo "[$(ts)] $1 exited"; docker logs --since $3 $1 2>&1 | grep -E "Error|Traceback|hook3" | tail -40; return 1; fi
    sleep 10
  done
  echo "[$(ts)] $1 not ready after $2 s"; docker logs --since $3 $1 2>&1 | tail -30; return 1
}
served() { curl -s localhost:$1/v1/models | python3 -c "import json,sys; d=json.load(sys.stdin)['data'][0]; print(d['id'], d.get('max_model_len'))" 2>/dev/null; }
RESTORED=0; STOPPED=0
restore_prod() {
  [ "$RESTORED" = 1 ] && return 0; RESTORED=1
  docker rm -f $NAME >/dev/null 2>&1; rm -f $CTRL/route/ON $CTRL/replay/ON $CTRL/inject.json
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
  docker inspect $PROD > $B/logs/prod_inspect_after.json
  python3 - $B/logs/prod_inspect_before.json $B/logs/prod_inspect_after.json <<'EOF'
import json, sys
a, b = json.load(open(sys.argv[1]))[0], json.load(open(sys.argv[2]))[0]
keys = [("Config", "Image"), ("Config", "Cmd"), ("Config", "Entrypoint"), ("Config", "Env"), ("HostConfig", "RestartPolicy"),
        ("HostConfig", "DeviceRequests"), ("HostConfig", "PortBindings"), ("Image",), ("Id",)]
g = lambda d, k: [d := (d.get(x) if isinstance(d, dict) else None) for x in k][-1]
diff = [".".join(k) for k in keys if g(a, k) != g(b, k)]
print("prod config identical to before:", not diff, diff, "| state:", b["State"]["Status"])
EOF
  curl -s localhost:8001/v1/completions -H 'Content-Type: application/json' -d '{"model":"qwen38-flash-next-awq","prompt":"The capital of France is","max_tokens":4,"temperature":0}' | python3 -c "import json,sys; print('smoke:', json.load(sys.stdin)['choices'][0]['text'])"
  echo "=== DT WINDOW 3: prod restored $(date -u) ==="
}
trap 'restore_prod' EXIT
trap 'echo "[$(ts)] trapped signal"; exit 1' INT TERM
abort() { echo "[$(ts)] ABORT: $1"; exit 1; }
stop_requested() { [ -e $B/STOP ] && { echo "[$(ts)] STOP file found"; return 0; }; return 1; }
score() { python3 $B/dt3_score.py $URL $1 $2 $3 || abort "scoring $1 failed"; }   # tag L K
cap_on() { echo "$1" > $CTRL/route/TAG; touch $CTRL/route/ON; }
cap_off() { rm -f $CTRL/route/ON; sleep 1; }
inject() { echo "$1" > $CTRL/inject.json.tmp && mv $CTRL/inject.json.tmp $CTRL/inject.json; sleep 1; }
no_inject() { rm -f $CTRL/inject.json; sleep 1; }
boot() { # $1 tag, $2.. flags for dt3_run_cmd.py
  local tag=$1; shift
  docker rm -f $NAME >/dev/null 2>&1
  local since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[$(ts)] === boot $tag ($*) ==="
  python3 $B/dt3_run_cmd.py $B/logs/prod_inspect_before.json $NAME 8002 "$@" >/dev/null || return 1
  wait_ready $NAME 2400 $since || return 1
  docker logs --since $since $NAME > $B/logs/boot_$tag.log 2>&1
  grep -h "hook3\|Auto-fit max_model_len\|GPU KV cache size" $B/logs/boot_$tag.log | sed 's/^.*INFO //; s/^([^)]*) //' | cut -c1-170 | sort | uniq -c | head -20
  for i in $(seq 1 30); do curl -sf -o /dev/null $URL/v1/models && break; sleep 2; done
  echo "[$(ts)] $tag served: $(served 8002)"
}
check_banner() { # $1 tag, $2 pattern, $3 what
  docker logs $NAME 2>&1 | grep -q "$2" && echo "[$(ts)] $1: $3 active" || { echo "[$(ts)] $1: $3 NOT seen in logs"; return 1; }
}

# ---------------- Step 0 ----------------
echo "[$(ts)] --- Step 0: preconditions ---"
docker ps --format '{{.Names}}' | grep -qx $PROD || abort "prod container $PROD is not running"
for f in hook3/sitecustomize.py hook3/inv_gemm.py dt3_run_cmd.py dt3_score.py dt3_decode.py dt3_bench.py dt3_cmp_bench.py dt3_analyze.py; do [ -s $B/$f ] || abort "missing $f"; done
[ -s $E2E/results/probe_windows.jsonl ] && [ -s $E2E/results/probe_rep_k20.npz ] || abort "probe windows / prod reference missing"
python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" $B/hook3/*.py $B/dt3_*.py || abort syntax
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); [ "$FREE_GB" -ge 20 ] || abort "only ${FREE_GB} GB free"
rm -rf $CTRL/route/E1_* $CTRL/route/E2_* $CTRL/route/ON $CTRL/route/TAG $CTRL/replay/* $CTRL/inject.json $B/results/s_E1_* $B/results/s_E2_*
rm -rf $B/vcache_E2; mkdir -p $B/vcache_E2; chmod 777 $B/vcache_E2 $CTRL $CTRL/route $CTRL/replay
docker inspect $PROD > $B/logs/prod_inspect_before.json
POLICY=$(docker inspect $PROD --format '{{.HostConfig.RestartPolicy.Name}}')
python3 $B/dt3_run_cmd.py $B/logs/prod_inspect_before.json $NAME 8002 --inv-moe --dry-run >/dev/null || abort "run cmd dry run"
for i in $(seq 1 60); do
  busy=$(curl -s localhost:8001/metrics | awk '/^vllm:num_requests_(running|waiting)\{/ {s+=$2} END {print s+0}')
  [ "$busy" = "0" ] && break; echo "prod busy ($busy), waiting"; sleep 10
done
echo "[$(ts)] --- stopping prod (downtime starts) ---"
docker update --restart=no $PROD >/dev/null; STOPPED=1; docker stop $PROD >/dev/null
sleep 5; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo

# ---------------- Step 1: CMP bench ----------------
echo "[$(ts)] --- Step 1: CMP 170HX cost/invariance bench (GPU 0) ---"
docker run --rm --runtime nvidia --gpus '"device=0"' -v $B:/w --entrypoint python3 $IMAGE /w/dt3_cmp_bench.py /w/results/cmp_bench_cmp.json 2>&1 | grep -v "Warning\|warn(\|INFO" | grep "marlin\|totals" || echo "CMP bench failed (continuing)"

# ---------------- Step 2: E1 ----------------
stop_requested && exit 0
if boot E1 --inv-moe; then
  check_banner E1 "invariant Marlin MoE active" "invariant Marlin"
  cap_on E1_c2048; score E1_L2048 2048 20; cap_off
  cap_on E1_c2047; score E1_L2047 2047 20; cap_off
  for L in 2040 2032 1984 1536 1024; do score E1_L$L $L 1; done
  for d in 'D2_L0_f0.0005|{"layers": [0], "frac": 0.0005, "cols": 16, "seed": 1}' \
           'D1_L0_f0.006|{"layers": [0], "frac": 0.006, "cols": 16, "seed": 1}' \
           'D4_L47_f0.006|{"layers": [47], "frac": 0.006, "cols": 16, "seed": 1}' \
           'D3_all_f0.006|{"layers": "all", "frac": 0.006, "cols": 16, "seed": 1}'; do
    t=${d%%|*}; inject "${d#*|}"; cap_on E1_inj_$t; score E1_inj_$t 2048 20; cap_off; no_inject
  done
  docker logs $NAME 2>&1 | grep "inject config loaded" | sed 's/^([^)]*) //' | sort | uniq -c | head
  for t in D1_L0_f0.006 D3_all_f0.006; do
    cfg=$([ $t = D1_L0_f0.006 ] && echo '{"layers": [0], "frac": 0.006, "cols": 16, "seed": 1}' || echo '{"layers": "all", "frac": 0.006, "cols": 16, "seed": 1}')
    inject "$cfg"; echo E1_c2048 > $CTRL/replay/SRC; touch $CTRL/replay/ON
    score E1_inj_${t}_replay 2048 20
    rm -f $CTRL/replay/ON; no_inject
  done
  docker logs $NAME 2>&1 | grep "replay" | sed 's/^([^)]*) //' | sort | uniq -c | head
  stop_requested || python3 $B/dt3_decode.py $URL E1_inv_moe
  stop_requested || (cd $B && python3 $B/dt3_bench.py E1_inv_moe $URL)
else
  echo "[$(ts)] E1 boot failed"
fi
docker rm -f $NAME >/dev/null 2>&1; sleep 3

# ---------------- Step 3: E2 ----------------
if ! stop_requested && boot E2 --inv-moe --inv-gemm --inv-qsa --fresh-cache $B/vcache_E2; then
  check_banner E2 "invariant Marlin MoE active" "invariant Marlin"
  check_banner E2 "invariant Triton GEMM in use" "invariant GEMM"
  check_banner E2 "QSA pinned split profile in use" "pinned QSA"
  score E2_L2048 2048 20; score E2_L2047 2047 20
  for L in 2040 2032 1984 1536 1024; do score E2_L$L $L 1; done
  stop_requested || python3 $B/dt3_decode.py $URL E2_full_invariant
  stop_requested || (cd $B && python3 $B/dt3_bench.py E2_full_invariant_1 $URL && python3 $B/dt3_bench.py E2_full_invariant_2 $URL)
else
  echo "[$(ts)] E2 skipped or boot failed"
fi
docker rm -f $NAME >/dev/null 2>&1; sleep 3

# ---------------- Step 4 ----------------
restore_prod
echo "=== DT WINDOW 3 DOWNTIME OVER $(date -u) ==="
python3 $B/dt3_analyze.py
echo "=== DT WINDOW 3 ALL DONE $(date -u) ==="
