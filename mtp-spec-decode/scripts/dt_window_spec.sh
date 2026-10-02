#!/bin/bash
# MTP spec-decode window: validate S0 (patched image, all switches off) -> S1 fixes + A -> S2 A+B (tau 1.0 / 0.8)
# -> S3 A+B+C in a test container on :8002 with prod stopped, then deploy the last stage that passed every gate as
# prod (old container kept stopped as qwen38-flash-next-pp3-int8all-prespec) and re-validate prod against the baseline.
# Gates per stage (mtp/stage_check.py): KLD corpus logprobs bit-identical to the prod baseline, greedy outputs
# identical, T=1 calibration of sampled tokens, padded-row '!' count. Owner OK'd downtime on 2026-10-02.
set -u
M=/home/user/qwen3nextflash/mtp; B=/home/user/qwen3nextflash/batchinv; KLD=/home/user/qwen3nextflash/kld
LOG=$M/logs/dt_window_spec.log
PROD=qwen38-flash-next-pp3-int8all; OLD=qwen38-flash-next-pp3-int8all-prespec; TEST=qwen38-spec-test
IMG=qwen38-flash-next:ple-fp8-pp3-detmoe-inv-det-spec
VOL=vllm-cache-spec; PRODVOL=vllm-cache-prod-det
INSPECT=$M/logs/spec_prod_inspect_before.json
exec > >(tee -a $LOG) 2>&1
ts() { date -u +%H:%M:%S; }
echo "=== DT WINDOW SPEC START $(date -u) ==="
SPEC0='{"method":"mtp","num_speculative_tokens":3}'
SPECA='{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'
SPECC='{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
ENV_A="VLLM_SPEC_REJECT_UNPROPOSED=1 VLLM_SPEC_DRAFT_INDEP_NOISE=1"

wait_ready() { # $1 container, $2 max s, $3 since
  for i in $(seq 1 $(($2 / 10))); do
    if docker logs --since $3 $1 2>&1 | grep -q "Starting vLLM server"; then echo "[$(ts)] $1 ready after $((i * 10))s"; sleep 5; return 0; fi
    if ! docker ps -q -f name=^$1$ | grep -q .; then echo "[$(ts)] $1 exited"; docker logs --since $3 $1 2>&1 | grep -E "Error|Traceback|spec-" | tail -40; return 1; fi
    sleep 10
  done
  echo "[$(ts)] $1 not ready after $2 s"; docker logs --since $3 $1 2>&1 | tail -30; return 1
}
served() { curl -s localhost:$1/v1/models | python3 -c "import json,sys; d=json.load(sys.stdin)['data'][0]; print(d['id'], d.get('max_model_len'))" 2>/dev/null; }
smoke() { curl -s localhost:$1/v1/completions -H 'Content-Type: application/json' -d '{"model":"qwen38-flash-next-awq","prompt":"The capital of France is","max_tokens":4,"temperature":0}' | python3 -c "import json,sys; print('smoke:', repr(json.load(sys.stdin)['choices'][0]['text']))"; }
up() { for i in $(seq 1 60); do curl -sf -o /dev/null localhost:$1/v1/models && return 0; sleep 5; done; return 1; }

DONE=0; STOPPED=0; SWAPPED=0
rollback() {
  [ "$DONE" = 1 ] && return 0
  echo "[$(ts)] --- ROLLBACK ---"
  docker rm -f $TEST >/dev/null 2>&1
  if [ "$SWAPPED" = 1 ]; then
    docker rm -f $PROD >/dev/null 2>&1; docker rename $OLD $PROD
  fi
  if [ "$STOPPED" = 1 ] || [ "$SWAPPED" = 1 ]; then
    docker update --restart=always $PROD >/dev/null
    local since=$(date -u +%Y-%m-%dT%H:%M:%SZ); docker start $PROD >/dev/null; wait_ready $PROD 1500 $since; up 8001
    echo "[$(ts)] prod served: $(served 8001)"; smoke 8001
  fi
  echo "=== DT WINDOW SPEC: ROLLED BACK $(date -u) ==="
}
trap rollback EXIT
trap 'echo "[$(ts)] trapped signal"; exit 1' INT TERM
abort() { echo "[$(ts)] ABORT: $1"; exit 1; }

declare -A PASS
run_stage() { # TAG SPECJSON [ENV=V ...]
  local tag=$1 spec=$2; shift 2
  local envargs=(); for e in "$@"; do envargs+=(--env "$e"); done
  echo "[$(ts)] ===== stage $tag  spec=$spec  env=$*"
  docker rm -f $TEST >/dev/null 2>&1
  local since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  python3 $B/dt5_run_cmd.py $INSPECT $TEST 8002 --image-hook --image $IMG --cache-volume $VOL --set-arg --speculative-config "$spec" "${envargs[@]}" > $M/logs/run_$tag.txt 2>&1 \
    || { cut -c1-800 $M/logs/run_$tag.txt; PASS[$tag]=0; return 1; }
  wait_ready $TEST 2700 $since || { PASS[$tag]=0; return 1; }
  up 8002 || { PASS[$tag]=0; return 1; }
  echo "[$(ts)] $tag served: $(served 8002)"; smoke 8002
  docker logs $TEST 2>&1 | grep -h "spec-draft\|Speculative\|speculative_config\|hook3\]" | sed 's/^.*INFO //' | cut -c1-260 | sort | uniq -c | head -8
  python3 $KLD/score.py collect --url http://127.0.0.1:8002/v1 --model qwen38-flash-next-awq --k 20 --out $M/results/${tag}_k20.npz --windows $KLD/corpus/windows.jsonl 2>&1 | grep -E "77/77|wrote|retry|Error|Traceback" | tail -3
  python3 $M/gen_suite.py http://127.0.0.1:8002 $tag 2>&1 | grep -v "^  t1\|^  pad" | tail -45
  echo "[$(ts)] $tag spec-fix log: $(docker logs $TEST 2>&1 | grep 'spec-fix(#58784)' | tail -1 | sed 's/^.*INFO //')"
  docker logs $TEST 2>&1 | grep -E "Traceback|ERROR" | head -5
  if python3 $M/stage_check.py $tag; then PASS[$tag]=1; else PASS[$tag]=0; fi
  echo "[$(ts)] ===== stage $tag PASS=${PASS[$tag]}"
  docker stop $TEST >/dev/null
}
acc_len() { python3 -c "import json; g=json.load(open('$M/results/gen_$1.json'))['t1']; print(round((g.get('clean') or g['all'])['mean_accept_len'], 4))"; }

# ---------------- Step 0: preconditions (prod still up) ----------------
echo "[$(ts)] --- Step 0: preconditions ---"
[ -s $M/results/PROD_base_k20.npz ] && [ -s $M/results/gen_PROD_base.json ] || abort "baseline missing"
docker ps --format '{{.Names}}' | grep -qx $PROD || abort "prod not running"
docker ps -a --format '{{.Names}}' | grep -qx $OLD && abort "$OLD already exists"
docker image inspect $IMG >/dev/null || abort "image missing"
docker inspect $PROD > $INSPECT
python3 $B/dt5_run_cmd.py $INSPECT $TEST 8002 --image-hook --image $IMG --cache-volume $VOL --set-arg --speculative-config "$SPEC0" --dry-run > /dev/null || abort "dry run"
docker volume rm $VOL >/dev/null 2>&1; docker volume create $VOL >/dev/null
docker run --rm -v $PRODVOL:/s:ro -v $VOL:/d --entrypoint sh $IMG -c "cp -a /s/. /d/" || abort "cache copy"
for i in $(seq 1 60); do
  busy=$(curl -s localhost:8001/metrics | awk '/^vllm:num_requests_(running|waiting)\{/ {s+=$2} END {print s+0}')
  [ "$busy" = "0" ] && break; echo "[$(ts)] prod busy ($busy), waiting"; sleep 10
done

# ---------------- Step 1: stop prod ----------------
echo "[$(ts)] --- Step 1: stop prod ---"
docker update --restart=no $PROD >/dev/null; docker stop $PROD >/dev/null; STOPPED=1; sleep 5

# ---------------- Step 2: stages ----------------
run_stage S0_stock "$SPEC0" || abort "S0 failed to run"
[ "${PASS[S0_stock]}" = 1 ] || abort "S0 (switches off) does not reproduce the prod baseline"
run_stage S1_A "$SPECA" $ENV_A || true
if [ "${PASS[S1_A]}" = 1 ]; then
  run_stage S2_AB_t10 "$SPECA" $ENV_A VLLM_SPEC_DRAFT_TOPKP=1 || true
  run_stage S2_AB_t08 "$SPECA" $ENV_A VLLM_SPEC_DRAFT_TOPKP=1 VLLM_SPEC_DRAFT_TAU=0.8 || true
fi
BEST_B=""; BEST_TAU=1.0
if [ "${PASS[S2_AB_t10]:-0}" = 1 ] || [ "${PASS[S2_AB_t08]:-0}" = 1 ]; then
  a10=0; a08=0
  [ "${PASS[S2_AB_t10]:-0}" = 1 ] && a10=$(acc_len S2_AB_t10)
  [ "${PASS[S2_AB_t08]:-0}" = 1 ] && a08=$(acc_len S2_AB_t08)
  echo "[$(ts)] B tau choice: mean accept len tau1.0=$a10 tau0.8=$a08"
  if python3 -c "import sys; sys.exit(0 if $a08 > $a10 else 1)"; then BEST_B=S2_AB_t08; BEST_TAU=0.8; else BEST_B=S2_AB_t10; BEST_TAU=1.0; fi
  run_stage S3_ABC "$SPECC" $ENV_A VLLM_SPEC_DRAFT_TOPKP=1 VLLM_SPEC_DRAFT_TAU=$BEST_TAU VLLM_SPEC_STEP_UNIQUE_RNG=1 || true
fi
docker rm -f $TEST >/dev/null 2>&1

# pick the final config: the furthest stage that passed
FINAL=""; FSPEC=""; FENV=""
if [ "${PASS[S3_ABC]:-0}" = 1 ]; then FINAL=S3_ABC; FSPEC=$SPECC; FENV="$ENV_A VLLM_SPEC_DRAFT_TOPKP=1 VLLM_SPEC_DRAFT_TAU=$BEST_TAU VLLM_SPEC_STEP_UNIQUE_RNG=1"
elif [ -n "$BEST_B" ]; then FINAL=$BEST_B; FSPEC=$SPECA; FENV="$ENV_A VLLM_SPEC_DRAFT_TOPKP=1 VLLM_SPEC_DRAFT_TAU=$BEST_TAU"
elif [ "${PASS[S1_A]:-0}" = 1 ]; then FINAL=S1_A; FSPEC=$SPECA; FENV="$ENV_A"
fi
echo "[$(ts)] stage results: $(for k in "${!PASS[@]}"; do echo -n "$k=${PASS[$k]} "; done)"
[ -n "$FINAL" ] || abort "no stage beyond S0 passed; keeping the old prod"
echo "[$(ts)] final config: $FINAL  spec=$FSPEC  env=$FENV"

# ---------------- Step 3: deploy as prod ----------------
echo "[$(ts)] --- Step 3: deploy $FINAL as prod (old container kept as $OLD) ---"
docker rename $PROD $OLD; SWAPPED=1
envargs=(); for e in $FENV; do envargs+=(--env "$e"); done
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 $B/dt5_run_cmd.py $INSPECT $PROD 8001 --image-hook --image $IMG --cache-volume $VOL --restart always --set-arg --speculative-config "$FSPEC" "${envargs[@]}" > $M/logs/run_prod.txt 2>&1 || { cut -c1-800 $M/logs/run_prod.txt; abort "docker run prod"; }
wait_ready $PROD 2700 $since || abort "prod boot"; up 8001 || abort "prod api"
CTX=$(served 8001 | awk '{print $2}')
if [ "$CTX" != "524288" ]; then
  echo "[$(ts)] context $CTX after first boot -> warm restart"
  since=$(date -u +%Y-%m-%dT%H:%M:%SZ); docker restart $PROD >/dev/null; wait_ready $PROD 2700 $since || abort "warm restart"; up 8001
  CTX=$(served 8001 | awk '{print $2}')
fi
[ "$CTX" = "524288" ] || abort "prod context $CTX != 524288"
echo "[$(ts)] prod served: $(served 8001)"; smoke 8001
docker logs $PROD 2>&1 | grep -h "spec-draft" | sed 's/^.*INFO //' | sort | uniq -c
python3 $KLD/score.py collect --url http://127.0.0.1:8001/v1 --model qwen38-flash-next-awq --k 20 --out $M/results/PROD_new_k20.npz --windows $KLD/corpus/windows.jsonl 2>&1 | grep -E "77/77|wrote|retry|Error|Traceback" | tail -3
python3 $M/gen_suite.py http://127.0.0.1:8001 PROD_new 2>&1 | grep -v "^  t1\|^  pad" | tail -45
echo "[$(ts)] prod spec-fix log: $(docker logs $PROD 2>&1 | grep 'spec-fix(#58784)' | tail -1 | sed 's/^.*INFO //')"
python3 $M/stage_check.py PROD_new || abort "deployed prod failed validation"
echo "[$(ts)] prod: $(docker inspect $PROD --format '{{.Config.Image}} restart={{.HostConfig.RestartPolicy.Name}}') served: $(served 8001)"
echo "[$(ts)] rollback container: $(docker inspect $OLD --format '{{.Name}} {{.State.Status}} restart={{.HostConfig.RestartPolicy.Name}}')"
DONE=1
echo "=== DT WINDOW SPEC: $FINAL deployed as prod $(date -u) ==="
