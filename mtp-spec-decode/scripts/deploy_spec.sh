#!/bin/bash
# Deploy the chosen spec-decode config as prod (owner's choice 2026-10-02: A + B, draft tau 1.0, no C).
# Old prod container kept stopped as qwen38-flash-next-pp3-int8all-prespec; any failure -> automatic swap back.
# Post-deploy validation vs the morning baseline: KLD corpus bit-identical, greedy identical, calibration, pad trials.
set -u
M=/home/user/qwen3nextflash/mtp; B=/home/user/qwen3nextflash/batchinv; KLD=/home/user/qwen3nextflash/kld
LOG=$M/logs/deploy_spec.log
PROD=qwen38-flash-next-pp3-int8all; OLD=qwen38-flash-next-pp3-int8all-prespec; TEST=qwen38-spec-test
IMG=qwen38-flash-next:ple-fp8-pp3-detmoe-inv-det-spec; VOL=vllm-cache-spec
INSPECT=$M/logs/spec_prod_inspect_before.json
FSPEC='{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'
FENV="VLLM_SPEC_REJECT_UNPROPOSED=1 VLLM_SPEC_DRAFT_INDEP_NOISE=1 VLLM_SPEC_DRAFT_TOPKP=1 VLLM_SPEC_DRAFT_TAU=1.0 VLLM_SPEC_PAD_NEW_REQS=0"
TAG=${TAG:-PROD_new2}
exec > >(tee -a $LOG) 2>&1
ts() { date -u +%H:%M:%S; }
echo "=== DEPLOY SPEC START $(date -u) ==="
wait_ready() {
  for i in $(seq 1 $(($2 / 10))); do
    if docker logs --since $3 $1 2>&1 | grep -q "Starting vLLM server"; then echo "[$(ts)] $1 ready after $((i * 10))s"; sleep 5; return 0; fi
    if ! docker ps -q -f name=^$1$ | grep -q .; then echo "[$(ts)] $1 exited"; docker logs --since $3 $1 2>&1 | grep -E "Error|Traceback|spec-" | tail -40; return 1; fi
    sleep 10
  done
  echo "[$(ts)] $1 not ready after $2 s"; return 1
}
served() { curl -s localhost:8001/v1/models | python3 -c "import json,sys; d=json.load(sys.stdin)['data'][0]; print(d['id'], d.get('max_model_len'))" 2>/dev/null; }
smoke() { curl -s localhost:8001/v1/completions -H 'Content-Type: application/json' -d '{"model":"qwen38-flash-next-awq","prompt":"The capital of France is","max_tokens":4,"temperature":0}' | python3 -c "import json,sys; print('smoke:', repr(json.load(sys.stdin)['choices'][0]['text']))"; }
up() { for i in $(seq 1 60); do curl -sf -o /dev/null localhost:8001/v1/models && return 0; sleep 5; done; return 1; }
DONE=0; SWAPPED=0
rollback() {
  [ "$DONE" = 1 ] && return 0
  echo "[$(ts)] --- ROLLBACK to the pre-spec container ---"
  if [ "$SWAPPED" = 1 ]; then docker rm -f $PROD >/dev/null 2>&1; docker rename $OLD $PROD; fi
  docker update --restart=always $PROD >/dev/null
  local since=$(date -u +%Y-%m-%dT%H:%M:%SZ); docker start $PROD >/dev/null; wait_ready $PROD 1500 $since; up
  echo "[$(ts)] prod served: $(served)"; smoke
  echo "=== DEPLOY SPEC: ROLLED BACK $(date -u) ==="
}
trap rollback EXIT
trap 'echo "[$(ts)] trapped signal"; exit 1' INT TERM
abort() { echo "[$(ts)] ABORT: $1"; exit 1; }

docker rm -f $TEST >/dev/null 2>&1
docker ps -a --format '{{.Names}}' | grep -qx $OLD && abort "$OLD already exists"
if [ "$(docker inspect $PROD --format '{{.State.Status}}')" = "running" ]; then
  docker inspect $PROD > $M/logs/spec_prod_inspect_before2.json
  [ "$(docker inspect $PROD --format '{{.Config.Image}}')" = "qwen38-flash-next:ple-fp8-pp3-detmoe-inv-det" ] || abort "prod is not the pre-spec image"
  for i in $(seq 1 60); do
    busy=$(curl -s localhost:8001/metrics | awk '/^vllm:num_requests_(running|waiting)\{/ {s+=$2} END {print s+0}')
    [ "$busy" = "0" ] && break; echo "[$(ts)] prod busy ($busy), waiting"; sleep 10
  done
  echo "[$(ts)] stopping pre-spec prod"; docker update --restart=no $PROD >/dev/null; docker stop $PROD >/dev/null
fi
[ "$(docker inspect $PROD --format '{{.State.Status}}')" = "exited" ] || abort "old prod is not stopped"
docker rename $PROD $OLD; docker update --restart=no $OLD >/dev/null; SWAPPED=1
envargs=(); for e in $FENV; do envargs+=(--env "$e"); done
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 $B/dt5_run_cmd.py $INSPECT $PROD 8001 --image-hook --image $IMG --cache-volume $VOL --restart always --set-arg --speculative-config "$FSPEC" "${envargs[@]}" > $M/logs/run_prod.txt 2>&1 || { cut -c1-800 $M/logs/run_prod.txt; abort "docker run prod"; }
wait_ready $PROD 2700 $since || abort "prod boot"; up || abort "prod api"
CTX=$(served | awk '{print $2}')
if [ "$CTX" != "524288" ]; then
  echo "[$(ts)] context $CTX after first boot -> warm restart"
  since=$(date -u +%Y-%m-%dT%H:%M:%SZ); docker restart $PROD >/dev/null; wait_ready $PROD 2700 $since || abort "warm restart"; up
  CTX=$(served | awk '{print $2}')
fi
[ "$CTX" = "524288" ] || abort "prod context $CTX != 524288"
echo "[$(ts)] PROD UP: $(served)"; smoke
docker logs $PROD 2>&1 | grep -h "spec-draft" | sed 's/^.*INFO //' | sort | uniq -c
echo "[$(ts)] --- validation vs PROD_base ---"
python3 $KLD/score.py collect --url http://127.0.0.1:8001/v1 --model qwen38-flash-next-awq --k 20 --out $M/results/${TAG}_k20.npz --windows $KLD/corpus/windows.jsonl 2>&1 | grep -E "77/77|wrote|retry|Error|Traceback" | tail -3
python3 $M/gen_suite.py http://127.0.0.1:8001 $TAG 2>&1 | grep -v "^  t1\|^  cal\|^  pad" | tail -45
echo "[$(ts)] prod spec-fix log: $(docker logs $PROD 2>&1 | grep 'spec-fix(#58784)' | tail -1 | sed 's/^.*INFO //')"
python3 $M/stage_check.py $TAG || abort "deployed prod failed validation"
echo "[$(ts)] prod: $(docker inspect $PROD --format '{{.Config.Image}} restart={{.HostConfig.RestartPolicy.Name}}') served: $(served)"
echo "[$(ts)] rollback container: $(docker inspect $OLD --format '{{.Name}} {{.State.Status}} restart={{.HostConfig.RestartPolicy.Name}}')"
DONE=1
echo "=== DEPLOY SPEC: A+B tau 1.0 (+ no new-request padding) deployed and validated $(date -u) ==="
